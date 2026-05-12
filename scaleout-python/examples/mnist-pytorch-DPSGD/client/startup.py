import os
import sys
import math
import yaml

import torch
import numpy as np

from data import load_data, prepare_data
from model import load_parameters, save_parameters, compile_model

from scaleout import EdgeClient
from scaleoututil.utils.model import ScaleoutModel
from torch.utils.data import TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
}


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def startup(client: EdgeClient):
    """Entry point called by Scaleout Edge."""
    prepare_data()
    MyClient(client)


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initiate_dp()

    def initiate_dp(self):

        settings_path = os.getenv("CLIENT_SETTINGS_PATH", os.path.join(self.client.package_path, "client_settings.yaml"))
        with open(settings_path, "r") as fh:
            try:
                self.dp_settings = yaml.safe_load(fh)

            except yaml.YAMLError as exc:
                print(exc)
        self.dp_settings["delta"] = float(self.dp_settings["delta"])
        self._create_train_loader()

        # Build model
        model = compile_model()
        model.to(self.device)
        model.train()

        opt_cfg = self.dp_settings["optimizer"]

        opt_name = opt_cfg["name"].lower()

        if opt_name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        optimizer = OPTIMIZERS[opt_name](
            model.parameters(),
            lr=opt_cfg.get("lr", 0.001),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

        self.privacy_engine = PrivacyEngine(accountant=self.dp_settings["accountant"])
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.dp_settings["epochs"] * self.dp_settings["global_rounds"],
            target_epsilon=self.dp_settings["epsilon"],
            target_delta=float(self.dp_settings["delta"]),
            max_grad_norm=self.dp_settings["max_grad_norm"],
            alphas=alphas,
        )

        # Track privacy budget spent over rounds
        self.round_index = 0
        self.epsilon_spent = {}

    def _create_train_loader(self, data_path=None):

        # Load data
        x_train, y_train = load_data(data_path)
        y_train = y_train.long()
        self.n_samples = x_train.shape[0]

        trainset = TensorDataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.dp_settings["batch_size"], num_workers=2)

    def train_dp(self):

        self.model.train()
        criterion = torch.nn.NLLLoss()

        data_points = 0
        with BatchMemoryManager(
            data_loader=self.train_loader, max_physical_batch_size=self.dp_settings["max_physical_batch_size"], optimizer=self.optimizer
        ) as memory_safe_data_loader:
            for epoch in range(self.dp_settings["epochs"]):
                for i, (images, target) in enumerate(memory_safe_data_loader):
                    if images.shape[0] == 0:
                        continue
                    self.optimizer.zero_grad()

                    data_points += images.shape[0]
                    images = images.to(self.device)
                    target = target.to(self.device)

                    # compute output
                    output = self.model(images)
                    loss = criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

    def train(
        self,
        scaleout_model: ScaleoutModel,
        settings,
        data_path=None,
        batch_size=32,
        epochs=1,
        lr=0.01,
    ):
        """Complete a model update.

        Load model paramters from ScaleoutModel (managed by the Scaleout client),
        perform a model update, and return updated parameters wrapped in a
        ScaleoutModel together with training metadata.

        :param scaleout_model: The incoming model parameters.
        :type scaleout_model: ScaleoutModel
        :param settings: Client settings (currently unused).
        :type settings: dict
        :param data_path: The path to the data file.
        :type data_path: str
        :param batch_size: The batch size to use.
        :type batch_size: int
        :param epochs: The number of epochs to train.
        :type epochs: int
        :param lr: The learning rate to use.
        :type lr: float
        """

        import time

        t0 = time.time()

        load_parameters(self.model, scaleout_model)
        self.round_index += 1

        self.train_dp()

        # Log the privacy budget spent so far
        try:
            self.epsilon_spent[self.round_index] = self.privacy_engine.get_epsilon(self.dp_settings["delta"])
            print(f"Epsilon after training {self.round_index} round" + ("s" if self.round_index > 1 else "") + f": {self.epsilon_spent[self.round_index]}")
        except ValueError:
            print("cant calculate epsilon")
        # Metadata needed for aggregation server side
        if self.epsilon_spent[self.round_index] > self.dp_settings["epsilon"] and self.dp_settings["hardlimit"]:
            print("Epsilon too high, not saving model")
            return None, {}

        metadata = {
            "num_examples": int(self.n_samples),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
        }

        # Save model update (mandatory)
        result_model = save_parameters(self.model)
        t1 = time.time()
        print("elapsed time for round ", self.round_index, ": ", t1 - t0, " seconds")

        return result_model, {"training_metadata": metadata}

    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Validate model.

        :param scaleout_model: The incoming model parameters.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        :return: A JSON-serializable report dict.
        :rtype: dict
        """
        # Load data
        x_train, y_train = load_data(data_path)
        x_test, y_test = load_data(data_path, is_train=False)

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device).long()
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device).long()

        # Load model
        model = compile_model()
        load_parameters(model, scaleout_model)
        model.to(self.device)
        model.eval()

        criterion = torch.nn.NLLLoss()

        with torch.no_grad():
            train_out = model(x_train)
            training_loss = criterion(train_out, y_train)
            training_preds = torch.argmax(train_out, dim=1)
            training_accuracy = (training_preds == y_train).float().mean()

            test_out = model(x_test)
            test_loss = criterion(test_out, y_test)
            test_preds = torch.argmax(test_out, dim=1)
            test_accuracy = (test_preds == y_test).float().mean()

        report = {
            "training_loss": float(training_loss.item()),
            "training_accuracy": float(training_accuracy.item()),
            "test_loss": float(test_loss.item()),
            "test_accuracy": float(test_accuracy.item()),
        }

        return report

