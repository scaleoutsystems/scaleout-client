import os
import random
import sys
import math
import threading
import time

import torch
import numpy as np

from data import load_data, prepare_data
from model import load_parameters, save_parameters

from scaleout import EdgeClient
from scaleoututil.utils.model import ScaleoutModel

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def startup(client: EdgeClient):
    """Entry point called by Scaleout Edge."""
    prepare_data()
    client = MyClient(client)
    client.start_log_attributes()


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)

    def start_log_attributes(self):
        """Start a background thread that logs attributes periodically."""
        t = threading.Thread(target=self._log_loop, daemon=True)
        t.start()

    # log randomly charing or not. The server functions implementation will only train on clients that are currently charging.
    def _log_loop(self):
        while True:
            value = random.choice(["True", "False"])
            attrs = {"charging": value}

            try:
                self.client.log_attributes(attrs)
                print(f"[ATTR] Logged: {attrs} (server functions only selectes clients which are charging).")
            except Exception as e:
                print(f"Failed logging attributes: {e}")

            time.sleep(30)

    def train(
        self,
        scaleout_model: ScaleoutModel,
        settings,
        data_path=None,
        batch_size=32,
        epochs=1,
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
        """

        # Load data
        x_train, y_train = load_data(data_path)
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device).long()

        # Load model
        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.train()
        lr = settings["learning_rate"]

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()

        n_samples = x_train.shape[0]
        n_batches = int(math.ceil(n_samples / batch_size))


        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for b in range(n_batches):
                # Regularly check if task is aborted
                self.client.check_task_abort()

                batch_x = x_train[b * batch_size : (b + 1) * batch_size]
                batch_y = y_train[b * batch_size : (b + 1) * batch_size]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

                if b % 100 == 0:
                    print(
                        f"Epoch {epoch}/{epochs - 1} | "
                        f"Batch {b}/{n_batches - 1} | "
                        f"Loss: {loss.item():.4f}"
                    )

            epoch_loss = running_loss / total
            epoch_acc = correct / total if total > 0 else 0.0

            # Log metrics to Scaleout
            self.client.log_metric(
                {
                    "training_loss": float(epoch_loss),
                    "training_accuracy": float(epoch_acc),
                }
            )
            print(
                f"Epoch {epoch} finished | "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

        # Metadata needed for aggregation server side
        metadata = {
            "num_examples": int(n_samples),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
        }

        # Save model update (mandatory)
        result_model = save_parameters(model)
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
        model = load_parameters(scaleout_model)
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

