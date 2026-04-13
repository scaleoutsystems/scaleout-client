import io
import os
import sys
from tensorflow.keras.callbacks import Callback


import numpy as np
from data import load_data, prepare_data
from model import load_parameters, save_parameters

from scaleout import EdgeClient
from scaleoututil.utils.model import ScaleoutModel

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def startup(client: EdgeClient):
    prepare_data()
    MyClient(client)


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)
        

    def train(self, scaleout_model: ScaleoutModel, settings, data_path=None, batch_size=32, epochs=5):
        """Complete a model update.

        Load model paramters from in_model_path (managed by the Scaleout client),
        perform a model update, and write updated paramters
        to out_model_path (picked up by the Scaleout client).

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param settings: currently unused
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

        # Load model
        model = load_parameters(scaleout_model)

        # Train
        for epoch in range(epochs):
            print("epoch: ", epoch, "/", epochs)
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, callbacks=[AbortTrainingCallback(self.client)])
            self.client.log_metric({"training_loss": history.history["loss"][-1], "training_accuracy": history.history["accuracy"][-1]})

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(x_train),
            "batch_size": batch_size,
            "epochs": epochs,
        }
        # Regularly check if the task has been aborted
        self.client.check_task_abort()  # Throws an exception if the task has been aborted
        # Save model update (mandatory)
        result_model = save_parameters(model)
        return result_model, {"training_metadata": metadata}

    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Validate model.

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param data_path: The path to the data file.
        :type data_path: str
        """
        # Load data
        x_train, y_train = load_data(data_path)
        x_test, y_test = load_data(data_path, is_train=False)

        # Load model
        model = load_parameters(scaleout_model)

        # Evaluate
        model_score = model.evaluate(x_train, y_train)
        model_score_test = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        # JSON schema
        report = {
            "training_loss": model_score[0],
            "training_accuracy": model_score[1],
            "test_loss": model_score_test[0],
            "test_accuracy": model_score_test[1],
        }

        return report

    def predict(self, scaleout_model: ScaleoutModel, data_path=None):
        # Using test data for prediction but another dataset could be loaded
        x_test, _ = load_data(data_path, is_train=False)

        # Load model
        model = load_parameters(scaleout_model)

        # Predict
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        return {"predictions": y_pred}
    

class AbortTrainingCallback(Callback):
    def __init__(self, EdgeClient):
        super().__init__()
        self.EdgeClient = EdgeClient

    def on_batch_begin(self, batch, logs=None):
        self.EdgeClient.check_task_abort()
