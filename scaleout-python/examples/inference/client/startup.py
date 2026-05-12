import time
from scaleout import EdgeClient, ScaleoutModel
from scaleoututil.helpers.helpers import get_helper

import numpy as np

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def startup(client: EdgeClient):
    MyClient(client)

class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client

        self.client.stage_model_callback = self.stage_model
        self.client.inference_callback = self.inference

        self._staged_model = []

        self._acc = 50
        self._model = None

    def stage_model(self, model: ScaleoutModel):
        print("Model staged:", model.model_id)
        if model not in self._staged_model:
            self._staged_model.append(model)

        staged_models = self.client.local_repository.models
        print("Staged models:", [m.model_id for m in staged_models])
        self.client.log_attributes({"CachedModels": str([m.model_id for m in staged_models])})
        self._set_current_model(model)

    def _set_current_model(self, model: ScaleoutModel):
        if model != self._model:
             print("Active model:", model.model_id)
             staged_models = self.client.local_repository.models
             self._acc = staged_models.index(model)*10 + 50
             self.client.log_attributes({"CurrentModel": model.model_id})
        self._model = model
        

    def inference(self, model: ScaleoutModel, input_data):
        print("Input data:", input_data)
        while True:
            self.client.log_telemetry({"acc": self._acc + np.random.randn()*2})
            time.sleep(5)