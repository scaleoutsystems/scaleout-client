from scaleoututil.serverfunctions.allowed_import import Dict, List, ServerFunctionsBase, np

# See allowed_imports for what packages you can use in this class.

# Example of fedavg using memory secure running aggregation with server functions.


# Class must be named ServerFunctions
class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.global_model = None
        self.total_examples = 0
        self.round = 0
        self.lr = 0.1

    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        # Decrease learning rate every 10 rounds
        if self.round % 10 == 0:
            self.lr = self.lr * 0.1
        # see client/train.py for how to load the client settings.
        self.round += 1
        return {"learning_rate": self.lr}

    def incremental_aggregate(self, client_id: str, model: List[np.ndarray], client_metadata: Dict, previous_global: List[np.ndarray]):
        # set previous global to fail safe if no updates
        self.previous_global = previous_global
        # Initialize the global model during the first aggregation.
        # Use the client metadata to get the number of examples the client has.
        num_examples = client_metadata.get("num_examples", 1)
        self.total_examples += num_examples

        if self.global_model is None:
            self.global_model = model
        else:
            # Incrementally update the global model using the weighted average
            for i in range(len(self.global_model)):
                self.global_model[i] = (self.global_model[i] * (self.total_examples - num_examples) + model[i] * num_examples) / self.total_examples

        print(f"Model aggregated with {num_examples} examples.")

    def get_incremental_aggregate_model(self) -> List[np.ndarray]:
        # Return the current running aggregate global model and reset it.
        ret = self.global_model
        self.global_model = None
        if ret is None:
            # if no model updates was received, return the previous global model.
            return self.previous_global
        return ret
