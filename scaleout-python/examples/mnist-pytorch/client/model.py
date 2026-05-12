import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaleoututil.helpers.helpers import get_helper
from scaleoututil.utils.model import ScaleoutModel

NUM_CLASSES = 10
HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def compile_model():
    """Compile the PyTorch model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    return Net()


def save_parameters(model: nn.Module) -> ScaleoutModel:
    """Save model parameters to a ScaleoutModel.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :return: A ScaleoutModel wrapping the model parameters.
    :rtype: ScaleoutModel
    """
    with torch.no_grad():
        parameters_np = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    return ScaleoutModel.from_training_model(parameters_np, helper)


def load_parameters(scaleout_model: ScaleoutModel) -> nn.Module:
    """Load model parameters from ScaleoutModel and populate model.

    :param scaleout_model: The ScaleoutModel containing the model parameters.
    :type scaleout_model: ScaleoutModel
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = scaleout_model.get_training_model(helper)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    model = compile_model()
    with torch.no_grad():
        parameters_np = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def build():
    # Hook used by `scaleout build`
    init_seed("seed.npz")


if __name__ == "__main__":
    init_seed("../seed.npz")
