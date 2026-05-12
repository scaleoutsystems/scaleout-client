import collections

import torch
from transformers import AutoModelForSequenceClassification

from scaleoututil.helpers.helpers import get_helper
from scaleoututil.utils.model import ScaleoutModel

MODEL = "google/bert_uncased_L-2_H-128_A-2"
HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    """Compile the pytorch model.
  
    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    return model


def save_parameters(model):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    """
    # Convert PyTorch state_dict to numpy arrays
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ScaleoutModel.from_training_model(parameters_np, helper)


def load_parameters(scaleout_model: ScaleoutModel):
    """Load model parameters from file and populate model.

    :param scaleout_model: The ScaleoutModel containing the model parameters.
    :type scaleout_model: ScaleoutModel
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    weights = scaleout_model.get_training_model(helper)
    model = compile_model()
    
    # Convert numpy arrays back to PyTorch state_dict
    params_dict = zip(model.state_dict().keys(), weights)
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
    # Convert to numpy arrays for saving
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def build():
    init_seed("seed.npz")


if __name__ == "__main__":
    init_seed("../seed.npz")