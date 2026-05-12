import tensorflow as tf

from scaleoututil.helpers.helpers import get_helper
from scaleoututil.utils.model import ScaleoutModel

NUM_CLASSES = 10
HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model(img_rows=28, img_cols=28):
    """Compile the TF model.

    param: img_rows: The number of rows in the image
    type: img_rows: int
    param: img_cols: The number of rows in the image
    type: img_cols: int
    return: The compiled model
    type: keras.model.Sequential
    """
    # Set input shape
    input_shape = (img_rows, img_cols, 1)

    # Define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    return model


def save_parameters(model):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: keras.model.Sequential
    """
    weights = model.get_weights()
    return ScaleoutModel.from_training_model(weights, helper)


def load_parameters(scaleout_model: ScaleoutModel):
    """Load model parameters from file and populate model.

    :param scaleout_model: The ScaleoutModel containing the model parameters.
    :type scaleout_model: ScaleoutModel
    :return: The loaded model.
    :rtype: keras.model.Sequential
    """
    weights = scaleout_model.get_training_model(helper)
    model = compile_model()
    model.set_weights(weights)
    return model


def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    weights = compile_model().get_weights()
    helper.save(weights, out_path)

def build():
    init_seed("seed.npz")

if __name__ == "__main__":
    init_seed("../seed.npz")
