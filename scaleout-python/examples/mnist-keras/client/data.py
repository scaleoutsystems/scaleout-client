import os
from math import floor

import numpy as np
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

NUM_CLASSES = 10


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Download data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.savez(f"{out_dir}/mnist.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load train or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH", abs_path + "/data/clients/1/mnist.npz")

    data = np.load(data_path)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X.astype("float32")
    X = np.expand_dims(X, -1)
    X = X / 255
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    return X, y


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return np.array(result)


def split(dataset="data/mnist.npz", outdir="data", n_splits=2):
    # Load and convert to dict
    package = np.load(dataset)
    data = {}
    for key, val in package.items():
        data[key] = splitset(val, n_splits)

    # Make dir if necessary
    if not os.path.exists(f"{outdir}/clients"):
        os.mkdir(f"{outdir}/clients")

    # Make splits
    for i in range(n_splits):
        subdir = f"{outdir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        np.savez(f"{subdir}/mnist.npz", x_train=data["x_train"][i], y_train=data["y_train"][i], x_test=data["x_test"][i], y_test=data["y_test"][i])


def prepare_data():
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data(abs_path + "/data")
        split(abs_path + "/data/mnist.npz", abs_path + "/data", n_splits=2)