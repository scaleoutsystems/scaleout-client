import os
from math import floor

import torch
import torchvision

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

# Always store data under client/data
OUT_DIR = os.path.join(abs_path, "data")


def get_data(out_dir=OUT_DIR):
    # Make dir if necessary
    os.makedirs(out_dir, exist_ok=True)

    # Only download if not already downloaded
    train_path = os.path.join(out_dir, "MNIST", "processed", "training.pt")
    test_path = os.path.join(out_dir, "MNIST", "processed", "test.pt")

    if not os.path.exists(train_path):
        torchvision.datasets.MNIST(
            root=out_dir,
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True,
        )
    if not os.path.exists(test_path):
        torchvision.datasets.MNIST(
            root=out_dir,
            transform=torchvision.transforms.ToTensor(),
            train=False,
            download=True,
        )


def load_data(data_path=None, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get(
            "SCALEOUT_DATA_PATH",
            os.path.join(OUT_DIR, "clients", "1", "mnist.pt"),
        )

    # weights_only=True requires PyTorch 2.0+; if you need legacy support, drop it.
    data = torch.load(data_path, weights_only=True)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255.0

    return X, y


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def split(out_dir=OUT_DIR):
    n_splits = int(os.environ.get("SCALEOUT_NUM_DATA_SPLITS", 2))

    # Make dir
    os.makedirs(os.path.join(out_dir, "clients"), exist_ok=True)

    # Load and convert to dict
    train_data = torchvision.datasets.MNIST(
        root=out_dir, transform=torchvision.transforms.ToTensor(), train=True
    )
    test_data = torchvision.datasets.MNIST(
        root=out_dir, transform=torchvision.transforms.ToTensor(), train=False
    )

    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(train_data.targets, n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(test_data.targets, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = os.path.join(out_dir, "clients", str(i + 1))
        os.makedirs(subdir, exist_ok=True)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            os.path.join(subdir, "mnist.pt"),
        )


def prepare_data(out_dir=OUT_DIR):
    """Prepare data once, matching the Keras example's `prepare_data()`."""
    client_file = os.path.join(out_dir, "clients", "1", "mnist.pt")
    if not os.path.exists(client_file):
        get_data(out_dir=out_dir)
        split(out_dir=out_dir)


if __name__ == "__main__":
    prepare_data()
