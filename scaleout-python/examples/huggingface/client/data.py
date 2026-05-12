import os
from math import floor

import torch
from datasets import load_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(data_path=None, is_train=True):
    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH", abs_path + "/data/clients/1/enron_spam.pt")
    data = torch.load(data_path, weights_only=False)
    if is_train:
        X = data["X_train"]
        y = data["y_train"]
    else:
        X = data["X_test"]
        y = data["y_test"]
    return X, y


def splitset(dataset, parts):
    n = len(dataset)
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def split(out_dir="data", n_splits=2):
    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.makedirs(f"{out_dir}/clients")

    dataset = load_dataset("SetFit/enron_spam")
    train_data = dataset["train"].to_pandas()
    test_data = dataset["test"].to_pandas()

    X_train = train_data["text"].to_numpy()
    y_train = train_data["label"].to_numpy()
    X_test = test_data["text"].to_numpy()
    y_test = test_data["label"].to_numpy()

    # Reduce data size for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:700]
    y_test = y_test[:700]

    data = {
        "X_train": splitset(X_train, n_splits),
        "y_train": splitset(y_train, n_splits),
        "X_test": splitset(X_test, n_splits),
        "y_test": splitset(y_test, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "X_train": data["X_train"][i],
                "y_train": data["y_train"][i],
                "X_test": data["X_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/enron_spam.pt",
        )


def prepare_data():
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        split(out_dir=abs_path + "/data")
