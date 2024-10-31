from pathlib import Path

import numpy as np
from six.moves import cPickle as pickle


def load_binary(filepath):
    with open(filepath, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        x = datadict["data"]
        y = datadict["labels"]
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
        return x, y


def load_CIFAR10(
    data_dir: Path,
    *,
    num_train=49000,
    num_val=1000,
    num_test=1000,
):
    if not data_dir.is_dir():
        raise ValueError("First argument must be an existing directory")

    xs, ys = [], []
    for batch in range(1, 6):
        x, y = load_binary(data_dir / f"data_batch_{batch}")
        xs.append(x)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    X_test, y_test = load_binary(data_dir / "test_batch")

    # subsample the data
    mask = list(range(num_train, num_train + num_val))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_train))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
