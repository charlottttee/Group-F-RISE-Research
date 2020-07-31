from typing import Mapping
from pathlib import Path
import h5py
import numpy as np


_DATADIR = Path(__file__).parent


def load_mnist() -> Mapping:

    with h5py.File(_DATADIR / "mnist.hdf5", "r") as f:
        out = {}
        out["train_images"] = f["train_images"][:]
        out["train_labels"] = f["train_labels"][:]
        out["test_images"] = f["test_images"][:]
        out["test_labels"] = f["test_labels"][:]

    return out


def raw_to_hdf5():

    fname = _DATADIR / "train-images-idx3-ubyte"
    train_images = open(fname, 'rb').read()
    train_images = np.frombuffer(train_images[16:], dtype="u1")
    train_images = train_images.reshape(60000, 28 * 28)

    fname = _DATADIR / "train-labels-idx1-ubyte"
    train_labels = open(fname, 'rb').read()
    train_labels = np.frombuffer(train_labels[8:], dtype="u1")

    fname = _DATADIR / "t10k-images-idx3-ubyte"
    test_images = open(fname, 'rb').read()
    test_images = np.frombuffer(test_images[16:], dtype="u1")
    test_images = test_images.reshape(10000, 28 * 28)

    fname = _DATADIR / "t10k-labels-idx1-ubyte"
    test_labels = open(fname, 'rb').read()
    test_labels = np.frombuffer(test_labels[8:], dtype="u1")



    with h5py.File(_DATADIR / "mnist.hdf5", "w") as f:
        f.create_dataset("train_labels", data=train_labels)
        f.create_dataset("train_images", data=train_images)
        f.create_dataset("test_labels", data=test_labels)
        f.create_dataset("test_images", data=test_images)


