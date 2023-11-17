import numpy as np
from torchvision.datasets import MNIST
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F

from cifar import flatten, make_labels, center, standardize, unflatten

def load_mnist(loss: str, datasets_folder=None) -> (TensorDataset, TensorDataset):
    if datasets_folder is None:
        DATASETS_FOLDER = os.environ["DATASETS"]

    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data.numpy() / 255), flatten(mnist_test.data.numpy() / 255)
    y_train, y_test = make_labels(torch.tensor(mnist_train.targets), loss), \
        make_labels(torch.tensor(mnist_test.targets), loss)
    #center_X_train, center_X_test = center(X_train, X_test)
    #standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    standardized_X_train, standardized_X_test = X_train, X_test
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test
