import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
import torch.nn.functional as F

from cifar import flatten, make_labels, center, standardize, unflatten

FASHION_SHAPE = (28, 28, 1)
FASHION_LABELS = ('top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot')

def load_fashion_mask(loss: str, datasets_folder=None, mask=None) -> Tuple[TensorDataset, TensorDataset]:
    if datasets_folder is None:
        datasets_folder = os.environ["DATASETS"]

    mnist_train = FashionMNIST(root=datasets_folder, download=True, train=True)
    if mask is not None:
        mnist_train = torch.utils.data.Subset(mnist_train, mask)
    mnist_test = FashionMNIST(root=datasets_folder, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data.numpy() / 255), flatten(mnist_test.data.numpy() / 255)
    y_train, y_test = make_labels(mnist_train.targets.clone().detach(), loss), \
        make_labels(mnist_test.targets.clone().detach(), loss)
    standardized_X_train, standardized_X_test = X_train, X_test
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test


def load_fashion(loss: str, datasets_folder=None, standardize_data=False, standarize_channel_wise=False) -> Tuple[TensorDataset, TensorDataset]:
    if datasets_folder is None:
        datasets_folder = os.environ["DATASETS"]

    mnist_train = FashionMNIST(root=datasets_folder, download=True, train=True)
    mnist_test = FashionMNIST(root=datasets_folder, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data.numpy() / 255), flatten(mnist_test.data.numpy() / 255)
    y_train, y_test = make_labels(mnist_train.targets.clone().detach(), loss), \
        make_labels(mnist_test.targets.clone().detach(), loss)
    
    if standardize_data:
        standardized_X_train, standardized_X_test = standardize(X_train, X_test, standarize_channel_wise)
    else:
        standardized_X_train, standardized_X_test = X_train, X_test

    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)

    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test
