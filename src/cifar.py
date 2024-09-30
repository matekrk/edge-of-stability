import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F

CIFAR_SHAPE = (32, 32, 3)
CIFAR_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

potential_mean_file = "data/cifar_train_mean.csv"
potential_std_file = "data/cifar_train_std.csv"

def center(X_train: np.ndarray, X_test: np.ndarray):
    if os.path.exists(potential_mean_file):
        mean = np.genfromtxt(potential_mean_file, dtype=float)
    else:
        mean = X_train.mean(0)
        np.savetxt("data/cifar_train_mean.csv", mean)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    if os.path.exists(potential_std_file):
        std = np.genfromtxt(potential_std_file, dtype=float)
    else:
        std = X_train.std(0)
        np.savetxt("data/cifar_train_std.csv", std)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str, datasets_folder=None, standardize_data=True) -> (TensorDataset, TensorDataset):
    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    if standardize_data:
        center_X_train, center_X_test = center(X_train, X_test)
        standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    else:
        standardized_X_train, standardized_X_test = X_train, X_test
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, CIFAR_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, CIFAR_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def create_uniform_image(background, shape, standardize = True):
    if background == "sky":
        pixel = [123, 191, 232]
    elif background == "red":
        pixel = [253, 0, 3]
    elif background == "green":
        pixel = [11, 241, 4]

    pixel = np.array(pixel)
    pixel = pixel / 255.0

    img = np.broadcast_to(pixel, shape).transpose(2, 0, 1)

    if standardize:
        mean = np.genfromtxt(potential_mean_file, dtype=float)
        mean = mean.reshape(*shape).transpose(2, 0, 1)
        std = np.genfromtxt(potential_std_file, dtype=float)
        std = std.reshape(*shape).transpose(2, 0, 1)
        img = np.divide((img - mean), std)

    return img

def predict_particular(network, standardize = True):
    sky = torch.tensor(create_uniform_image("sky", CIFAR_SHAPE, standardize), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
    red = torch.tensor(create_uniform_image("red", CIFAR_SHAPE, standardize), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
    green = torch.tensor(create_uniform_image("green", CIFAR_SHAPE, standardize), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)

    with torch.no_grad():
        pred = network(torch.unsqueeze(sky, 0))
        logits_sky = pred[0,:]
        pred = network(torch.unsqueeze(red, 0))
        logits_red = pred[0, :]
        pred = network(torch.unsqueeze(green, 0))
        logits_green = pred[0, :]

    return {'sky': logits_sky.clone().detach().cpu(), 'red': logits_red.clone().detach().cpu(), 'green': logits_green.clone().detach().cpu()}
