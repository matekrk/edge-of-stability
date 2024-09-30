import numpy as np
from torchvision.datasets import SVHN
from torchvision import transforms
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F
from data_cifar import AugmentTensorDataset, cifar_transform, flatten, unflatten, make_labels, standardize
from data_coloured_mnist import predict_particular_coloured_mnist

SVHN_SHAPE = (32, 32, 3)
SVHN_LABELS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

stats = ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
potential_mean_file = "data/svhn_train_mean.csv"
potential_std_file = "data/svhn_train_std.csv"

def load_svhn_mask(loss: str, datasets_folder=None, standardize_data=False, standarize_channel_wise=False, augment=False, normalize=False, mask=None) -> Tuple[TensorDataset, TensorDataset]:
    def target_transform(target):
        return int(target) - 1
    
    if datasets_folder is None:
        DATASETS_FOLDER = os.environ["DATASETS"]

    svhn_train = SVHN(root=DATASETS_FOLDER, download=True, split="train")
    SVHN_LOADED_SHAPE = (3, 32, 32)
    if mask is not None:
        svhn_train = torch.utils.data.Subset(svhn_train, mask)
    svhn_val = SVHN(root=DATASETS_FOLDER, download=True, split="extra") #, target_transform=target_transform)
    svhn_test = SVHN(root=DATASETS_FOLDER, download=True, split="test") #, target_transform=target_transform)
    X_train, X_test = flatten(svhn_train.data / 255), flatten(svhn_test.data / 255)
    y_train, y_test = make_labels(torch.from_numpy(svhn_train.labels), loss), \
        make_labels(torch.from_numpy(svhn_test.labels), loss)
    
    if standardize_data:
        standardized_X_train, standardized_X_test = standardize(X_train, X_test, standarize_channel_wise)
    else:
        standardized_X_train, standardized_X_test = X_train, X_test
    
    if augment:
        transform = cifar_transform(augment, normalize)
        train = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_train, SVHN_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_test, SVHN_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
    else:
        train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, SVHN_LOADED_SHAPE)).float(), y_train)
        test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, SVHN_LOADED_SHAPE)).float(), y_test)
    
    return train, test

def predict_particular_svhn(network, return_dict=True, softmax=False, standardize=False, true_label=True):
    return predict_particular_coloured_mnist(network, return_dict, softmax, standardize, true_label)
