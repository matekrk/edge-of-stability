import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F


CIFAR_SHAPE = (32, 32, 3)
CIFAR_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

stats = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617)) # (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.201)
# stats_imagenet = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
potential_mean_file = "data/cifar_train_mean.csv"
potential_std_file = "data/cifar_train_std.csv"

def possible_cifar_names():
    lst = []
    base = "cifar10"
    cat = [None, 1, 2, 5, 10, 20]
    testcat = [None, 1]
    aug = [None, "_augm", "_n_augm"]
    std = [None, "_st", "_ch_st"]

    for c in cat:
        n0 = base[:]
        if c is not None:
            n0 += f"-{c}k"
        for tc in testcat:
            n1 = n0[:]
            if tc is not None:
                n1 += f"-{tc}k"
            for a in aug:
                n2 = n1[:]
                if a is not None:
                    n2 += a
                for s in std:
                    n3 = n2[:]
                    if s is not None:
                        n3 += s
                    lst.append(n3)
    return lst

def possible_cifar_c_names():
    return []

class AugmentTensorDataset(TensorDataset):

    def __init__(self, transform, *tensors):      
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = [tensor[index] for tensor in self.tensors]
        if self.transform is not None:
            x[0] = self.transform(x[0])
        return tuple(x)

    def __len__(self):
        return self.tensors[0].size(0)

def center(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None, channel_wise: bool = True):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    if channel_wise:
        if os.path.exists(potential_mean_file):
            mean = np.genfromtxt(potential_mean_file, dtype=float)
        else:
            mean = X_combined.mean(0)
            np.savetxt(potential_mean_file, mean)
    else:
        mean = stats[0]
    return X_train - mean, X_test - mean, X_val - mean

def scale(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None, channel_wise: bool = True):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    if channel_wise:
        if os.path.exists(potential_std_file):
            std = np.genfromtxt(potential_std_file, dtype=float)
        else:
            std = X_combined.std(0)
            np.savetxt(potential_std_file, std)
    else:
        std = stats[1]
    return X_train / std, X_test / std, X_val / std

def standardize(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None, channel_wise: bool = True):
    center_X_train, center_X_test, center_X_val = center(X_train, X_test, X_val, channel_wise)
    standardized_X_train, standardized_X_test, standarized_X_val = scale(center_X_train, center_X_test, X_val, channel_wise)
    return standardized_X_train, standardized_X_test, standarized_X_val

def flatten(arr: np.ndarray):
    if len(arr) == 0:
        return arr
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    if len(arr) == 0:
        return arr
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

def cifar_transform(augment, normalize):
    normalization = transforms.Normalize(stats[0], stats[1])
    
    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [  transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # transforms.RandomAffine(0),
                            transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                            ] + transform_list

    if normalize:
        transform_list += [normalization]
    
    return transforms.Compose([transform_list])

def load_cifar(loss: str, datasets_folder=None, standardize_data=False, standarize_channel_wise=False, augment=False, normalize=False) -> Tuple[TensorDataset, TensorDataset]:
    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    if standardize_data:
        standardized_X_train, standardized_X_test = standardize(X_train, X_test, standarize_channel_wise)
    else:
        standardized_X_train, standardized_X_test = X_train, X_test
    
    if augment:
        transform = cifar_transform(augment, normalize)
        train = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_train, CIFAR_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_test, CIFAR_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
    else:
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

def predict_particular_cifar(network, standardize = True, return_dict = False, softmax = False, true_label=True):

    names = ["sky", "red", "green"]
    d_name_img = {}
    d_name_logits = {}

    for name in names:
        d_name_img[name] = torch.tensor(create_uniform_image(name, CIFAR_SHAPE, standardize), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
        with torch.no_grad():
            v = network(torch.unsqueeze(d_name_img[name], 0))[0, :]
            if softmax:
                v = torch.softmax(v, dim=0)
            d_name_logits[name] = v

    if return_dict:
        d = {}
        for name in names:
            d[name] = {}
            for i, c in enumerate(CIFAR_LABELS):
                d[name][c] = d_name_logits[name][i]
        return d

    return d_name_logits
