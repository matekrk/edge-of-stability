import random
from typing import Tuple, Union, List
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset, DataLoader

from new_data_cifar100 import load_cifar100
from new_data_cifar10 import load_cifar10, possible_cifar10_names, possible_cifar10_c_names
from new_data_coloured_mnist import load_coloured_mnist, possible_coloured_mnist_names
from new_data_mnist import load_mnist, possible_mnist_names, possible_mnist_c_names
from new_data_fashion import load_fashion, possible_fashion_names

from new_data_utils import AugmentTensorDataset

DATASETS = possible_fashion_names() + possible_cifar10_names() + possible_cifar10_c_names() + possible_coloured_mnist_names() + possible_mnist_names() + possible_mnist_c_names() + possible_fashion_names()

def prepare_data(train_dataset_name: str, test_dataset_names: List[str], loss: str, train_batch_size: int, test_batch_size: int):
    data_train, mean, std = load_dataset(train_dataset_name, loss, stats=None)
    data_train_loader = get_dataloader(data_train, train_batch_size)
    data_tests, data_test_loaders = [], []
    for test_dataset_name in test_dataset_names:
        data_tests.append(load_dataset(test_dataset_name, loss, (mean, std))[0])
        data_test_loaders.append(get_dataloader(data_tests[-1], test_batch_size))
    return data_train, data_train_loader, data_tests, data_test_loaders


def load_dataset(dataset_name: str, loss: str, stats: Tuple) -> Union[TensorDataset, AugmentTensorDataset]: # Tuple[Union[TensorDataset, AugmentTensorDataset], Union[TensorDataset, AugmentTensorDataset]]
    
    if dataset_name.startswith("cifar100"):
        data, mean, std = load_cifar100(dataset_name, loss, stats=stats)
    elif dataset_name.startswith("cifar10"):
        data, mean, std = load_cifar10(dataset_name, loss, stats=stats)
    elif dataset_name.startswith("coloured"):
        data, mean, std = load_coloured_mnist(dataset_name, loss, stats=stats)
    elif dataset_name.startswith("mnist"):
        data, mean, std = load_mnist(dataset_name, loss, stats=stats)
    elif dataset_name.startswith("fashion"):
        data, mean, std = load_fashion(dataset_name, loss, stats=stats)
    
    
    return data, mean, std

def extract_subset(dataset, num_subset: int, random_subset: bool = True, complementary: bool = False):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
        if complementary:
            indices = list(set(range(len(dataset))) - set(indices))
    else:
        if complementary:
            indices = [i for i in range(num_subset, len(dataset))]
        else:
            indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)

def split_dataset(train_dataset, split_val: float, random_subset: bool = True):
    num_samples = len(train_dataset)
    num_train_samples = int(split_val * num_samples)

    train_dataset = extract_subset(train_dataset, num_train_samples, random_subset)
    val_dataset = extract_subset(train_dataset, num_samples - num_train_samples, random_subset, complementary=True)

    return train_dataset, val_dataset

def get_data(train_dataset_name, test_dataset_names, loss, split_val):
    train = load_dataset(train_dataset_name, loss)
    if split_val:
        train, val = split_dataset(train, split_val)
    else:
        val = None
    test = []
    for test_dataset_name in test_dataset_names:
        test.append(load_dataset(test_dataset_name, loss))
    return train, val, test

def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
