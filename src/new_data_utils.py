import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import Tuple, Union
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import Dataset, DataLoader
import os
import torch
from torch import Tensor
import torch.nn.functional as F

def center_traintestval(X_train: Union[np.ndarray, torch.Tensor], X_test: Union[np.ndarray, torch.Tensor], X_val: Union[np.ndarray, torch.Tensor] = None, channel_wise: bool = True, potential_mean_file = None, stats = None):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    if channel_wise:
        if potential_mean_file is not None and os.path.exists(potential_mean_file):
            mean = np.genfromtxt(potential_mean_file, dtype=float)
        else:
            mean = X_combined.mean(0)
            np.savetxt(potential_mean_file, mean)
    else:
        assert stats is not None
        mean = stats[0]
    return X_train - mean, X_test - mean, X_val - mean

def scale_traintestval(X_train: Union[np.ndarray, torch.Tensor], X_test: Union[np.ndarray, torch.Tensor], X_val: Union[np.ndarray, torch.Tensor] = None, channel_wise: bool = True, potential_std_file = None, stats = None):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    if channel_wise:
        if potential_std_file is not None and os.path.exists(potential_std_file):
            std = np.genfromtxt(potential_std_file, dtype=float)
        else:
            std = X_combined.std(0)
            np.savetxt(potential_std_file, std)
    else:
        std = stats[1]
    return X_train / std, X_test / std, X_val / std

def standardize_traintestval(X_train: Union[np.ndarray, torch.Tensor], X_test: Union[np.ndarray, torch.Tensor], X_val: Union[np.ndarray, torch.Tensor] = None, channel_wise: bool = True):
    center_X_train, center_X_test, center_X_val = center_traintestval(X_train, X_test, X_val, channel_wise)
    standardized_X_train, standardized_X_test, standarized_X_val = scale_traintestval(center_X_train, center_X_test, X_val, channel_wise)
    return standardized_X_train, standardized_X_test, standarized_X_val

def center(X_data: Union[np.ndarray, torch.Tensor], stats = None, channel_wise: bool = True, potential_mean_file: str = None):
    if channel_wise:
        if stats is not None:
            mean = stats[0]
        else:
            if potential_mean_file is not None and os.path.exists(potential_mean_file):
                mean = np.genfromtxt(potential_mean_file, dtype=float)
            else:
                mean = X_data.mean(0)
                np.savetxt(potential_mean_file, mean)
    else:
        assert stats is not None
        mean = stats[0]
    if isinstance(X_data, torch.Tensor):
        mean = torch.from_numpy(mean)
    return X_data - mean, mean

def scale(X_data: Union[np.ndarray, torch.Tensor], stats = None, channel_wise: bool = True, potential_std_file: str = None):
    if channel_wise:
        if stats is not None:
            std = stats[1]
        else:
            if potential_std_file is not None and os.path.exists(potential_std_file):
                std = np.genfromtxt(potential_std_file, dtype=float)
            else:
                std = X_data.std(0)
                np.savetxt(potential_std_file, std)
    else:
        std = stats[1]
    if isinstance(X_data, torch.Tensor):
        std = torch.from_numpy(std)
    return X_data / std, std

def standardize(X_data: Union[np.ndarray, torch.Tensor], stats = None, channel_wise: bool = True, potential_mean_file: str = None, potential_std_file: str = None):
    X_data, mean = center(X_data, stats, channel_wise, potential_mean_file)
    X_data, std = scale(X_data, stats, channel_wise, potential_std_file)
    return X_data, mean, std

def flatten(arr: Union[np.ndarray, torch.Tensor]):
    if len(arr) == 0:
        return arr
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: Union[np.ndarray, torch.Tensor], shape: Tuple):
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

def split_np_random(data, labels=None, ratio=0.8, seed=8):
    np.random.seed(seed)
    if labels is not None: 
        p = np.random.permutation(len(data))
        data[p], labels[p] = data[p], labels[p]
    else:
        np.random.shuffle(data)

    index = int(ratio * data.shape[0])
    training, testing = data[:index], data[index:]
    training_labels, testing_labels = labels[:index], labels[index:]
    if labels is not None:
        return (training, training_labels), (testing, testing_labels)
    else:
        return training, testing
    
def split_torch_random(data, lables=None, ratio=0.8, seed=8):
    torch.manual_seed(seed)

    if labels is not None:
        indices = torch.randperm(len(data))

        data = data[indices]
        labels = labels[indices]
    else:
        torch.shuffle(data)

    split_index = int(ratio * len(data))

    train_data, test_data = data[:split_index], data[split_index:]
    if labels is not None:
        train_labels, test_labels = labels[:split_index], labels[split_index:]
        return (train_data, train_labels), (test_data, test_labels)
    else:
        return train_data, test_data
    
def natural_image_transform(augment: bool, normalize: bool, stats):
    # only cifar?
    
    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [  transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # transforms.RandomAffine(0),
                            transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                            ] + transform_list

    if normalize:
        transform_list += [transforms.Normalize(stats[0], stats[1])]
    
    return transforms.Compose(transform_list)

def fashion_transform(augment: bool, normalize: bool, stats):
    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [  transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # transforms.RandomAffine(0),
                            transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
                            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                            ] + transform_list
        
    if normalize:
        transform_list += [transforms.Normalize(stats[0], stats[1])]
    
    return transforms.Compose(transform_list)

class AugmentTensorDataset(TensorDataset):

    def __init__(self, transform, *tensors):      
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = [tensor[index] for tensor in self.tensors]
        if self.transform is not None:
            x[0] = self.transform(x[0]).to(torch.float32)
        return tuple(x)

    def __len__(self):
        return self.tensors[0].shape[0]

def iterate_dataset(dataset: Dataset, batch_size: int, counter = None):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    batch_id = 0
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()
        batch_id += 1
        if counter is not None and batch_id >= counter:
            break

def get_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
