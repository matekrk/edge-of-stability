import os
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision.datasets import MNIST, VisionDataset

from new_data_labels import MNIST as MNIST_LABELS
from new_data_utils import flatten, unflatten, make_labels, standardize, fashion_transform, AugmentTensorDataset

MNIST_SHAPE = (28, 28, 1)
MNIST_STATS_FILES = ("mnist_train_mean.csv", "mnist_train_std.csv")

def possible_mnist_names():
    lst = []
    base_train = "mnist"
    size_train = [None, 1, 2, 5, 10, 20, 50]
    base_test = "mnistt"
    size_test = [None, 1, 2, 5, 10]
    aug = [None, "_augm", "_n_augm"]
    std = [None, "_st", "_ch_st"]

    for base, sizes in [(base_train, size_train), (base_test, size_test)]:
        for size in sizes:
            n0 = base[:]
            if size is not None:
                n0 += f"-{size}k"
            for a in aug:
                n1 = n0[:]
                if a is not None:
                    n1 += a
                for s in std:
                    n3 = n1[:]
                    if s is not None:
                        n3 += s
                    lst.append(n3)
    return lst

def possible_corruptions():
    return  [
        "brightness",
        "canny_edges",
        "dotted_line",
        "fog",
        "glass_blur",
        "identity",
        "impulse_noise",
        "motion_blur",
        "rotate",
        "scale",
        "shear",
        "shot_noise",
        "spatter",
        "stripe",
        "translate",
        "zigzag"
        ]

def possible_mnist_c_names():
    lst = []
    corruptions = possible_corruptions()
    base_train = "mnistc"
    size_train = [None, 1, 2, 5, 10, 20, 50]
    base_test = "mnistct"
    size_test = [None, 1, 2, 5, 10]
    sizes = [None, 1, 2, 5, 10]
    aug = [None, "_augm", "_n_augm"]
    std = [None, "_st", "_ch_st"]

    for c in corruptions:
        for base, sizes in [(base_train, size_train), (base_test, size_test)]:
            n0 = base[:] + f"_{c}"
            for size in sizes:
                n1 = n0[:]
                if size is not None:
                    n1 += f"-{size}k"
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

def unpack_mnist(name: str):

    def get_size(s: str):
        for p in [1, 2, 5, 10, 20, 50]:
            if s.startswith("-" + str(p) + "k"):
                return p, 1+len(str(p))+1
            if s.startswith("_"):
                return None, 0
        return None, 0
            
    def get_extra(s: str):
        n_augm, augm, ch_st, st = False, False, False, False
        base = 0
        if s.startswith("n_augm_"):
            n_augm = True
            base += 6
        elif s.startswith("augm_"):
            augm = True
            base += 5
        
        if s[base:].startswith("ch_st"):
            ch_st = True
        if s[base:].startswith("_st"):
            st = True

        return n_augm, augm, ch_st, st
    
    def get_corruption(s: str):
        for corruption in possible_corruptions():
            if s.startswith(corruption):
                len_corruption = len(corruption)
                return corruption, len_corruption
    
    train = True
    corrupted, size, augm, n_augm, st, ch_st, corruption = [None for _ in range(7)]
    
    len_base = len("mnist")
    if name.startswith("mnistc"):
        if name.startswith("mnistct"):
            train = False
            len_base += 1
        corrupted = True
        len_base += 1
        corruption, len_corruption = get_corruption(name[len_base+1:])
        len_base += len_corruption
    elif name.startswith("mnistt"):
        train = False
        len_base += 1
    
    len_base += 1

    size, len_size = get_size(name[len_base:])
    n_augm, augm, ch_st, st = get_extra(name[len_base+len_size+1:])

    return corrupted, train, size, augm, n_augm, st, ch_st, corruption


def load_mnist(name: str, loss: str, datasets_folder=None, stats = None):
    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    corrupted, train, size, augm, n_augm, st, ch_st, corruption = unpack_mnist(name)

    transform = fashion_transform(augm, n_augm, stats)
    target_transform = None

    if corrupted:
        data = MNISTC(corruption, DATASETS_FOLDER, train, transform, target_transform)
    else:
        data = MNIST(DATASETS_FOLDER, train, transform, target_transform)

    if size is not None:
        assert 1000*size <= len(data)
        new_data, new_targets = data.data[:1000*size], data.targets[:1000*size]
    else:
        new_data, new_targets = data.data, data.targets

    new_data = np.array(new_data)

    new_targets = make_labels(torch.tensor(new_targets), loss)
    new_data = flatten(new_data / 255.0)
    mean, std = None, None
    if st:
        new_data, mean, std = standardize(new_data, stats, ch_st, os.path.join(DATASETS_FOLDER, MNIST_STATS_FILES[0]), os.path.join(DATASETS_FOLDER, MNIST_STATS_FILES[1]))

    new_data = unflatten(new_data, MNIST_SHAPE) # .permute(0, 3, 1, 2)
    return AugmentTensorDataset(transform, new_data, new_targets), mean, std

    # if size is not None:
    #     assert 1000*size <= len(data)
    #     data.data, data.targets = data.data[:1000*size], data.targets[:1000*size]

    # data.targets = make_labels(torch.tensor(data.targets), loss)

    # data.data = flatten(data.data / 255.0)
    # mean, std = None, None
    # if st:
    #     data.data, mean, std = standardize(data.data, stats, ch_st, os.path.join(DATASETS_FOLDER, MNIST_STATS_FILES[0]), os.path.join(DATASETS_FOLDER, MNIST_STATS_FILES[1]),)
    # data.data = unflatten(data.data, MNIST_SHAPE) # .permute(0, 3, 1, 2)
    # return data, mean, std


class MNISTC(VisionDataset):
    def __init__(self, type:str,  datasets_folder: str = None, train: bool = True, 
                 transform=None, target_transform=None):
        if datasets_folder is None:
            if "DATASETS" in os.environ:
                DATASETS_FOLDER = os.environ["DATASETS"]
            else:
                DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        else:
            DATASETS_FOLDER = datasets_folder

        DIR_CORRUPTION = "MNIST-C"
        str_type = "train" if train else "test"

        assert type in possible_corruptions()

        super(MNISTC, self).__init__(
            DATASETS_FOLDER, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), type, str_type + "_images.npy")
        target_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), type, str_type + "_labels.npy")
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)
