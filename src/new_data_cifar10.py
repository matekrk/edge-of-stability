import os
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import TensorDataset
from torchvision.datasets import CIFAR10, VisionDataset

from new_data_labels import CIFAR10 as CIFAR10_LABELS
from new_data_utils import flatten, unflatten, make_labels, standardize, natural_image_transform, AugmentTensorDataset


CIFAR_SHAPE = (32, 32, 3)
CIFAR_STATS = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617)) # (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.201)
CIFAR_STATS_FILES = ("cifar_train_mean.csv", "cifar_train_std.csv")
# stats_imagenet = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def possible_cifar10_names():
    lst = []
    base_train = "cifar10"
    size_train = [None, 1, 2, 5, 10, 20, 50]
    base_test = "cifar10t"
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
        "gaussian_noise",
        "shot_noise",
        "speckle_noise",
        "impulse_noise",
        "defocus_blur",
        "gaussian_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "spatter",
        "saturate",
        "frost",
        ]

def possible_cifar10_c_names():
    lst = []
    base = "cifar10c"
    corruptions = possible_corruptions()
    serverities = [1,2,3,4,5]
    sizes = [None, 1, 2, 5, 10]
    aug = [None, "_augm", "_n_augm"]
    std = [None, "_st", "_ch_st"]

    for c in corruptions:
        n0 = base[:] + f"_{c}"
        for s in serverities:
            n1 = n0[:]
            n1 += f"{s}"
            for size in sizes:
                n2 = n1[:]
                if size is not None:
                    n2 += f"-{size}k"
                for a in aug:
                    n3 = n2[:]
                    if a is not None:
                        n3 += a
                    for s in std:
                        n4 = n3[:]
                        if s is not None:
                            n4 += s
                        lst.append(n4)
    return lst

def unpack_cifar10(name: str):

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
                serverity = int(s[len_corruption])
                return corruption, serverity, len_corruption+1

    assert name in possible_cifar10_names() or name in possible_cifar10_c_names()

    train = True
    corrupted, size, augm, n_augm, st, ch_st, corruption, serverity = [None for _ in range(8)]

    len_base = len("cifar10")

    if name.startswith("cifar10c"):
        corrupted = True
        len_base += 1
        corruption, serverity, len_corruption = get_corruption(name[len_base+1:])
        len_base += len_corruption
    else:
        if name.startswith("cifar10t"):
            train = False
            len_base += 1
        
    size, len_size = get_size(name[len_base:])
    n_augm, augm, ch_st, st = get_extra(name[len_base+len_size+1:])

    return corrupted, train, size, augm, n_augm, st, ch_st, corruption, serverity

def load_cifar10(name: str, loss: str, datasets_folder=None, stats = None):

    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    corrupted, train, size, augm, n_augm, st, ch_st, corruption, serverity = unpack_cifar10(name)

    transform = natural_image_transform(augm, n_augm, stats)
    target_transform = None

    if corrupted:
        data = CIFAR10C(corruption, serverity, DATASETS_FOLDER, transform, target_transform)
    else:
        data = CIFAR10(DATASETS_FOLDER, train, transform, target_transform)

    if size is not None:
        assert 1000*size <= len(data)
        new_data, new_targets = data.data[:1000*size], data.targets[:1000*size]
    else:
        new_data, new_targets = data.data, data.targets

    new_targets = make_labels(torch.tensor(new_targets), loss)
    new_data = flatten(new_data / 255.0)
    mean, std = None, None
    if st:
        new_data, mean, std = standardize(new_data, stats, ch_st, os.path.join(DATASETS_FOLDER, CIFAR_STATS_FILES[0]), os.path.join(DATASETS_FOLDER, CIFAR_STATS_FILES[1]))

    new_data = unflatten(new_data, CIFAR_SHAPE) # .transpose((0, 3, 1, 2))
    return AugmentTensorDataset(transform, new_data, new_targets), mean, std
    # if size is not None:
    #     assert 1000*size <= len(data)
    #     data.data, data.targets = data.data[:1000*size], data.targets[:1000*size]

    # data.targets = make_labels(torch.tensor(data.targets), loss)

    # data.data = flatten(data.data / 255.0)

    # mean, std = None, None
    # if st:
    #     data.data, mean, std = standardize(data.data, stats, ch_st, os.path.join(DATASETS_FOLDER, CIFAR_STATS_FILES[0]), os.path.join(DATASETS_FOLDER, CIFAR_STATS_FILES[1]),)
    # data.data = unflatten(data.data, CIFAR_SHAPE).transpose((0, 3, 1, 2))
    # return data, mean, std

class CIFAR10C(VisionDataset):
    def __init__(self, type: str, serverity: int, datasets_folder: str = None,
                 transform=None, target_transform=None):
        #MEAN = [0.49139968, 0.48215841, 0.44653091]
        #STD  = [0.24703223, 0.24348513, 0.26158784]
        if datasets_folder is None:
            if "DATASETS" in os.environ:
                DATASETS_FOLDER = os.environ["DATASETS"]
            else:
                DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        else:
            DATASETS_FOLDER = datasets_folder

        DIR_CORRUPTION = "CIFAR-10-C"

        assert type in possible_corruptions()
        assert serverity in [1, 2, 3, 4, 5]
        super(CIFAR10C, self).__init__(
            DATASETS_FOLDER, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), type + ".npy")
        target_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), "labels.npy")
        
        self.data = np.load(data_path)[(serverity-1)*10000: serverity*10000]
        self.targets = np.load(target_path)[(serverity-1)*10000: serverity*10000]
        
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
        mean = np.genfromtxt(CIFAR_STATS_FILES[0], dtype=float)
        mean = mean.reshape(*shape).transpose(2, 0, 1)
        std = np.genfromtxt(CIFAR_STATS_FILES[1], dtype=float)
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
            for i, c in enumerate(CIFAR10_LABELS):
                d[name][c] = d_name_logits[name][i]
        return d

    return d_name_logits
