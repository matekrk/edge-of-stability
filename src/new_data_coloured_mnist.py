import os
from typing import Tuple
import h5py
import random
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data.dataset import TensorDataset
import torchvision.transforms as transforms

from new_data_utils import split_np_random
from new_data_utils import flatten, unflatten, make_labels, standardize, natural_image_transform, AugmentTensorDataset

COLOURED_MNIST_SHAPE = (28, 28, 3)
COLOURED_MNIST_STATS = ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617)) # FIXME
COLOURED_MNIST_BACKGROUD = (
    (255, 0, 0),    # 0 # RED
    (0, 255, 0),    # 1 # GREEN
    (0, 0, 255),    # 2 # BLUE
    (255, 255, 0),  # 3 # YELLOW
    (0, 255, 255),  # 4 # CYAN
    (196, 0, 255),  # 5 # PURPLE
    (128, 128, 0),  # 6 # OLIVE
    (128, 128, 128),# 7 # GREY
    (255, 153, 51), # 8 # ORANGE
    (255, 204, 229) # 9 # PINK
)
COLOURED_MNIST_BACKGROUD_NAMES = ("R", "G", "B", "Y", "C", "P", "O", "S", "A", "N")
COLOURED_MNIST_DIR = "coloured"
COLOURED_MNIST_STATS_FILES = ("coloured_mnist_train_mean.csv", "coloured_mnist_train_std.csv")

def get_file(path, type_str: str, random_background: bool, random_mistakes: float):
    if random_background:
        return (
            os.path.join(path, COLOURED_MNIST_DIR, f"random_background", f"{type_str}.npy"), 
            os.path.join(path, COLOURED_MNIST_DIR, f"random_background", f"{type_str}_labels.npy")
        )
    elif random_mistakes:
        return (
            os.path.join(path, COLOURED_MNIST_DIR, f"errors{random_mistakes}", f"{type_str}.npy"),
            os.path.join(path, COLOURED_MNIST_DIR, f"errors{random_mistakes}", f"{type_str}_labels.npy")
        )
    else:
        return (
            os.path.join(path, COLOURED_MNIST_DIR, "RGBYCPOSAN", f"{type_str}.npy"),
            os.path.join(path, COLOURED_MNIST_DIR, "RGBYCPOSAN", f"{type_str}_labels.npy")
        )

def possible_coloured_mnist_names():
    base = "coloured"
    size_train = [None, 1, 2, 5, 10, 20, 50]
    size_test = [None, 1, 2, 5, 10]
    rms = [None] + [0.1 * s for s in range(11)]
    aug = [None, "_augm", "_n_augm"]
    std = [None, "_st", "_ch_st"]
    lst = []
    for ap in [None, "rb", "rm"]:
        n0 = base[:]
        if ap is not None:
            n0 += ap
        for rm in rms:
            n1 = n0[:]
            if ap is None and rm is not None:
                continue
            elif ap == "rb":
                n1 += ap
            elif ap == "rm":
                if rm is None:
                    continue
                n1 += "{:.1f}".format(rm)
                
            for t in ["", "t"]:
                n2 = n1[:]
                if t == "t":
                    n2 += t
                    sizes = size_test
                else:
                    sizes = size_train
                for size in sizes:
                    n3 = n2[:]
                    if size is not None:
                        n3 += f"-{size}k"
                    for a in aug:
                        n4 = n3[:]
                        if a is not None:
                            n4 += a
                        for s in std:
                            n5 = n4[:]
                            if s is not None:
                                n5 += s
                            lst.append(n5)
    return lst


## CREATE

def colour_image(digit_image, label, random_background = False, particular_background = None, random_error=0.0, tolerance = 2):

    black_pixel = (0, 0, 0)
    white_pixel = (255, 255, 255)
    black_image = np.broadcast_to(black_pixel, COLOURED_MNIST_SHAPE)
    white_image = np.broadcast_to(white_pixel, COLOURED_MNIST_SHAPE)

    digit_pil = transforms.ToPILImage()(digit_image)
    color_image = digit_pil.convert('RGB')
    np_image = np.asarray(color_image).copy()

    if random_background:
        rand_int = random.randint(0, len(COLOURED_MNIST_BACKGROUD)-1)
        background_pixel = COLOURED_MNIST_BACKGROUD[rand_int]
    else:
        fool = random.random() < random_error
        if fool:
            select_background = random.choice([i for i in range(10) if i != label])
        else:
            if particular_background is None:
                select_background = label
            else:
                select_background = particular_background
        background_pixel = COLOURED_MNIST_BACKGROUD[select_background]

    white_mask = np.all(abs(color_image - white_image) <= tolerance, axis=-1)
    black_mask = np.all(abs(color_image - black_image) <= tolerance, axis=-1)
    
    np_image[black_mask] = background_pixel
    np_image[~black_mask] = np.expand_dims((1.0 - np_image[~black_mask][:,0] / 255), axis=1) @ np.expand_dims(np.array(background_pixel), axis=0)

    return np_image

def create_coloured_mnist(path, train, random_background=False, random_mistakes = 0.0):
    assert not random_background and random_mistakes == 0.0

    type_str = "train" if train else "test"

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_standard = MNIST(root=path, train=train, download=True, transform=transform)

    X_data, y_data = mnist_train_standard.data.numpy(), mnist_train_standard.targets

    images_lst = []
    labels_lst = []

    for digit_image, label in zip(X_data, y_data):
        
        coloured_image = colour_image(digit_image, label, random_background=random_background, random_error=random_mistakes)
        images_lst.append(coloured_image)
        labels_lst.append(label)

    images_file, labels_file = get_file(path, type_str, random_background, random_mistakes)
    np.save(images_file, np.array(images_lst))
    np.save(labels_file, np.array(labels_lst))

    return images_file, labels_file

## USE

def unpack_coloured_mnist(name: str):

    def get_base(s: str):
        len_base = len("coloured")
        train = True
        random_background = False
        random_mistake = 0.0
        if s.startswith("colouredrb"):
            if s.startswith("colouredrbt"):
                len_base += 1
                train = False
            len_base += 2
            random_background = True
        elif s.startswith("colouredrm"):
            if s.startswith("colouredrbt"):
                len_base += 1
                train = False
            # easy sol
            len_base += 2
            random_mistake = float(s[len_base:len_base+3])
            len_base += 3
        elif s.startswith("colouredt"):
            len_base += 1
            train = False

        return train, random_background, random_mistakes, len_base
            

    def get_size(s: str):
        for p in [1, 2, 5, 10, 20, 50]:
            if s.startswith("-" + str(p) + "k"):
                return p, 1+len(str(p))+1
            if s.startswith("_"):
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
    
    assert name in possible_coloured_mnist_names()

    train = True
    size, augm, n_augm, st, ch_st = None, None, None, None, None

    train, random_background, random_mistakes, len_base = get_base(name)

    size, len_size = get_size(name[len_base:])
    n_augm, augm, ch_st, st = get_extra(name[len_base+len_size+1:])
    return train, random_background, random_mistakes, size, augm, n_augm, st, ch_st


def load_coloured_mnist(name: str, loss: str, datasets_folder=None, stats = None):

    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    train, random_background, random_mistakes, size, augm, n_augm, st, ch_st = unpack_coloured_mnist(name)
    type_str = "train" if train else "test"
    images_file, labels_file = get_file(DATASETS_FOLDER, type_str, random_background, random_mistakes)

    if os.path.exists(images_file):
        images = np.load(images_file)
        labels = np.load(labels_file)
    else:
        images, labels = create_coloured_mnist(DATASETS_FOLDER, train, random_background, random_mistakes)

    images = torch.from_numpy(images.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    if size is not None:
        assert 1000*size <= len(images)
        images, labels = images[:1000*size], labels[:1000*size]

    labels = make_labels(torch.tensor(labels), loss)

    images = flatten(images / 255.0)
    if st:
        images, mean, std = standardize(images, stats, ch_st, os.path.join(DATASETS_FOLDER, COLOURED_MNIST_DIR, COLOURED_MNIST_STATS_FILES[0]), os.path.join(DATASETS_FOLDER, COLOURED_MNIST_DIR, COLOURED_MNIST_STATS_FILES[1]),)
    images = unflatten(images, COLOURED_MNIST_SHAPE).transpose((0, 3, 1, 2))

    transform = natural_image_transform(augm, n_augm, COLOURED_MNIST_STATS)
    data = AugmentTensorDataset(transform, images, labels)
    return data, mean, std
