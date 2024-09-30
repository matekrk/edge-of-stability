import h5py
import random
import numpy as np
from torchvision.datasets import MNIST
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from data_cifar import AugmentTensorDataset, cifar_transform, flatten, unflatten, make_labels

def possible_coloured_names():
    lst = []
    base = "coloured_mnist_split"
    for split in ["1.0", "0.8"]:
        basesplit = "coloured_mnist_split" + split
        for tr in ["", "_TRrandom"]:
            mistakes = [""]
            if tr == "":
                mistakes = mistakes + [f"_err{procent}" for procent in (0.01, 0.05, 0.1, 0.2, 0.5)]
            for err in mistakes:
                basesplit_err = basesplit + err
                basesplit_tr = basesplit_err + tr
                for val in ["", "_VALrandom"]:
                    basesplit_tr_val = basesplit_tr + val
                    for te in ["", "_TErandom"]:
                        basesplit_tr_val_te = basesplit_tr_val + te
                        lst.append(basesplit_tr_val_te)
    return lst

MNIST_COLOURED_SHAPE = (28, 28, 3)
MNIST_LABELS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

MNIST_COLOURED_BACKGROUD = (
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

MNIST_COLOURED_BACKGROUD_NAMES = ("R", "G", "B", "Y", "C", "P", "O", "S", "A", "N")

def colour_image(digit_image, label, random_background = False, particular_background = None, random_error=0.0, tolerance = 2):

    black_pixel = (0, 0, 0)
    white_pixel = (255, 255, 255)
    black_image = np.broadcast_to(black_pixel, MNIST_COLOURED_SHAPE)
    white_image = np.broadcast_to(white_pixel, MNIST_COLOURED_SHAPE)

    digit_pil = transforms.ToPILImage()(digit_image)
    color_image = digit_pil.convert('RGB')
    np_image = np.asarray(color_image).copy()

    if random_background:
        rand_int = random.randint(0, len(MNIST_COLOURED_BACKGROUD)-1)
        background_pixel = MNIST_COLOURED_BACKGROUD[rand_int]
    else:
        fool = random.random() < random_error
        if fool:
            select_background = random.choice([i for i in range(10) if i != label])
        else:
            if particular_background is None:
                select_background = label
            else:
                select_background = particular_background
        background_pixel = MNIST_COLOURED_BACKGROUD[select_background]

    white_mask = np.all(abs(color_image - white_image) <= tolerance, axis=-1)
    black_mask = np.all(abs(color_image - black_image) <= tolerance, axis=-1)
    
    np_image[black_mask] = background_pixel
    np_image[~black_mask] = np.expand_dims((1.0 - np_image[~black_mask][:,0] / 255), axis=1) @ np.expand_dims(np.array(background_pixel), axis=0)

    return np_image

def create_coloured_mnist(path, random_background_train=False, random_mistakes_train = 0.0, random_background_val=False, random_background_test=False, ratio_train_val = 1.0):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_standard = MNIST(root=path, train=True, download=True, transform=transform)

    X_train, y_train = mnist_train_standard.data.numpy(), mnist_train_standard.targets

    (X_train, y_train), (X_val, y_val) = split_np_random(X_train, y_train, ratio=ratio_train_val, seed=8)

    train_images_lst, val_images_lst, test_images_lst = [], [], []
    train_labels_lst, val_labels_lst, test_labels_lst = [], [], []

    for digit_image, label in zip(X_train, y_train):
        
        coloured_image = colour_image(digit_image, label, random_background=random_background_train, random_error=random_mistakes_train)
        train_images_lst.append(coloured_image)
        train_labels_lst.append(label)

    for digit_image, label in zip(X_val, y_val):
        coloured_image = colour_image(digit_image, label, random_background=random_background_val)
        val_images_lst.append(coloured_image)
        val_labels_lst.append(label)
    
    mnist_test_standard = MNIST(root=path, train=False, download=True, transform=transform)
    X_test, y_test = mnist_test_standard.data.numpy(), mnist_test_standard.targets

    for digit_image, label in zip(X_test, y_test):
        coloured_image = colour_image(digit_image, label, random_background=random_background_test)
        test_images_lst.append(coloured_image)
        test_labels_lst.append(label)

    if random_background_train:
        if random_background_val:
            if random_background_test:
                appendix = "_TRrandom_VALrandom_TErandom"
            else:
                appendix = "_TRrandom_VALrandom"
        else:
            if random_background_test:
                appendix = "_TRrandom_TErandom"
            else:
                appendix = "_TRrandom"
            
    else:
        if random_mistakes_train:
            random_appendix = f"_err{random_mistakes_train}"
        else:
            random_appendix = ""
        if random_background_val:
            if random_background_test:
                appendix = random_appendix + "_VALrandom_TErandom"
            else:
                appendix = random_appendix + "_VALrandom"
        else:
            if random_background_test:
                appendix = random_appendix + "_TErandom"
            else:
                appendix = random_appendix + ""

    with h5py.File(f"{path}/coloured_mnist_split{ratio_train_val}{appendix}.h5", 'w') as hf:
        hf.create_dataset('images_train', data=np.array(train_images_lst), chunks=True, compression='gzip')
        hf.create_dataset('labels_train', data=np.array(train_labels_lst), chunks=True, compression='gzip')
        hf.create_dataset('images_val', data=np.array(val_images_lst), chunks=True, compression='gzip')
        hf.create_dataset('labels_val', data=np.array(val_labels_lst), chunks=True, compression='gzip')
        hf.create_dataset('images_test', data=np.array(test_images_lst), chunks=True, compression='gzip')
        hf.create_dataset('labels_test', data=np.array(test_labels_lst), chunks=True, compression='gzip')

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

def center(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    mean = X_combined.mean(0)
    return X_train - mean, X_test - mean, X_val - mean

def scale(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None):
    if X_val is not None:
        X_combined = np.concatenate((X_train, X_val), axis=0)
    else:
        X_combined = X_train
    std = X_combined.std(0)
    return X_train / std, X_test / std, X_val / std

def standardize(X_train: np.ndarray, X_test: np.ndarray, X_val: np.ndarray = None):
    center_X_train, center_X_test, center_X_val = center(X_train, X_test, X_val)
    standardized_X_train, standardized_X_test, standarized_X_val = scale(center_X_train, center_X_test, center_X_val)
    return standardized_X_train, standardized_X_test, standarized_X_val


def load_coloured_mnist_mask(loss: str, datasets_folder=None, 
                             random_background_train=False, random_mistakes_train=0.0, random_background_val=False, random_background_test=False, 
                             ratio_train_val=1.0, mask=None,
                             standardize_data=False, augment=False, normalize=False) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    if random_background_train:
        if random_background_val:
            if random_background_test:
                appendix = "_TRrandom_VALrandom_TErandom"
            else:
                appendix = "_TRrandom_VALrandom"
        else:
            if random_background_test:
                appendix = "_TRrandom_TErandom"
            else:
                appendix = "_TRrandom"
            
    else:
        if random_mistakes_train:
            random_appendix = f"_err{random_mistakes_train}"
        else:
            random_appendix = ""
        if random_background_val:
            if random_background_test:
                appendix = random_appendix + "_VALrandom_TErandom"
            else:
                appendix = random_appendix + "_VALrandom"
        else:
            if random_background_test:
                appendix = random_appendix + "_TErandom"
            else:
                appendix = random_appendix + ""

    potential_file_name = os.path.join(DATASETS_FOLDER, f"coloured_mnist_split{ratio_train_val}{appendix}.h5")
    if os.path.exists(potential_file_name):
        d = h5py.File(potential_file_name, "r")
    else:
        create_coloured_mnist(path=DATASETS_FOLDER, random_background_train=random_background_train, random_mistakes_train=random_mistakes_train,
                                  random_background_val=random_background_val, random_background_test=random_background_test, ratio_train_val=ratio_train_val)
        d = h5py.File(potential_file_name, "r")

    X_train = d.get("images_train")[:].astype(np.float32)
    y_train = d.get("labels_train")[:].astype(np.int64)
    X_val = d.get("images_val")[:].astype(np.float32)
    y_val = d.get("labels_val")[:].astype(np.int64)
    X_test = d.get("images_test")[:].astype(np.float32)
    y_test = d.get("labels_test")[:].astype(np.int64)


    X_train, X_val, X_test = flatten(X_train / 255), flatten(X_val / 255), flatten(X_test / 255)
    y_train, y_val, y_test = make_labels(torch.tensor(y_train), loss), \
        make_labels(torch.tensor(y_val), loss), \
        make_labels(torch.tensor(y_test), loss)
    if standardize_data:
        standardized_X_train, standardized_X_val, standardized_X_test = standardize(X_train, X_test, X_val=X_val, standarize_channel_wise=False)
    else:
        standardized_X_train, standardized_X_val, standardized_X_test = X_train, X_val, X_test
    
    if augment:
        transform = cifar_transform(augment, normalize)
        train = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_train, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_test, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
        if len(X_val):
            val = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_val, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_val)
        else:
            val = AugmentTensorDataset(torch.from_numpy(standardized_X_val), y_val)
    else:
        train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
        if len(X_val):
            val = TensorDataset(torch.from_numpy(unflatten(standardized_X_val, MNIST_COLOURED_SHAPE).transpose((0, 3, 1, 2))).float(), y_val)
        else:
            val = TensorDataset(torch.from_numpy(standardized_X_val), y_val)
    return train, val, test


def generate_custom(digits, digit, c):
    img = colour_image(digits[digit], digit, random_background=False, particular_background=c)
    return img


def predict_particular_coloured_mnist(network, return_dict=True, softmax=False, standardize=False, true_label=True):
    
    DATASETS_FOLDER = os.environ["DATASETS"]
    digits = np.load(os.path.join(DATASETS_FOLDER, "mnist_test_onedigiteach.npy"))
    
    names = []
    d_name_logits = {}

    for i in range(len(MNIST_LABELS)):
        for c in range(len(MNIST_COLOURED_BACKGROUD)):
            name = f"d{i}c{MNIST_COLOURED_BACKGROUD_NAMES[c]}"
            if true_label:
                name += f"t{MNIST_LABELS[i]}"
            img = generate_custom(digits, i, c)
            img = torch.from_numpy(img).to(torch.float32).permute((2, 0, 1)).to(next(network.parameters()).device)
            names.append(name)
            with torch.no_grad():
                v = network(torch.unsqueeze(img, 0))[0, :]
                if softmax:
                    v = torch.softmax(v, dim=0)
                d_name_logits[name] = v

    if return_dict:
        d = {}
        for name in names:
            d[name] = {}
            for i, c in enumerate(MNIST_LABELS):
                d[name][c] = d_name_logits[name][i]
        return d

    return d_name_logits
