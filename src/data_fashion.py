import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
import torch.nn.functional as F

from data_cifar import AugmentTensorDataset, flatten, make_labels, center, standardize, unflatten

FASHION_SHAPE = (28, 28, 1)
FASHION_LABELS = ('top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot')

def load_mnist_mask(loss: str, datasets_folder=None, mask=None) -> Tuple[TensorDataset, TensorDataset]:
    if datasets_folder is None:
        DATASETS_FOLDER = os.environ["DATASETS"]

    mnist_train = FashionMNIST(root=DATASETS_FOLDER, download=True, train=True)
    if mask is not None:
        mnist_train = torch.utils.data.Subset(mnist_train, mask)
    mnist_test = FashionMNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data.numpy() / 255), flatten(mnist_test.data.numpy() / 255)
    y_train, y_test = make_labels(mnist_train.targets.clone().detach(), loss), \
        make_labels(mnist_test.targets.clone().detach(), loss)
    standardized_X_train, standardized_X_test = X_train, X_test
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def fashion_transform(augment):
    
    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [  transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # transforms.RandomAffine(0),
                            transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
                            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                            ] + transform_list
    
    return transforms.Compose([transform_list])

def load_fashion_mask(loss: str, datasets_folder=None, standardize_data=False, standarize_channel_wise=False, augment=False, normalize=False, mask=None) -> Tuple[TensorDataset, TensorDataset]:
    if datasets_folder is None:
        DATASETS_FOLDER = os.environ["DATASETS"]

    mnist_train = FashionMNIST(root=DATASETS_FOLDER, download=True, train=True)
    mnist_test = FashionMNIST(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(mnist_train.data.numpy() / 255), flatten(mnist_test.data.numpy() / 255)
    y_train, y_test = make_labels(mnist_train.targets.clone().detach(), loss), \
        make_labels(mnist_test.targets.clone().detach(), loss)
    
    if standardize_data:
        standardized_X_train, standardized_X_test = standardize(X_train, X_test, standarize_channel_wise)
    else:
        standardized_X_train, standardized_X_test = X_train, X_test

    if augment:
        transform = fashion_transform(augment, normalize)
        train = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_train, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = AugmentTensorDataset(transform, torch.from_numpy(unflatten(standardized_X_test, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)
    else:
        train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_train)
        test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, FASHION_SHAPE).transpose((0, 3, 1, 2))).float(), y_test)

    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (28, 28, 1)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test



def generate_custom(option):

    black_pixel = 0
    mix_pixel = 0.5
    white_pixel = 1

    pixel = black_pixel
    img = np.full(FASHION_SHAPE, pixel).transpose(2, 0, 1)
    # img = np.broadcast_to(pixel, MNIST_SHAPE).transpose(2, 0, 1)

    if option == "black":
        return img
    elif option == "checkboard":
        checkerboard = np.indices((1, 28, 28)).sum(axis=0) % 2
        img[checkerboard == 0] = white_pixel
        return img
    elif option == "mid":
        box_size = 8
        start = (28 - box_size) // 2
        end = start + box_size
        img[start:end, start:end] = mix_pixel
        return img

def generate_mean(dataset, n_imgs):
    def mean_image(img1, img2):
        return (img1 + img2) / 2
    
    inds = np.random.choice(range(len(dataset)), 2*n_imgs)

    imgs = np.zeros((n_imgs, *FASHION_SHAPE)) # .reshape(n_imgs, -1)

    for i in range(n_imgs):
        imgs[i] = mean_image(dataset[inds[i]], dataset[inds[-1-i]])

    return imgs

def predict_particular_fashion(network, standardize=False, return_dict=True, softmax=False, blends=False, true_label=True):
    
    DATASETS_FOLDER = os.environ["DATASETS"]
    fashion_mnist_test = FashionMNIST(root=DATASETS_FOLDER, download=True, train=False)
    fashion_mnist_test = fashion_mnist_test.data.numpy() / 255
    fashion_mnist_test = fashion_mnist_test.reshape(len(fashion_mnist_test), *FASHION_SHAPE)
    
    names = []
    d_name_img = {}
    d_name_logits = {}

    if blends:
        means = generate_mean(fashion_mnist_test, 10)

        for i in list(range(10)):
            name = str(i)
            names.append(name)
            d_name_img[name] = torch.tensor(means[i], device=next(network.parameters()).device, dtype=next(network.parameters()).dtype).permute(2, 0, 1)
            with torch.no_grad():
                v = network(torch.unsqueeze(d_name_img[name], 0))[0, :]
                if softmax:
                    v = torch.softmax(v, dim=0)
                d_name_logits[name] = v

    for name in ["black", "checkboard", "mid"]:
        names.append(name)
        d_name_img[name] = torch.tensor(generate_custom(name), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
        with torch.no_grad():
            v = network(torch.unsqueeze(d_name_img[name], 0))[0, :]
            if softmax:
                v = torch.softmax(v, dim=0)
            d_name_logits[name] = v

    if return_dict:
        d = {}
        for name in names:
            d[name] = {}
            for i, c in enumerate(FASHION_LABELS):
                d[name][c] = d_name_logits[name][i]
        return d

    return d_name_logits
