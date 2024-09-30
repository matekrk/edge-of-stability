import random
from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset
from data_cifar import load_cifar, AugmentTensorDataset, possible_cifar_names, predict_particular_cifar, CIFAR_LABELS
from data_cifar_c import load_cifar_corrupted
from data_coloured_mnist import load_coloured_mnist_mask, possible_coloured_names, predict_particular_coloured_mnist
from data_svhn import load_svhn_mask, predict_particular_svhn, SVHN_LABELS
from data_mnist import load_mnist, predict_particular_mnist, MNIST_LABELS
from data_fashion import load_fashion_mask, predict_particular_fashion, FASHION_LABELS
from synthetic import make_chebyshev_dataset, make_linear_dataset
# from wikitext import load_wikitext_2

DATASETS = possible_cifar_names() + \
            possible_coloured_names() + [
    "fashion", "fashion-5k", "fashion-10k", "fashion-20k", 
    "mnist", "mnist-5k", "mnist-10k", "mnist-20k",
    "svhn", "svhn-5k", "svhn-10k", "svhn-20k",
    "chebyshev-3-20", "chebyshev-4-20", "chebyshev-5-20", "linear-50-50"
    ]

def get_labels(dataset):
    if dataset.startswith("cifar"):
        LABELS = CIFAR_LABELS
    elif dataset.startswith("coloured"):
        LABELS = MNIST_LABELS
    elif dataset.startswith("svhn"):
        LABELS = SVHN_LABELS
    elif dataset.startswith("mnist"):
        LABELS = MNIST_LABELS 
    elif dataset.startswith("fashion"):
        LABELS = FASHION_LABELS
    return LABELS

def get_predict_particular(analysis_cifar, analysis_coloured_mnist, analysis_svhn, analysis_mnist, analysis_fashion):
    predict_fns = []
    if analysis_cifar:
        predict_fns.append(predict_particular_cifar)
    elif analysis_coloured_mnist:
        predict_fns.append(predict_particular_coloured_mnist)
    elif analysis_svhn:
        predict_fns.append(predict_particular_svhn)
    elif analysis_mnist:
        predict_fns.append(predict_particular_mnist)
    elif analysis_fashion:
        predict_fns.append(predict_particular_fashion)
    return predict_fns

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def num_input_channels(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10"):
        return 3
    elif dataset_name == 'fashion' or dataset_name == "mnist":
        return 1

def image_size(dataset_name: str) -> int:
    if dataset_name.startswith("cifar10"):
        return 32
    elif dataset_name == 'fashion' or dataset_name == "mnist":
        return 28

def num_classes(dataset_name: str) -> int:
    if dataset_name.startswith('cifar10'):
        return 10
    elif dataset_name == 'fashion'  or dataset_name == "mnist":
        return 10

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))
    else:
        raise NotImplementedError("unknown pooling: {}".format(pooling))

def num_pixels(dataset_name: str) -> int:
    return num_input_channels(dataset_name) * image_size(dataset_name)**2

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)

# FIX val
def load_dataset(dataset_name: str, loss: str) -> Tuple[Union[TensorDataset, AugmentTensorDataset], Union[TensorDataset, AugmentTensorDataset]]:

    if dataset_name.startswith("cifar10"):

        normalize, augment, standarize_channel_wise, stardardize = False, False, False, False

        if dataset_name.endswith("_st"):
            stardardize = True
            if dataset_name[:-3].endswith("_ch"):
                standarize_channel_wise = True
                if dataset_name[:-6].endswith("_augm"):
                    augment = True
            elif dataset_name[:-3].endswith("_augm"):
                augment = True
                if dataset_name[:-5].endswith("_n"):
                    normalize = True
        elif dataset_name.endswith("_augm"):
            augment = True
            if dataset_name[:-5].endswith("_n"):
                normalize = True

        train, test = load_cifar(loss) #, standardize_data=stardardize, standarize_channel_wise=standarize_channel_wise, augment=augment, normalize=normalize)

        if not dataset_name.startswith("cifar10c"):
            corrupted = True
            start_index = 8
        else:
            corrupted = False
            start_index = 10
            serverity = int(dataset_name[8])

        if dataset_name[start_index:].startswith("1k"):
            take_first_train = 1000
        elif dataset_name[start_index:].startswith("2k"):
            take_first_train = 2000
        elif dataset_name[start_index:].startswith("5k"):
            take_first_train = 5000
        elif dataset_name[start_index:].startswith("10k"):
            take_first_train = 10000
            start_index += 1
        else:
            take_first_train = 0
            start_index -= 2

        start_index += 2

        if dataset_name[start_index:].startswith("1k"):
            take_first_test = 1000
        elif dataset_name[start_index:].startswith("2k"):
            take_first_test = 2000
        elif dataset_name[start_index:].startswith("5k"):
            take_first_test = 5000
        elif dataset_name[start_index:].startswith("10k"):
            take_first_test = 10000
        else:
            take_first_test = 0

        if take_first_train:
            train = take_first(train, take_first_train)
        if not corrupted and take_first_test:
            test = take_first(test, take_first_test)

        if corrupted:
            test = load_cifar_corrupted
            # FIXME: finish
        
        return train, test
    
    if dataset_name.startswith("coloured"):

        assert dataset_name.startswith("coloured_mnist_split1.0") # len 23 # FIXME
        index = 23

        if dataset_name[index:].startswith("_err0.05"):
            random_mistakes_train = 0.05
            index += 8
        elif dataset_name[index:].startswith("_err0."):
            flt = dataset_name[index+4:index+7]
            random_mistakes_train = float(flt)
            index += 7

        # names: "coloured_mnist" / "coloured_mnist_random" / "coloured_mnist_T{RE}random" 

        random_background_train, random_background_val, random_background_test = False, False, False
        if dataset_name[index:].startswith("_TRrandom_VALrandom_TErandom"):
            random_background_train = True
            random_background_val = True
            random_background_test = True
            index += 28
        elif dataset_name[index:].startswith("_TRrandom_VALrandom"):
            random_background_train = True
            random_background_val = True
            random_background_test = False
            index += 19
        elif dataset_name[index:].startswith("_TRrandom_TErandom"):
            random_background_train = True
            random_background_val = False
            random_background_test = True
            index += 18
        elif dataset_name[index:].startswith("_TRrandom"):
            random_background_train = True
            random_background_val = False
            random_background_test = False
            index += 9
        elif dataset_name[index:].startswith("_VALrandom_TErandom"):
            random_background_train = False
            random_background_val = True
            random_background_test = True
            index += 19
        elif dataset_name[index:].startswith("_VALrandom"):
            random_background_train = False
            random_background_val = True
            random_background_test = False
            index += 10
        elif dataset_name[index:].startswith("_TErandom"):
            random_background_train = False
            random_background_val = False
            random_background_test = True
            index += 9

        normalize, augment, standarize_channel_wise, stardardize = False, False, False, False

        # FIXME above

        train, val, test = load_coloured_mnist_mask(loss, random_background_train=random_background_train, random_mistakes_train=random_mistakes_train, random_background_val=random_background_val, random_background_test=random_background_test, ratio_train_val=1.0)
        
        if dataset_name[index:].startswith("-1k"):
            return take_first(train, 1000), test
        elif dataset_name[index:].startswith("-1k-1k"):
            return take_first(train, 1000), take_first(test, 1000)
        elif dataset_name[index:].startswith("-2k"):
            return take_first(train, 2000), test
        elif dataset_name[index:].startswith("-2k-1k"):
            return take_first(train, 2000), take_first(test, 1000)
        elif dataset_name[index:].startswith("-5k"):
            return take_first(train, 5000), test
        elif dataset_name[index:].startswith("-5k-1k"):
            return take_first(train, 5000), take_first(test, 1000)
        elif dataset_name[index:].startswith("-10k"):
            return take_first(train, 10000), test
        elif dataset_name[index:].startswith("-10k-1k"):
            return take_first(train, 10000), take_first(test, 1000)
        
        else:
            return train, test

    elif dataset_name == 'svhn':
        return load_svhn_mask(loss)
    elif dataset_name == "svhn-5k":
        train, test = load_svhn_mask(loss)
        return take_first(train, 5000), test
    elif dataset_name == "svhn-5k-1k":
        train, test = load_svhn_mask(loss)
        return take_first(train, 5000), take_first(test, 1000)
    elif dataset_name == "svhn-10k":
        train, test = load_svhn_mask(loss)
        return take_first(train, 10000), test
    elif dataset_name == "svhn-10k-1k":
        train, test = load_svhn_mask(loss)
        return take_first(train, 10000), take_first(test, 1000)
    elif dataset_name == "svhn-20k":
        train, test = load_svhn_mask(loss)
        return take_first(train, 20000), test

    elif dataset_name == 'mnist':
        return load_mnist(loss)
    elif dataset_name == "mnist-5k":
        train, test = load_mnist(loss)
        return take_first(train, 5000), test
    elif dataset_name == "mnist-10k":
        train, test = load_mnist(loss)
        return take_first(train, 10000), test
    elif dataset_name == "mnist-20k":
        train, test = load_mnist(loss)
        return take_first(train, 20000), test
    
    elif dataset_name == 'fashion':
        return load_fashion_mask(loss)
    elif dataset_name == "fashion-5k":
        train, test = load_fashion_mask(loss)
        return take_first(train, 5000), test
    elif dataset_name == "fashion-5k-1k":
        train, test = load_fashion_mask(loss)
        return take_first(train, 5000), take_first(test, 1000)
    elif dataset_name == "fashion-10k":
        train, test = load_fashion_mask(loss)
        return take_first(train, 10000), test
    elif dataset_name == "fashion-10k-1k":
        train, test = load_fashion_mask(loss)
        return take_first(train, 10000), take_first(test, 1000)
    elif dataset_name == "fashion-20k":
        train, test = load_fashion_mask(loss)
        return take_first(train, 20000), test
    
    elif dataset_name == "chebyshev-5-20":
        return make_chebyshev_dataset(k=5, n=20)
    elif dataset_name == "chebyshev-4-20":
        return make_chebyshev_dataset(k=4, n=20)
    elif dataset_name == "chebyshev-3-20":
        return make_chebyshev_dataset(k=3, n=20)
    elif dataset_name == 'linear-50-50':
        return make_linear_dataset(n=50, d=50)

