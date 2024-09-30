import os
import random
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from data_cifar import make_labels, flatten
# from data_generic import take_first, extract_subset
from utilities import load_txt

DIR_CORRUPTION = "CIFAR-10-C"
FILE_CORRUPTION_TYPES = "corruptions.txt"
stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, type : str, datasets_folder : str = None,
                 transform=None, target_transform=None):
        if datasets_folder is None:
            if "DATASETS" in os.environ:
                DATASETS_FOLDER = os.environ["DATASETS"]
            else:
                DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        else:
            DATASETS_FOLDER = datasets_folder


        corruptions = load_txt(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION, FILE_CORRUPTION_TYPES))
        assert type in corruptions
        super(CIFAR10C, self).__init__(
            DATASETS_FOLDER, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), type + '.npy')
        target_path = os.path.join(os.path.join(DATASETS_FOLDER, DIR_CORRUPTION), 'labels.npy')
        
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


def cifar_c_transform(augment, normalize):
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

def load_cifar_corrupted(loss, corruptions, serverities, datasets_folder = None, ratio_train_test=1.0, take_first_num = 0, subset_ratio=1.0, standardize_data=False, standarize_channel_wise=False, augment=False, normalize=False):
    if datasets_folder is None:
        if "DATASETS" in os.environ:
            DATASETS_FOLDER = os.environ["DATASETS"]
        else:
            DATASETS_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        DATASETS_FOLDER = datasets_folder

    X_train, y_train = from_cifar_do_data()

    transform_test = cifar_c_transform(augment=augment, normalize=normalize)

    all_dataset = []
    for corrupt in corruptions:
        dataset = CIFAR10C(corrupt, DATASETS_FOLDER, transform_test)

        for serverity in serverities:

            if serverity != 0:
                dataset = dataset[(serverity-1)*1000:serverity*1000]

            X_test, y_test = dataset.data.numpy(), dataset.targets
            (X_train, y_train), (X_val, y_val) = split_np_random(X_train, y_train, ratio=ratio_train_test, seed=8)

        if take_first_num:
            dataset = take_first(dataset, take_first_num)
        if subset_ratio != 1.0:
            dataset = extract_subset(dataset, int(subset_ratio*len(dataset)), random_subset=True)

        X_train, X_val = flatten(X_train.data / 255), flatten(X_val.data / 255)
        y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
            make_labels(torch.tensor(cifar10_test.targets), loss)
        if standardize_data:
            standardized_X_train, standardized_X_test = standardize(X_train, X_test, standarize_channel_wise)
        else:
            standardized_X_train, standardized_X_test = X_train, X_test
        
        all_dataset.append(dataset)
        
    all_dataset = ConcatDataset(all_dataset)


    #with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
    #    for ci, cname in enumerate(opt.corruptions):
    # if cname == 'natural':
    #pass
