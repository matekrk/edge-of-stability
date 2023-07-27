from tqdm import tqdm
from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize, RandomCrop
from torch.utils.data.sampler import SubsetRandomSampler

DEFAULT_PHYS_BS = 1000
RATIO = 5000 // DEFAULT_PHYS_BS
ABRIDGED_SIZE = 5000
DATASETS_FOLDER = "/home/mateusz.pyla/stan/data"

## UTILS

def overlay_y_on_x_rgb(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :, 0, :10] *= 0.0
    #x_[range(x.shape[0]), :, 0, y.long()] = x.max()
    for i in range(x.shape[0]):
        x_[i, :, 0, y[i].long()] = x.max()
    return x_

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data (RGB) [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    for i in range(x.shape[0]):
        x_[i, y[i].long()] = x.max()
    #x_[range(x.shape[0]), y.long()] = x.max()
    return x_

class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()

class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()

class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()

class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0

class SimpleLoss(nn.Module):
    def __init__(self, threshold=1.0):
        super(SimpleLoss, self).__init__()
        self.threshold = threshold

    def forward(self, g_pos, g_neg, threshold=None):
        if threshold is None:
            threshold = self.threshold
        return torch.log(1 + torch.exp(torch.cat([
                    -g_pos + threshold,
                    g_neg - threshold]))).mean()

class SimpleAccuracy(nn.Module):
    def __init__(self):
        super(SimpleAccuracy, self).__init__()

    def forward(self, input, target):
        return input.eq(target).float().mean() # input = net.predict(x)

def get_loss_and_acc(loss: str, threshold: Optional[int]=None):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    elif loss == "simple":
        return SimpleLoss(threshold), SimpleAccuracy()
    raise NotImplementedError(f"no such loss function: {loss}")

# SHARP

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset, vector, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(nn.utils.parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()

    for (X_pos, X_neg, y) in iterate_dataset_posneg(dataset, physical_batch_size):
        X_pos, X_neg, y = X_pos.cuda(), X_neg.cuda(), y.cuda()
        g_pos = network.forward(X_pos).pow(2).mean(1)
        g_neg = network.forward(X_neg).pow(2).mean(1)
        loss = loss_fn(g_pos, g_neg)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = nn.utils.parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += nn.utils.parameters_to_vector(grads)
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset, neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(nn.utils.parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    to_return = []
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses

## DATA

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
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
    elif loss == "simple":
        return _one_hot(y, 10, 0)

def iterate_dataset(dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()

def iterate_dataset_posneg(dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data with positives and negatives."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X_pos, batch_X_neg, batch_y) in loader:
        yield batch_X_pos.cuda(), batch_X_neg.cuda(), batch_y.cuda()

def load_cifar_help(loss: str, datasets_folder=DATASETS_FOLDER) -> Tuple[TensorDataset, TensorDataset]:

    cifar10_train = CIFAR10(root=datasets_folder, download=True, train=True)
    cifar10_test = CIFAR10(root=datasets_folder, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(standardized_X_train.astype(float)), y_train)
    test = TensorDataset(torch.from_numpy(standardized_X_test.astype(float)), y_test)
    #train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    #test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_dataset(dataset_name, loss):
    if dataset_name == "cifar10-5k-1k":
        train, test = load_cifar_help(loss)
        return take_first(train, 5000), take_first(test, 1000)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
class CustomTensorDatasetPosNeg(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, h_pos, h_neg, y, transform=None):
        tensors = [h_pos, h_neg, y]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x_pos = self.tensors[0][index]
        x_neg = self.tensors[1][index]
        y = self.tensors[2][index]
        if self.transform:
            x_pos = self.transform(x_pos)
            x_neg = self.transform(x_neg)
        return x_pos, x_neg, y

    def __len__(self):
        return self.tensors[0].size(0)
    
    def get_positives(self):
        X_pos = self.tensors[0]
        if self.transform:
            X_pos = self.transform(X_pos)
        return X_pos
    
    def get_negatives(self):
        X_neg = self.tensors[0]
        if self.transform:
            X_neg = self.transform(X_neg)
        return X_neg

def old_MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def old_CIFAR_loaders(train_batch_size=50000, test_batch_size=10000):

    """
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
    ])
    """
    transform_train = Compose([
    #RandomCrop(32, padding=4),
    Resize(32),
    #RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = Compose([
        Resize(32),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    #dataset_valid = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_valid)
    dataset_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_train = len(dataset_train)
    #indices = list(range(num_train))
    #split = int(np.floor(args.valid_size * num_train))
    #train_idx, valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size) #, sampler=train_sampler)
    #valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=test_batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader


## ARCHI

class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

        self.threshold = 2

        self.num_epochs = len(self.layers) * [1]

    def predict(self, x, numb_all_classes=10):
        goodness_per_label = []
        for label in range(numb_all_classes):
            label_onehot = torch.zeros((len(x), numb_all_classes))
            label_onehot[:, label] = 1
            h = overlay_y_on_x(x, label_onehot)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, dataset, loss, num_epochs=None, physical_batch_size=DEFAULT_PHYS_BS, abridged_size=ABRIDGED_SIZE, eos_every=-1):

        if num_epochs is None:
            num_epochs = self.num_epochs

        lst_losss, lst_sharp, lst_losss_test, lst_sharp_test = [], [], [], []

        lst_acc_train, lst_acc_test = [], []

        train_dataset, test_dataset = load_dataset(dataset, loss)
        abridged_train = take_first(train_dataset, abridged_size)

        for j, (X, y) in enumerate(iterate_dataset(train_dataset, len(train_dataset))):
            assert j == 0 # one full batch #FIXME later
            y_train = y.detach()
            X, y = X.cuda(), y.cuda()
            X_pos = overlay_y_on_x(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_y_on_x(X, y[rnd])
        current_dataset = CustomTensorDatasetPosNeg(X_pos, X_neg, y)
        
        """
        for (X, y) in iterate_dataset(test_dataset, len(test_dataset)):
            y_test = y.detach()
            X, y = X.cuda(), y.cuda()
            X_pos = overlay_y_on_x(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_y_on_x(X, y[rnd])
        current_test = CustomTensorDatasetPosNeg(X_pos, X_neg, y)
        """

        for (X, y) in iterate_dataset(abridged_train, abridged_size):
            y_abridged = y.detach()
            X, y = X.cuda(), y.cuda()
            X_pos = overlay_y_on_x(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_y_on_x(X, y[rnd])
        current_abridged = CustomTensorDatasetPosNeg(X_pos, X_neg, y)

        acc_train = self.eval(abridged_train, loss, physical_batch_size=len(abridged_train))
        acc_test = self.eval(test_dataset, loss, physical_batch_size=len(test_dataset))

        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            loss_val, sharp_val = layer.train(current_dataset, current_abridged, loss, physical_batch_size=physical_batch_size, num_epochs=num_epochs[i], test=test_dataset, eos_every=eos_every)
            
            acc_train = self.eval(abridged_train, loss, physical_batch_size=len(abridged_train))
            acc_test = self.eval(test_dataset, loss, physical_batch_size=len(test_dataset))

            lst_losss.append(loss_val.detach().numpy())
            lst_sharp.append(sharp_val.detach().numpy())
            lst_acc_train.append(acc_train.item())
            lst_acc_test.append(acc_test.item())

            with torch.no_grad():
                current_dataset = CustomTensorDatasetPosNeg(layer.forward(current_dataset.get_positives()), layer.forward(current_dataset.get_negatives()), y_train)
                # current_test_dataset = CustomTensorDatasetPosNeg(layer.forward(current_test.get_positives()), layer.forward(current_test.get_negatives()), y_test)
                current_abridged = CustomTensorDatasetPosNeg(layer.forward(current_abridged.get_positives()), layer.forward(current_abridged.get_negatives()), y_abridged)

        # return lst_h_pos, lst_h_neg, lst_losss, lst_sharp
        return lst_losss, lst_sharp, lst_acc_train, lst_acc_test
    
    @torch.no_grad()
    def eval(self, dataset, loss, physical_batch_size=DEFAULT_PHYS_BS):

        loss_fn, acc_fn = get_loss_and_acc(loss, threshold=self.threshold)

        for j, (X, y) in enumerate(tqdm(iterate_dataset(dataset, len(dataset)))):
            y_hat = self.predict(X)
            acc = acc_fn(y.argmax(1), y_hat)

        return acc
        


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = SGD(self.parameters(), lr=0.3) # Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T.double()) +
            self.bias.unsqueeze(0))

    def train(self, current_posneg, current_posneg_abridged, loss, physical_batch_size=DEFAULT_PHYS_BS, num_epochs=None, test=None, eos_every=-1):

        if num_epochs is None:
            num_epochs = self.num_epochs

        if eos_every != -1:
            self.sharpness = torch.zeros(RATIO * num_epochs // eos_every)
        else:
            self.sharpness = torch.zeros(RATIO * num_epochs)
        self.train_loss = torch.zeros(RATIO * num_epochs)
        self.train_acc = torch.zeros(RATIO * num_epochs)
        self.test_loss = torch.zeros(RATIO * num_epochs)
        self.test_acc = torch.zeros(RATIO * num_epochs)

        loss_fn, acc_fn = get_loss_and_acc(loss, threshold=self.threshold)

        for i in tqdm(range(num_epochs)):
            for j, (H_pos, H_neg, y) in enumerate(tqdm(iterate_dataset_posneg(current_posneg, physical_batch_size))):
                
                g_pos = self.forward(H_pos).pow(2).mean(1)
                g_neg = self.forward(H_neg).pow(2).mean(1)

                # The following loss pushes pos (neg) samples to
                # values larger (smaller) than the self.threshold.
                loss_val = loss_fn(g_pos, g_neg)
                self.train_loss[i+j] = loss_val
                #loss = torch.log(1 + torch.exp(torch.cat([
                #    -g_pos + self.threshold,
                #    g_neg - self.threshold]))).mean()
                
                if i % eos_every == 0 and j == 0:
                    evals = get_hessian_eigenvalues(self, loss_fn, current_posneg_abridged, neigs=2, physical_batch_size=physical_batch_size)
                    self.sharpness[i//eos_every] = evals[0]

                self.opt.zero_grad()
                # this backward just compute the derivative and hence
                # is not considered backpropagation.
                loss_val.backward()
                self.opt.step()

                #if test is not None:
                #    self.evaluate(test)

        return self.train_loss, self.sharpness

    @torch.no_grad()
    def evaluate(self, test_dataset, loss):

        loss_fn, acc_fn = get_loss_and_acc(loss, threshold=self.threshold)

        for j, (H_pos, H_neg, y) in enumerate(tqdm(iterate_dataset_posneg(test_dataset, len(test_dataset)))):
            g_pos = self.forward(H_pos).pow(2).mean(1)
            g_neg = self.forward(H_neg).pow(2).mean(1)

            # FIXME what now maybe linear probing?

        return

## PLOT

def plot_rez_3(lst_losss, lst_sharp, lst_acc_train, lst_acc_test, path):
    fig = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(2, 3, 1)
    plt.plot(lst_losss[0])
    plt.title("train loss")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 3, 2)
    plt.plot(lst_sharp[0])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 3, 3)
    plt.plot(lst_losss[1])
    plt.title("train loss")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 3, 4)
    plt.plot(lst_sharp[1])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 3, 5)
    plt.plot(lst_acc_train)
    plt.title("train acc")
    plt.xlabel("layer")

    plt.subplot(2, 3, 6)
    plt.plot(lst_acc_test)
    plt.title("test acc")
    plt.xlabel("layer")

    plt.tight_layout()

    plt.savefig(path)

def plot_rez_4(lst_losss, lst_sharp, lst_acc_train, lst_acc_test, path):
    fig = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(2, 4, 1)
    plt.plot(lst_losss[0])
    plt.title("train loss")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 4, 2)
    plt.plot(lst_sharp[0])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 4, 3)
    plt.plot(lst_losss[1])
    plt.title("train loss")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 4, 4)
    plt.plot(lst_sharp[1])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 4, 5)
    plt.plot(lst_losss[2])
    plt.title("train loss")
    plt.xlabel("iteration layer 2")

    plt.subplot(2, 4, 6)
    plt.plot(lst_sharp[2])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 2")

    plt.subplot(2, 4, 7)
    plt.plot(lst_acc_train)
    plt.title("train acc")
    plt.xlabel("layer")

    plt.subplot(2, 4, 8)
    plt.plot(lst_acc_test)
    plt.title("test acc")
    plt.xlabel("layer")

    plt.tight_layout()

    plt.savefig(path)

    #plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    #plt.axhline(2. / gd_lr, linestyle='dotted')


## MAIN

def main_old():
    torch.manual_seed(1234)
    train_loader, test_loader = old_MNIST_loaders()

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())    
    

def main():

    dataset="cifar10-5k-1k"
    loss="simple"

    torch.manual_seed(1234)
    neurons = [3072, 500, 500]
    #epochs = [10, 10]
    epochs = [1000, 1000]
    net = Net(neurons)
    
    lst_losss, lst_sharp, lst_acc_train, lst_acc_test = net.train(dataset, loss, epochs, eos_every=-1)

    print(lst_losss, lst_sharp, lst_acc_train, lst_acc_test)

    path = "/home/mateusz.pyla/stan/edge-of-stability/figures/ffa/3072_500_500_D.png"
    plot_rez_3(lst_losss, lst_sharp, lst_acc_train, lst_acc_test, path)

if __name__ == "__main__":
    # main_old()
    main()
