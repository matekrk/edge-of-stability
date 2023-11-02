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

import wandb

DEFAULT_PHYS_BS = 1000
RATIO = 50000 // DEFAULT_PHYS_BS
ABRIDGED_SIZE = 5000
DATASETS_FOLDER = "/home/mateusz.pyla/stan/data"
# DATASETS_FOLDER = "/home/mateuszpyla/Pulpit/phd/stan"
DEVICE="cuda"
# DEVICE="cpu"

## UTILS

def overlay_y_on_x_rgb(x, labels):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    num_classes=max(labels)+1
    x_ = x.clone()
    x_[:, :, 0, :num_classes] *= 0.0
    x_[range(x.shape[0]), :, 0, labels.long()] = x.max()
    return x_

def overlay_y_on_x(x, labels):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    num_classes=max(labels)+1
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), labels.long()] = x.max()
    return x_

def overlay_onehot_y_on_x_rgb(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    labels = _from_one_hot(y)
    num_classes=y.shape[1]
    x_ = x.clone()
    x_[:, :, 0, :num_classes] *= 0.0
    x_[range(x.shape[0]), :, 0, labels.long()] = x.max()
    return x_

def overlay_onehot_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    labels = _from_one_hot(y)
    num_classes=y.shape[1]
    x_ = x.clone()
    x_[:, 0, 0, :num_classes] *= 0.0
    x_[range(x.shape[0]), 0, 0, labels.long()] = x.max()
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
        return input.eq(target).float().mean()

def get_loss_and_acc(loss: str, threshold: Optional[int]=None):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    elif loss == "simple":
        return SimpleLoss(threshold), SimpleAccuracy()
    raise NotImplementedError(f"no such loss function: {loss}")

def simple_goodness(h):
    """
    For non-MLP layer, it flattens.
    """
    if len(h.shape) > 2:
        h = torch.flatten(h, start_dim=1)
    return h.pow(2).mean(1)

def calculate_acc_from_goodness(g_pos, g_neg, threshold):
    """
    Given 2 batches of pos and neg data (one number per sample) calculate accuracy w.r.t threshold 
    """
    # positive above threshold
    pos_acc = torch.count_nonzero(torch.greater_equal(g_pos, threshold)) / len(g_pos)
    # negative below threshold
    neg_acc = torch.count_nonzero(torch.less_equal(g_neg, threshold)) / len(g_neg)
    return pos_acc, neg_acc

# SHARP

def compute_hvp(network: nn.Module, goodness_fn, loss_fn: nn.Module,
                dataset, vector, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(nn.utils.parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.to(DEVICE)

    for (X_pos, X_neg, y) in iterate_dataset_posneg(dataset, physical_batch_size):
        X_pos, X_neg, y = X_pos.to(DEVICE), X_neg.to(DEVICE), y.to(DEVICE)
        h_pos, g_pos = network.full_forward(X_pos, goodness_fn)
        h_neg, g_neg = network.full_forward(X_neg, goodness_fn)
        loss = loss_fn(g_pos, g_neg)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True, allow_unused=True)
        dot = nn.utils.parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += nn.utils.parameters_to_vector(grads)
    return hvp

def compute_hvp_layer(network: nn.Module, loss_fn: nn.Module,
                dataset, vector, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(nn.utils.parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.to(DEVICE)

    for (X_pos, X_neg, y) in iterate_dataset_posneg(dataset, physical_batch_size):
        X_pos, X_neg, y = X_pos.to(DEVICE), X_neg.to(DEVICE), y.to(DEVICE)
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
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(DEVICE)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(network: nn.Module, goodness_fn, loss_fn: nn.Module, dataset, neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, goodness_fn, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(nn.utils.parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

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

def _from_one_hot(one_hot: Tensor):
    return torch.argmax(one_hot, dim=1)

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
        yield batch_X.to(DEVICE), batch_y.to(DEVICE)

def iterate_dataset_posneg(dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data with positives and negatives."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X_pos, batch_X_neg, batch_y) in loader:
        yield batch_X_pos.to(DEVICE), batch_X_neg.to(DEVICE), batch_y.to(DEVICE)

def load_cifar_help(loss: str, datasets_folder=DATASETS_FOLDER) -> Tuple[TensorDataset, TensorDataset]:

    cifar10_train = CIFAR10(root=datasets_folder, download=True, train=True)
    cifar10_test = CIFAR10(root=datasets_folder, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(standardized_X_train.astype(float)).float(), y_train)
    test = TensorDataset(torch.from_numpy(standardized_X_test.astype(float)).float(), y_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_mnist_help(loss: str, datasets_folder=DATASETS_FOLDER):
    """
    Return train and test MNIST of Bx1xDxD (with D=28) TensorDataset type with pixels in [0, 1]
    """

    mnist_train = MNIST(root=datasets_folder, download=True, train=True)
    mnist_test = MNIST(root=datasets_folder, download=True, train=False)
    X_train, X_test = mnist_train.data / 255, mnist_test.data / 255
    y_train, y_test = make_labels(torch.tensor(mnist_train.targets), loss), \
        make_labels(torch.tensor(mnist_test.targets), loss)
    standardized_X_train, standardized_X_test = X_train, X_test
    train = TensorDataset(standardized_X_train.float().view(-1, 1, 28, 28), y_train)
    test = TensorDataset(standardized_X_test.float().view(-1, 1, 28, 28), y_test)
    return train, test


def load_dataset(dataset_name, loss):
    if dataset_name == "cifar10-5k-1k":
        train, test = load_cifar_help(loss)
        return take_first(train, 5000), take_first(test, 1000)
    
    if dataset_name == "cifar10-50k-1k":
        train, test = load_cifar_help(loss)
        return take_first(train, 50000), take_first(test, 1000)
    
    elif dataset_name == "mnist-5k-1k":
        train, test = load_mnist_help(loss)
        return take_first(train, 5000), take_first(test, 1000)
    
    elif dataset_name == "mnist-50k-1k":
        train, test = load_mnist_help(loss)
        return take_first(train, 50000), take_first(test, 1000)

class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms. Returns loaders (x, y)
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
    """
    TensorDataset with support of transforms. Returns loaders (x_pos, x_neg, y)
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
        X_neg = self.tensors[1]
        if self.transform:
            X_neg = self.transform(X_neg)
        return X_neg

## ARCHI

class Net2D(nn.Module):
    def __init__(self, dims=None, drop1=0.0, drop2=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # chnl-in, out, krnl
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)   # [64*4*4, x]
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)     # 10 classes
        self.pool1 = nn.MaxPool2d(2, stride=2)   # kernel, stride
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        """ 
        Classic inference
        """
        # convolution phase         # x is [bs, 1, 28, 28]
        z = torch.relu(self.conv1(x))   # Size([bs, 32, 24, 24])
        z = self.pool1(z)           # Size([bs, 32, 12, 12])
        z = self.drop1(z)
        z = torch.relu(self.conv2(z))   # Size([bs, 64, 8, 8])
        z = self.pool2(z)           # Size([bs, 64, 4, 4])
        z = z.reshape(-1, 1024)     # Size([bs, 1024])
        z = torch.relu(self.fc1(z))     # Size([bs, 512])
        z = self.drop2(z)
        z = torch.relu(self.fc2(z))     # Size([bs, 256])
        z = self.fc3(z)             # Size([bs, 10])
        return z
    
    def partial_forward(self, x, i, goodness_fn=None):
        """ 
        Accumulated goodness
        """

        if goodness_fn is not None:
            goodness = []

        z = torch.relu(self.conv1(x))
        z = self.pool1(z)
        if goodness_fn is not None:
            goodness += [goodness_fn(z)]
            if i == 0:
                return z, sum(goodness)
        if i == 0:
            return z
        
        # z = self.pool1(z)
        z = torch.relu(self.conv2(z))
        z = self.pool2(z)
        if goodness_fn is not None:
            goodness += [goodness_fn(z)]
            if i == 1:
                return z, sum(goodness)
        if i == 1:
            return z
        
        # z = self.pool2(z)
        z = z.reshape(-1, 1024)
        z = torch.relu(self.fc1(z))
        if goodness_fn is not None:
            goodness += [goodness_fn(z)]
            if i == 2:
                return z, sum(goodness)
        if i == 2:
            return z
        
        z = torch.relu(self.fc2(z))
        if goodness_fn is not None:
            goodness += [goodness_fn(z)]
            if i == 3:
                return z, sum(goodness)
        if i == 3:
            return z

        z = self.fc3(z)
        if goodness_fn is not None:
            goodness += [goodness_fn(z)]
            return z, sum(goodness)
        return z

    def full_forward(self, x, goodness_fn=None):
        """ 
        Accumulated goodness
        """
        return self.partial_forward(x, -1, goodness_fn)
    
    def forward_with_goodness_overlay(self, x, goodness_fn, overlay_fn, numb_all_classes=10):
        """ 
        For all possible overlay compute accumulated goodness.
        """
        goodness_per_label = []
        for label in range(numb_all_classes):
            label_onehot = torch.zeros((len(x), numb_all_classes))
            label_onehot[:, label] = 1
            x = overlay_fn(x, label_onehot)
            
            h, g = self.full_forward(x, goodness_fn)

            goodness_per_label += [g.unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label
    
    def predict(self, x, goodness_fn, overlay_fn, numb_all_classes=10):
        """ 
        Predict class with the largest accumulated goodness.
        """
        goodness_per_label = self.forward_with_goodness_overlay(x, goodness_fn, overlay_fn, numb_all_classes)
        predict_best_label = goodness_per_label.argmax(1)
        del goodness_per_label
        
        return predict_best_label
    
    def train_whole(self, dataset, goodness_fn, overlay_fn, loss, threshold=None, lr=None, num_epochs=None, optimizer="sgd", physical_batch_size=DEFAULT_PHYS_BS, abridged_size=ABRIDGED_SIZE, eos_every=-1):

        if threshold is None:
            threshold = self.threshold

        if lr is None:
            lr = self.lr

        if num_epochs is None:
            num_epochs = self.num_epochs
        
        #FIXME
        num_epochs = max(num_epochs)

        loss_fn, acc_fn = get_loss_and_acc(loss, threshold)

        self.train_loss = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.train_acc = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.train_acc_pos = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.train_acc_neg = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.test_loss = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.test_acc = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.test_acc_pos = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.test_acc_neg = torch.zeros((len(self.layers), RATIO * num_epochs))
        self.sharp_val = torch.zeros((len(self.layers), num_epochs // eos_every)) if eos_every != -1 else np.zeros((len(self.layers), num_epochs))


        train_dataset, test_dataset = load_dataset(dataset, loss)
        abridged_train = take_first(train_dataset, abridged_size)

        for j, (X, y) in enumerate(iterate_dataset(train_dataset, len(train_dataset))):
            assert j == 0 # one full batch #FIXME later
            y_train = y.detach()
            X, y = X.to(DEVICE), y.to(DEVICE)
            X_pos = overlay_fn(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_fn(X, y[rnd])
        current_dataset = CustomTensorDatasetPosNeg(X_pos, X_neg, y_train)
        
        for (X, y) in iterate_dataset(test_dataset, len(test_dataset)):
            y_test = y.detach()
            X, y = X.to(DEVICE), y.to(DEVICE)
            X_pos = overlay_fn(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_fn(X, y[rnd])
        current_test = CustomTensorDatasetPosNeg(X_pos, X_neg, y_test)

        for (X, y) in iterate_dataset(abridged_train, abridged_size):
            y_abridged = y.detach()
            X, y = X.to(DEVICE), y.to(DEVICE)
            X_pos = overlay_fn(X, y)
            rnd = torch.randperm(X.size(0))
            X_neg = overlay_fn(X, y[rnd])
        current_abridged = CustomTensorDatasetPosNeg(X_pos, X_neg, y_abridged)

        
        for i, layer in enumerate(self.layers):

            print('eval layer', i, ' before training...')
            acc_train = self.eval(abridged_train, goodness_fn, overlay_fn, loss, threshold, physical_batch_size=len(abridged_train))
            acc_test = self.eval(test_dataset, goodness_fn, overlay_fn, loss, threshold, physical_batch_size=len(test_dataset))
            print('not trained layer,', i, '...', '|train acc|', acc_train.item(), '|test acc|', acc_test.item())

            print('training layer', i, '...')

            if optimizer == "sgd":
                layer.opt = SGD(layer.parameters(), lr=lr)
            elif optimizer == "adam":
                layer.opt = Adam(layer.parameters(), lr=lr)

            wandb.define_metric(f"layer_{i}/train/step")
            wandb.define_metric(f"layer_{i}/train/*", step_metric=f"layer_{i}/train/step")
            wandb.define_metric(f"layer_{i}/test/*", step_metric=f"layer_{i}/test/step")

            layer.train()

            for epoch in tqdm(range(num_epochs)):

                for iter, (X_pos, X_neg, y) in enumerate(iterate_dataset_posneg(current_dataset, physical_batch_size)):

                    lastweight0 = layer.weight[0].clone()
                    layer.opt.zero_grad()
                    
                    H_pos = self.partial_forward(X_pos, i)
                    g_pos = goodness_fn(H_pos) # Only current goodness matters
                    H_neg = self.partial_forward(X_neg, i)
                    g_neg = goodness_fn(H_neg) # Only current goodness matters

                    loss_value = loss_fn(g_pos, g_neg)
                    self.train_loss[i, epoch*RATIO+iter] = loss_value.item()

                    ap, an = calculate_acc_from_goodness(g_pos, g_neg, threshold)
                    ap, an = ap.item(), an.item()
                    self.train_acc_pos[i, epoch*RATIO+iter] = ap
                    self.train_acc_neg[i, epoch*RATIO+iter] = an

                    #for _, (X, y) in enumerate(tqdm(iterate_dataset(abridged_train, abridged_size))):
                        # y_hat = self.predict(X, goodness_fn, overlay_fn, numb_all_classes=10) # All goodness matters
                        # acc = acc_fn(y.argmax(1), y_hat)
                    acc = self.eval(abridged_train, goodness_fn, overlay_fn, loss, threshold, physical_batch_size=len(abridged_train))
                    self.train_acc[i, epoch*RATIO+iter] = acc.item()

                    wandb.log({f'layer_{i}/train/step': epoch*RATIO+iter, f'layer_{i}/train/step_epoch': epoch, f'layer_{i}/train/loss': self.train_loss[i, epoch*RATIO+iter], f'layer_{i}/train/goodness_pos_mean': g_pos.mean().item(), f'layer_{i}/train/goodness_neg_mean': g_neg.mean().item(), f'layer_{i}/train/acc_pos': ap, f'layer_{i}/train/acc_neg': an, f'layer_{i}/train/acc': self.train_acc[i, epoch*RATIO+iter]})

                    if epoch % eos_every == 0 and iter == 0 and current_test is not None:
                        test_losses, acces_test, acces_test_pos, acces_test_neg = [], [], [], []

                        for _, (X, y) in enumerate(tqdm(iterate_dataset(test_dataset, len(test_dataset)))):
                            y_hat = self.predict(X, goodness_fn, overlay_fn, numb_all_classes=10)
                            acc = acc_fn(y.argmax(1), y_hat)
                            acces_test.append(acc)

                        for _, (X_pos, X_neg, y) in enumerate(tqdm(iterate_dataset_posneg(current_test, len(current_test)))):
                            g_pos = goodness_fn(self.partial_forward(X_pos, i))
                            g_neg = goodness_fn(self.partial_forward(X_neg, i))

                            test_losses.append(loss_fn(g_pos, g_neg).item())
                            ap, an = calculate_acc_from_goodness(g_pos, g_neg, threshold)
                            acces_test_pos.append(ap.item())
                            acces_test_neg.append(an.item())

                        loss_test_value = sum(test_losses) / len(test_losses)
                        acc_test_value = sum(acces_test) / len(acces_test)
                        acc_test_value_pos = sum(acces_test_pos) / len(acces_test_pos)
                        acc_test_value_neg = sum(acces_test_neg) / len(acces_test_neg)

                        self.test_loss[i, epoch*RATIO+iter] = loss_test_value
                        self.test_acc[i, epoch*RATIO+iter] = acc_test_value
                        self.test_acc_pos[i, epoch*RATIO+iter] = acc_test_value_pos 
                        self.test_acc_neg[i, epoch*RATIO+iter] = acc_test_value_neg
                        wandb.log({f'layer_{i}/test/step': epoch*RATIO+iter, f'layer_{i}/test/loss': self.test_loss[i, epoch*RATIO+iter], f'layer_{i}/test/acc_pos': acc_test_value_pos, f'layer_{i}/test/acc_neg': acc_test_value_neg, f'layer_{i}/test/acc': self.test_acc[i, epoch*RATIO+iter]})

                    if epoch % eos_every == 0 and iter == 0:
                        evals = get_hessian_eigenvalues(self, goodness_fn, loss_fn, current_abridged, neigs=5, physical_batch_size=physical_batch_size)
                        self.sharp_val[i, epoch//eos_every] = evals[0]
                        if evals[0].isnan():
                            print("SHARPNESS NAN!")
                        print(f"Sharpness {evals[0]}")

                        wandb.log({f'sharpness/train/step': (i*num_epochs*RATIO)+epoch*RATIO+iter, f'sharpness/train/e1': evals[0], f'sharpness/train/e2': evals[1], f'sharpness/train/e3': evals[2], f'sharpness/train/e4': evals[3], f'sharpness/train/e5': evals[4]})
                        wandb.log({f'layer_{i}/train/step': epoch*RATIO+iter, f'layer_{i}/train/e1': evals[0], f'layer_{i}/train/e2': evals[1], f'layer_{i}/train/e3': evals[2], f'layer_{i}/train/e4': evals[3], f'layer_{i}/train/e5': evals[4]})

                    loss_value.backward()
                    layer.opt.step()
                    wandb.log({f'layer_{i}/train/step': epoch*RATIO+iter, f'layer_{i}/train/diff': (layer.weight[0] - lastweight0).mean().item()})

                wandb.log({f'layer_{i}/train/step': epoch*RATIO+iter, f'layer_{i}/train/step_epoch': epoch, f'layer_{i}/train/loss_epoch': self.train_loss[i, epoch*RATIO:epoch*RATIO+iter].mean(), f'layer_{i}/train/acc_epoch': self.train_acc[i, epoch*RATIO:epoch*RATIO+iter].mean(), f'layer_{i}/test/step': epoch*RATIO+iter, f'layer_{i}/test/loss_epoch': self.test_loss[i, epoch*RATIO:epoch*RATIO+iter].mean(), f'layer_{i}/test/acc_epoch': self.test_acc[i, epoch*RATIO:epoch*RATIO+iter].mean()})

            print('eval layer', i, ' after training...')
            acc_train = self.eval(abridged_train, goodness_fn, overlay_fn, loss, threshold, physical_batch_size=len(abridged_train))
            acc_test = self.eval(test_dataset, goodness_fn, overlay_fn, loss, threshold, physical_batch_size=len(test_dataset))
            print('trained layer,', i, '...', '|train acc|', acc_train.item(), '|test acc|', acc_test.item())

            if i == 1:
                break

            #FIXME
            # lr /= 2

        return self.train_loss, self.test_loss, self.train_acc, self.test_acc, self.sharp_val

    
    @torch.no_grad()
    def eval(self, dataset, goodness_fn, overlay_fn, loss, threshold=None, physical_batch_size=DEFAULT_PHYS_BS):

        if threshold is None:
            threshold = self.threshold

        loss_fn, acc_fn = get_loss_and_acc(loss, threshold)

        for j, (X, y) in enumerate(tqdm(iterate_dataset(dataset, len(dataset)))):
            y_hat = self.predict(X, goodness_fn, overlay_fn)
            acc = acc_fn(y.argmax(1), y_hat)

        return acc
        


## PLOT

def plot_rez_2(lst_losss, lst_sharp, lst_acc_train, lst_acc_test, path):
    fig = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(2, 2, 1)
    plt.plot(lst_losss[0])
    plt.title("train loss")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 2, 2)
    plt.plot(lst_sharp[0])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 2, 3)
    plt.plot(lst_acc_train[0])
    plt.title("train acc")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 2, 4)
    plt.plot(lst_acc_test[0])
    plt.title("test acc")
    plt.xlabel("iteration layer 0")

    plt.tight_layout()
    wandb.log({'rez': wandb.Image(plt.gcf())}) # caption="Aggregation"
    plt.savefig(path)

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
    plt.plot(lst_sharp[1])
    plt.title("train sharpness")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 3, 4)
    plt.plot(lst_losss[1])
    plt.title("train loss")
    plt.xlabel("iteration layer 1")

    plt.subplot(2, 3, 5)
    plt.plot(lst_acc_train[0])
    plt.title("train acc")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 3, 6)
    plt.plot(lst_acc_test[0])
    plt.title("test acc")
    plt.xlabel("iteration layer 0")

    plt.tight_layout()
    wandb.log({'rez': wandb.Image(plt.gcf())})
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
    plt.plot(lst_acc_train[0])
    plt.title("train acc")
    plt.xlabel("iteration layer 0")

    plt.subplot(2, 4, 8)
    plt.plot(lst_acc_test[0])
    plt.title("test acc")
    plt.xlabel("iteration layer 0")

    plt.tight_layout()
    wandb.log({'rez': wandb.Image(plt.gcf())})
    plt.savefig(path)

    #plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    #plt.axhline(2. / gd_lr, linestyle='dotted')


## MAIN

def main():

    f = open("/home/mateusz.pyla/stan/edge-of-stability/wandb_key.txt", "r")
    wandb_key = f.read()
    f = open("/home/mateusz.pyla/stan/edge-of-stability/wandb_entity.txt", "r")
    wandb_entity = f.read()
    wandb.login(key=wandb_key)
    wandb_project = "FFA_EOS_CONV"
    wandb_log = "/home/mateusz.pyla/stan/edge-of-stability/FFA"
    
    cifar = False

    if cifar:
        dataset="cifar10-50k-1k"
        overlay_fn = overlay_onehot_y_on_x_rgb
    else:
        dataset="mnist-50k-1k"
        overlay_fn = overlay_onehot_y_on_x

    loss="simple"
    goodness_fn=simple_goodness
    threshold=1.0
    lr=0.3

    eos_every = 10

    seed = 1

    torch.manual_seed(seed)
    num_epochs = 50
    epochs = [num_epochs] * (2+3)
    if len(epochs) == 1:
        plot_fn = plot_rez_2
    elif len(epochs) == 2:
        plot_fn = plot_rez_3
    else:
        plot_fn = plot_rez_4
    path = "/home/mateusz.pyla/stan/edge-of-stability/FFA/"
    file_name = f"FFAconv_data{dataset}_layers{len(epochs)}_epochs{num_epochs}_threshold{threshold}_lr{lr}_eos{eos_every}_seed{seed}.png"
    
    config = {"data": dataset, "loss": loss, "threshold": threshold, "lr": lr, "epochs": num_epochs, "eos_every": eos_every, "len_layers": len(epochs), "file_name": file_name, "seed": seed}
    run = wandb.init(project=wandb_project, entity=wandb_entity, dir=wandb_log, config=config) # mode="disabled"

    net = Net2D()
    net = net.to(DEVICE)

    lst_losss_train, lst_losss_test, lst_acc_train, lst_acc_test, lst_sharp = net.train_whole(dataset, goodness_fn, overlay_fn, loss, threshold, lr, epochs, eos_every=eos_every)

    # print(lst_losss_train, lst_losss_test, lst_acc_train, lst_acc_test, lst_sharp)
    
    # plot_fn(lst_losss_train, lst_sharp, lst_acc_train, lst_acc_test, path + file_name)

if __name__ == "__main__":
    main()
