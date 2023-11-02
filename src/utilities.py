from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os

from relative_space import relative_projection, transform_space

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000

CIFAR_SHAPE = (32, 32, 3)

def create_uniform_image(background, shape):
    if background == "sky":
        pixel = [123, 191, 232]
    elif background == "red":
        pixel = [253, 0, 3]
    elif background == "green":
        pixel = [11, 241, 4]

    assert len(pixel) == shape[-1]
    return np.broadcast_to(pixel, shape).transpose(2, 0, 1)


def get_gd_directory(dataset: str, lr: float, arch_id: str, seed: int, opt: str, loss: str, beta: float = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/{opt}"
    if opt == "gd":
        return f"{directory}/lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        return f"{directory}/lr_{lr}_beta_{beta}"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float) -> Optimizer:
    if opt == "gd":
        return SGD(parameters, lr=lr)
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)

def save_features(features, step, traj_directory):
    torch.save(features, f"{traj_directory}/features_{step}.pt")


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()

def reduced_batch(network: nn.Module, loss_function: nn.Module, X, Y, strategy="norm", gamma=2.0):
    metrics = []
    for i in range(len(X)):
        network.zero_grad()
        x, y = X[i], Y[i]
        loss_val = loss_function(network(torch.unsqueeze(x, 0)), torch.unsqueeze(y, 0))
        loss_val.backward()
        if strategy == "norm":
            total_norm = 0.0
            for p in network.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()
            total_norm = total_norm ** (1. / 2)
            metrics.append(total_norm)
        elif strategy == "sum trace fim":
            total_tr_fim = 0.

    network.zero_grad()
    metrics = np.array(metrics)
    mean_metrics = np.mean(metrics)
    std_metrics = np.std(metrics)
    ind_r = abs(metrics - mean_metrics) < gamma * std_metrics
    return X[ind_r], Y[ind_r]

def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS, return_features=False):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    to_return = []
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            if return_features:
                preds, features = network(X, return_features=return_features)
                to_return.append(features)
            else:
                preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    if return_features:
        return losses, torch.cat(to_return)
    return losses

def compute_space(network: nn.Module, dataset: Dataset, batch_size: int = DEFAULT_PHYS_BS):
    all_features = []
    all_y = []
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds, features = network(X, return_features=True)
            features_transformed = transform_space(features)
            all_features.append(features.view(-1, 512))
            if len(y.shape) > 1:
                all_y.append(torch.flatten(torch.argmax(y, dim=1))) # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1) for CE
            else:
                all_y.append(y)
            #torch.cat((all_features, features_transformed), 0)
            #torch.cat((all_y, y), 0)
    all_features = torch.stack(all_features)
    all_y = torch.stack(all_y)
    return all_features, all_y

def calculate_trajectory_point(features, num_concat=50):
    relevant_features = features[num_concat].reshape(num_concat, -1).flatten()

def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
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


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


def obtained_eos(evalues, lr, window=10, relative_error=0.1):
    if len(evalues) < window:
        return False
    convergence = 2/lr
    for i in range(1, window+1):
        if abs(evalues[-i] - convergence) > relative_error * convergence:
            return False
    return True

def compute_logits(network: nn.Module, loss_functions: List[nn.Module]):
    L = len(loss_functions)
    
    sky = torch.tensor(create_uniform_image("sky", CIFAR_SHAPE), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
    red = torch.tensor(create_uniform_image("red", CIFAR_SHAPE), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)
    green = torch.tensor(create_uniform_image("green", CIFAR_SHAPE), device=next(network.parameters()).device, dtype=next(network.parameters()).dtype)

    losses_sky = [0. for _ in range(10)]
    losses_red = [0. for _ in range(10)]
    losses_green = [0. for _ in range(10)]
    with torch.no_grad():
        pred = network(torch.unsqueeze(sky, 0))
        for y in range(10):
            losses_sky[y] = pred[0, y].item()
        pred = network(torch.unsqueeze(red, 0))
        for y in range(10):
            losses_red[y] = pred[0, y].item()
        pred = network(torch.unsqueeze(green, 0))
        for y in range(10):
            losses_green[y] = pred[0, y].item()

    return torch.tensor(losses_sky), torch.tensor(losses_red), torch.tensor(losses_green)


def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device='cuda')
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient


class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)


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

