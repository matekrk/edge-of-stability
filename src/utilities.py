from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim import swa_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os

from sam import SAM

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000


def get_gd_directory(dataset: str, arch_id: str, loss: str, opt: str, lr: float, eig_freq: int, seed: int, 
                     beta: float = None, delta: float = None, start_step: int = 0):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/{loss}/{opt}/"
    if opt == "sgd":
        directory += f"lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        directory += f"{directory}/lr_{lr}_beta_{beta}"
    if delta is None:
        directory += "/"
    else:
        directory += f"_delta{delta}/"
    directory += f"seed_{seed}/"
    return f"{directory}freq_{eig_freq}/start_{start_step}/"


def get_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/flow/tick_{tick}"


def get_modified_flow_directory(dataset: str, arch_id: str, seed: int, loss: str, gd_lr: float, tick: float):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]
    return f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/modified_flow_lr_{gd_lr}/tick_{tick}"


def get_gd_params(model_parameters, base_lr, last_layer: bool = False):
    return [{'params': param, 'lr': base_lr} for param in model_parameters]


def get_gd_optimizer(parameters, opt: str, lr: float, momentum: float, delta: float = 0.0,
                     sam: bool = False, sam_rho: float = 0.0, swa: bool = False, swa_lr: float = 0.0,
                     cosine_annealing_lr: bool = False) -> Optimizer:
    
    if delta is None:
        delta = 0.0

    parameters = get_gd_params(parameters, lr, True)

    if sam:
        assert sam_rho is not None
        if opt == "sgd":
            return SAM(parameters, SGD, sam_rho, lr=lr, dampening=delta)
    if opt == "sgd":
        opti = SGD(parameters, lr=lr, dampening=delta)
        if swa:
            return swa_utils.SWALR(opti, swa_lr=swa_lr)
        else:
            return opti
    elif opt == "polyak":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=False, dampening=delta)
    elif opt == "nesterov":
        return SGD(parameters, lr=lr, momentum=momentum, nesterov=True)


@torch.no_grad()
def update_ema(ema_model, model, ema_decay):
    if ema_model is not None and ema_decay > 0:
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.sub_((1 - ema_decay) * (ema_param - model_param))


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]], step: int = None):
    """Save a bunch of tensors."""
    suffix = f"final_{step}" if step is not None else "final"
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_{suffix}")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses

def compute_loss_for_single_instance(network, loss_function, image, label):
    y_pred = network(image.unsqueeze(0))
    loss = loss_function(y_pred, label.unsqueeze(0))
    return loss

def compute_grad_norm(grads):
    grads = [param_grad.detach().flatten() for param_grad in grads if param_grad is not None]
    norm = torch.cat(grads).norm()
    return norm

def get_loss_and_acc(loss: str):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")

"""
def get_loss_and_acc(loss: str, individual: bool = False):
    ""Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. ""
    if loss == "mse":
        return SquaredLoss(individual), SquaredAccuracy(individual)
    elif loss == "ce":
        reduction = None if individual else "sum"
        return nn.CrossEntropyLoss(reduction=reduction), AccuracyCE(individual)
    raise NotImplementedError(f"no such loss function: {loss}")
"""


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

def compute_empirical_sharpness(network: nn.Module, loss_fn: nn.Module, X, y):

    network.zero_grad()

    dim = len(parameters_to_vector((network.parameters())))
    neigs = 1

    def local_compute_hvp(vector):
        vector = vector.cuda()
        loss = loss_fn(network(X), y) / len(X)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        return parameters_to_vector(grads)

    matrix_vector = lambda delta: local_compute_hvp(delta).detach().cpu()

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, _ = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).item()


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


def obtained_eos(evalues, lr, window=5, relative_error=0.1):
    if len(evalues) < window:
        return False
    convergence = 2/lr
    for i in range(1, window+1):
        if convergence - evalues[-i] > relative_error * convergence:
            return False
    return True


def weights_init(module, mask):
    torch.nn.init.xavier_uniform(module.weight[mask].data)


def split_batch(X, y, metrics, gamma: float = 1.0):
    mean_metrics = np.mean(metrics)
    std_metrics = np.std(metrics)
    inliners = abs(metrics - mean_metrics) < gamma * std_metrics
    outliners = np.invert(inliners)
    return X[inliners], y[inliners], X[outliners], y[outliners]


def str_to_layers(network, strs):

    layers = []

    if type(strs) == "str":
        strs = [strs]
    elif strs == []:
        strs = ["fc1", "fc2", "fc3", "conv1", "conv2"]
    
    for str in strs:
        if str == "fc1":
            layers.append(network.fc1)
        if str == "fc2":
            layers.append(network.fc2)
        if str == "fc3":
            layers.append(network.fc3)
        if str == "conv1":
            layers.append(network.conv1)
        if str == "conv2":
            layers.append(network.conv2)
    
    return layers

def num_parameters(network):
    return len(parameters_to_vector(network.parameters()))

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
    def __init__(self): # , individual: bool = False):
        super(SquaredLoss, self).__init__() #individual)

    def forward(self, input: Tensor, target: Tensor):
        f = 0.5 * ((input - target) ** 2)
        if self.individual:
            return f.sum(dim=1)
        return f.sum()


class SquaredAccuracy(nn.Module):
    def __init__(self): # , individual: bool = False):
        super(SquaredAccuracy, self).__init__()
        #self.individual = individual

    def forward(self, input, target):
        f = (input.argmax(1) == target.argmax(1)).float()
        if self.individual:
            return f
        return f.sum()


class AccuracyCE(nn.Module):
    def __init__(self):#, individual: bool = False):
        super(AccuracyCE, self).__init__()
        #self.individual = individual

    def forward(self, input, target):
        f = (input.argmax(1) == target).float()
        if self.individual:
            return f
        return f.sum()


class VoidLoss(nn.Module):
    def __init__(self, individual: bool = False):
        super(VoidLoss, self).__init__()

    def forward(self, X, Y):
        return 0

