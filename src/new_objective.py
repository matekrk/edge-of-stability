from typing import List
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, Tensor
import torch

from new_sharpness import compute_sharpness
from new_data_utils import Dataset, DataLoader, iterate_dataset

class VoidMetric(nn.Module):
    def __init__(self, individual: bool = False):
        super(VoidMetric, self).__init__()
        self.individual = individual

    def forward(self, input: Tensor, target: Tensor):
        return 0

class SquaredLoss(VoidMetric):
    def __init__(self, individual: bool = False):
        super(SquaredLoss, self).__init__(individual)

    def forward(self, input: Tensor, target: Tensor):
        f = 0.5 * ((input - target) ** 2)
        if self.individual:
            return f.sum(dim=1)
        return f.sum()

class SquaredAccuracy(VoidMetric):
    def __init__(self, individual: bool = False):
        super(SquaredAccuracy, self).__init__(individual)

    def forward(self, input, target):
        f = (input.argmax(1) == target.argmax(1)).float()
        if self.individual:
            return f
        return f.sum()

class AccuracyCE(VoidMetric):
    def __init__(self, individual: bool = False):
        super(AccuracyCE, self).__init__(individual)

    def forward(self, input, target):
        f = (input.argmax(1) == target).float()
        if self.individual:
            return f
        return f.sum()

def get_loss_and_acc(loss: str, individual: bool = False):
    """Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction. """
    if loss == "mse":
        return SquaredLoss(individual), SquaredAccuracy(individual)
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction = "none" if individual else "sum"), AccuracyCE(individual)
    raise NotImplementedError(f"no such loss function: {loss}")

def omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2, eps=1e-3, return_absolute=False):

    all_weights = []
    n_params = 0
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue
        all_weights.append(param)
        n_params += param.numel()

    total_L0_norm = 0
    for weights in all_weights:
        total_L0_norm += torch.count_nonzero((torch.abs(weights.data) < eps))

    total_L0_norm = total_L0_norm / n_params

    total_L1_norm = 0
    for weights in all_weights:
        total_L1_norm += torch.norm(weights, 1).item()

    total_L2_norm = 0
    for weights in all_weights:
        total_L2_norm += torch.norm(weights).item()

    total_weighted_norm = omega_wd_0 * total_L0_norm + omega_wd_1 * total_L1_norm + omega_wd_2 * total_L2_norm

    if return_absolute:
        return total_weighted_norm, total_L0_norm, total_L1_norm, total_L2_norm
    return total_weighted_norm

def sharpness_penalty(network, loss_fn, X, y, sharpness_dict, tau_0, tau_1, return_absolute=False):
    fisher_penalty, hessian_penalty = 0.0, 0.0
    if tau_0:
        fisher_penalty = compute_sharpness(network, loss_fn, "diag_fim", batch=(X,y))
    if tau_1:
        hessian_penalty = compute_sharpness(network, loss_fn, "lanczos", batch=(X,y))
    weighted_penalty = tau_0 * fisher_penalty + tau_1 * hessian_penalty
    if return_absolute:
        return weighted_penalty, fisher_penalty, hessian_penalty
    else:
        return weighted_penalty
    
def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset, batch_size: int, no_grad: bool = False):
    """Compute loss over a dataset."""
    device = next(network.parameters()).device
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    if no_grad:
        with torch.no_grad():
            for (X, y) in iterate_dataset(dataset, batch_size):
                X, y = X.to(device), y.to(device)
                preds = network(X)
                for l, loss_fn in enumerate(loss_functions):
                    losses[l] += loss_fn(preds, y) / len(dataset)
    else:
            for (X, y) in iterate_dataset(dataset, batch_size):
                X, y = X.to(device), y.to(device)
                preds = network(X)
                for l, loss_fn in enumerate(loss_functions):
                    losses[l] += loss_fn(preds, y) / len(dataset)
    return losses

def compute_losses_dataloader(network: nn.Module, loss_functions: List[nn.Module], dataloader: DataLoader, no_grad: bool = False):
    device = next(network.parameters()).device
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    if no_grad:
        with torch.no_grad():
            for (X, y) in dataloader:
                X, y = X.to(device), y.to(device)
                preds = network(X)
                for l, loss_fn in enumerate(loss_functions):
                    losses[l] += loss_fn(preds, y) / len(dataloader.dataset)
    else:
            for (X, y) in dataloader:
                X, y = X.to(device), y.to(device)
                preds = network(X)
                for l, loss_fn in enumerate(loss_functions):
                    losses[l] += loss_fn(preds, y) / len(dataloader.dataset)
    return losses

def compute_loss_for_single_instance(network, loss_function, image, label, no_grad: bool = False):
    device = next(network.parameters()).device
    image, label = image.to(device), label.to(device)
    if no_grad:
        with torch.no_grad():
            loss = loss_function(network(image.unsqueeze(0)), label.unsqueeze(0))
    else:
            loss = loss_function(network(image.unsqueeze(0)), label.unsqueeze(0))
    return loss

def compute_grad_norm(grads):
    grads = [param_grad.detach().flatten() for param_grad in grads if param_grad is not None]
    norm = torch.cat(grads).norm()
    return norm

def compute_metrics_per_class(network, dataset, batch_size, no_grad: bool = True):
    device = next(network.parameters()).device
    all_preds = []
    all_labels = []
    if no_grad:
        with torch.no_grad():
            for (X, y) in iterate_dataset(dataset, batch_size):
                X, y = X.to(device), y.to(device)
                preds = network(X)
                _, predicted_labels = torch.max(preds, 1)
                all_preds.extend(predicted_labels.tolist())
                if len(y.shape) == 2:
                    _, labels = torch.max(y, 1)
                else:
                    labels = y
                all_labels.extend(labels.tolist())
    else:
            for (X, y) in iterate_dataset(dataset, batch_size):
                X, y = X.to(device), y.to(device)
                preds = network(X)
                _, predicted_labels = torch.max(preds, 1)
                all_preds.extend(predicted_labels.tolist())
                if len(y.shape) == 2:
                    _, labels = torch.max(y, 1)
                else:
                    labels = y
                all_labels.extend(labels.tolist())
    precisions, recalls, f1s, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    return precisions, recalls, f1s
