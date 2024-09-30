import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset

from new_data_utils import iterate_dataset

def flatten_gradients(network):
    """Flattens gradients of a PyTorch model into a single vector."""
    grad_list = []
    for param in network.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.view(-1))
    return torch.cat(grad_list)

def project_gradient(gradient, vectors):
    gradient_np = gradient.cpu().numpy()
    eigenvectors_np = vectors.numpy()
    projected_gradient_np = eigenvectors_np.dot(eigenvectors_np.T.dot(gradient_np))
    orthogonal_component_np = gradient_np - projected_gradient_np
    projected_gradient = torch.from_numpy(projected_gradient_np)
    orthogonal_component = torch.from_numpy(orthogonal_component_np)
    projected_gradient_norm_squared = torch.norm(projected_gradient, p=2) ** 2
    orthogonal_component_norm_squared = torch.norm(orthogonal_component, p=2) ** 2
    return projected_gradient, orthogonal_component, projected_gradient_norm_squared / (projected_gradient_norm_squared + orthogonal_component_norm_squared)

def obtained_eos(evalues, lr, window=5, relative_error=0.1):
    convergence = 2/lr
    if len(evalues) < window:
        if all(evalues > convergence):
            return True
        return False
    for i in range(1, window+1):
        if convergence - evalues[-i] > relative_error * convergence:
            return False
    return True

def compute_sharpness(network, loss_fn, method, dataset = None, batch = None, batch_size: int = None, repeats: int = None,
                      neigs = None, num_iterations = None, only_top: bool = True, return_evecs: bool = False):
    assert method in ["diag_fim", "fim", "lanczos", "power_iteration"]
    assert (dataset is None) != (batch is None) # xor
    if dataset is not None:
        assert batch_size is not None
        batchy = False
    else:
        batchy = True

    # smart: 
    # dict: {name: method}
    # if batchy
    # sharp = method(args)
    # sharp = method(args)

    if method == "diag_fim":
        if batchy:
            _, fisher_trace = fim_diag_batch(network, loss_fn, batch)
        else:
            _, fisher_trace = fim_diag_dataset(network, loss_fn, dataset, batch_size, repeats)
        sharpness = fisher_trace
    elif method == "fim":
        if batchy:
            _, fisher_trace = fim_batch(network, loss_fn, batch)
        else:
            _, fisher_trace = fim_dataset(network, loss_fn, dataset, batch_size, repeats)
        sharpness = fisher_trace
    elif method == "lanczos":
        assert neigs is not None
        out = get_hessian_eigenvalues(network, loss_fn, dataset, batch, neigs, batch_size, repeats, return_evecs)
        if return_evecs:
            evals, evecs = out
        else:
            evals, evecs = out, None
        sharpness = evals[0] if only_top else evals
        return sharpness, evecs
    elif method == "power_iteration":
        assert num_iterations is not None
        top_eig, _ = power_iteration_efficient(network, loss_fn, dataset, batch_size, num_iterations, repeats)
        sharpness = top_eig

    return sharpness

def fim_diag_dataset(network: nn.Module, loss_fn: nn.Module, dataset, batch_size: int, repeats:int = None):
    device = next(network.parameters()).device
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset) if repeats is None else min(repeats * batch_size, len(dataset))
    fisher_diag_info = torch.zeros(p)
    network.zero_grad()
    batch_id = 0
    for (X, y) in iterate_dataset(dataset, batch_size, repeats):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(network(X), y) / n
        loss.backward()

        for i, param in enumerate(network.parameters()):
            fisher_diag_info[i] += torch.sum(param.grad ** 2)

        network.zero_grad()
        batch_id += 1
        if repeats is not None and batch_id >= repeats:
            break
    fisher_trace = torch.sum(fisher_diag_info)
    return fisher_diag_info, fisher_trace

def fim_diag_batch(network: nn.Module, loss_fn: nn.Module, batch):
    device = next(network.parameters()).device
    p = len(parameters_to_vector(network.parameters()))
    n = len(batch)
    fisher_diag_info = torch.zeros(p)
    network.zero_grad()
    X, y = batch
    X, y = X.to(device), y.to(device)
    loss = loss_fn(network(X), y) / n
    loss.backward()
    for i, param in enumerate(network.parameters()):
        fisher_diag_info[i] += torch.sum(param.grad ** 2)
    fisher_trace = torch.sum(fisher_diag_info)
    return fisher_diag_info, fisher_trace

def fim_dataset(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, batch_size: int, repeats = None):
    device = next(network.parameters()).device
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset) if repeats is None else min(repeats * batch_size, len(dataset))
    counter = 0
    for (X, y) in iterate_dataset(dataset, batch_size, repeats):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(network(X), y) / n

        grads = torch.autograd.grad(loss, network.parameters(), create_graph=True)[0]
        grads = grads.view(-1)
        if counter:
            fim += torch.outer(grads, grads)
        else:
            fim = torch.outer(grads, grads)
        counter += 1
    fim /= counter

    # if ever want to compute eigs of fim:
    # sparse_fisher_info = spla.csr_matrix(fisher_info.numpy())
    # eigenvalues, _ = spla.eigsh(sparse_fisher_info, k=num_eigenvalues, which='LM')

    fisher_trace = torch.trace(fim)
    return fim, fisher_trace

def fim_batch(network: nn.Module, loss_fn: nn.Module, batch: torch.Tensor):
    device = next(network.parameters()).device
    p = len(parameters_to_vector(network.parameters()))
    n = len(batch)
    X, y = batch
    X, y = X.to(device), y.to(device)
    loss = loss_fn(network(X), y) / n
    grads = torch.autograd.grad(loss, network.parameters(), create_graph=True)[0]
    grads = grads.view(-1)
    fim = torch.outer(grads, grads)
    fisher_trace = torch.trace(fim)
    return fim, fisher_trace

def compute_hvp_dataset(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: torch.Tensor, batch_size: int, repeats: int = None):
    """Compute a Hessian-vector product."""
    device = next(network.parameters()).device # or just cuda?
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset) if repeats is None else min(len(dataset), repeats * batch_size)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for (X, y) in iterate_dataset(dataset, batch_size, repeats):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def compute_hvp_batch(network: nn.Module, loss_fn: nn.Module,
                batch, vector: torch.Tensor):
    """Compute a Hessian-vector product."""
    device = next(network.parameters()).device # or just cuda?
    p = len(parameters_to_vector(network.parameters()))
    n = len(batch)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    X, y = batch
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


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset, batch: torch.Tensor,
                            neigs:int = 6, batch_size:int = 1000, repeats: int = None, return_evecs: bool = False):
    """ Compute the leading Hessian eigenvalues. """
    assert (batch is None) != (dataset is None) # xor
    if dataset is not None:
        hvp_delta = lambda delta: compute_hvp_dataset(network, loss_fn, dataset,
                                            delta, batch_size=batch_size, repeats=repeats).detach().cpu()
    else:
        hvp_delta = lambda delta: compute_hvp_batch(network, loss_fn, batch,
                                            delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    if return_evecs:
        return evals, evecs
    return evals

# combine these two methods or like fim two methods
def compute_empirical_sharpness(network: nn.Module, loss_fn: nn.Module, X, y):
    device = next(network.parameters()).device # or just cuda?
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
                     dataset: Dataset, batch_size: int, repeats: int = None):
    """ Compute the gradient of the loss function at the current network parameters. """
    device = next(network.parameters()).device
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device=device)
    for (X, y) in iterate_dataset(dataset, batch_size, repeats):
        X, y = X.to(device), y.to(device)
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

    def __init__(self, network: nn.Module, new_params: torch.Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())

def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, batch_size=batch_size)

def power_iteration_efficient(network, loss_fn, dataset, batch_size, num_iterations=10, repeats=10):
    device = next(network.parameters()).device
    n = len(dataset) if repeats is None else min(len(dataset), batch_size * repeats)
    vec = torch.randn(sum(p.numel() for p in network.parameters()), device=device)
    vec = vec / torch.norm(vec)
    batch_id = 0
    for _ in range(num_iterations):
        Hv = torch.zeros_like(vec)
        for (X, y) in iterate_dataset(dataset, batch_size, repeats):
            X, y = X.to(device), y.to(device)
            loss = loss_fn(network(X), y) / n
            grads = torch.autograd.grad(loss, network.parameters(), create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])

            Hv_contrib = compute_hessian_vector_product(grad_vec, network.parameters(), vec)
            Hv += Hv_contrib

            batch_id += 1
            if batch_id >= repeats:
              break
        Hv /= repeats
        lambda_ = torch.dot(Hv, vec)
        vec = Hv / torch.norm(Hv)
    return lambda_, vec

def power_iteration_efficient_dataloader(model, dataloader, criterion, num_iterations=10, repeats=10):
    device = next(model.parameters()).device
    vec = torch.randn(sum(p.numel() for p in model.parameters()), device=device)
    vec = vec / torch.norm(vec)

    for _ in range(num_iterations):
        Hv = torch.zeros_like(vec)
        for batch_id, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])

            Hv_contrib = compute_hessian_vector_product(grad_vec, model.parameters(), vec)
            Hv += Hv_contrib

            if batch_id > repeats:
              break
        Hv /= repeats
        lambda_ = torch.dot(Hv, vec)
        vec = Hv / torch.norm(Hv)
    return lambda_, vec

def compute_hessian_vector_product(grad_vec, parameters, vec):
    jacobian_vec_product = torch.autograd.grad(grad_vec, parameters, grad_outputs=vec, retain_graph=True)
    return torch.cat([jvp.contiguous().view(-1) for jvp in jacobian_vec_product])
