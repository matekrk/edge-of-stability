from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from torch import nn
import torch.optim as optim

import torch_pruning as tp
from pruning_train import *
from pruning_prune import MagnitudeImportanceReversed
from pruning_viz import *

def load_importance(pruning_importance_str, pruning_norm=2):
    if pruning_importance_str == "magnitude_min":
        method = tp.importance.MagnitudeImportance(p=pruning_norm)
    elif pruning_importance_str == "magnitude_max":
        method = MagnitudeImportanceReversed(p=pruning_norm)

    return method

def count_zero_parameters(model, eps=0.0001, relative = False):
    zero_count = 0
    for param in model.parameters():
        if param is not None:
            zero_count += torch.sum(abs(param) < eps).item()
    if relative:
        zero_count /= sum(p.numel() for p in model.parameters())
    return zero_count

def compute_frobenius_norm(model):
    norm_squared = 0.0
    for param in model.parameters():
        norm_squared += torch.norm(param, p=2, dim=[0, 1]).pow(2)
    return torch.sqrt(norm_squared)

def prepare_data_finetune(loss, finetune_dataset_name, finetune_batch_size):
    data_train, mean, std = load_dataset(finetune_dataset_name, loss, stats=None)
    data_train_loader = get_dataloader(data_train, finetune_batch_size)
    return data_train, data_train_loader

def finetune(model, loss_fn, finetune_dataset, batch_size, lr, epochs, device):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for e in range(epochs):
        running_loss = []
        for (X, y) in iterate_dataset(finetune_dataset, batch_size):
            opt.zero_grad()
            X.to(device)
            out = model(X)
            loss = loss_fn(out, y) / len(X)
            running_loss.append(loss.item())
            loss.backward()
            opt.step()

def compute_faithfullness(network, subnetwork, loss, dataset, batch_size, no_grad = False, per_class = False):
    """Compute loss over a dataset.""",
    device = next(network.parameters()).device
    subnetwork.to(device)
    if per_class:
        n_classes = 10 # hardcoded:
        faithfullness = torch.zeros(n_classes)
        total = torch.zeros(n_classes)
    else:
        faithfullness = torch.zeros(1)
        total = 0
    if no_grad:
        with torch.no_grad():
            for (X, y) in iterate_dataset(dataset, batch_size):
                X, y = X.to(device), y.to(device)
                if loss == "mse":
                    y = torch.argmax(y, dim=-1)
                preds = network(X)
                preds_subnet = subnetwork(X)
                if per_class:
                    for cl in range(n_classes):
                        mask = y == cl
                        faithfullness[cl] += (torch.argmax(preds[mask], dim=-1) == torch.argmax(preds_subnet[mask], dim=-1)).sum().item()
                        total[cl] += mask.sum().item()
                else:
                    faithfullness += (torch.argmax(preds, dim=-1) == torch.argmax(preds_subnet, dim=-1)).sum().item()
                    total += len(X)
    else:
        for (X, y) in iterate_dataset(dataset, batch_size):
            X, y = X.to(device), y.to(device)
            if loss == "mse":
                y = torch.argmax(y, dim=-1)
            preds = network(X)
            preds_subnet = subnetwork(X)
            if per_class:
                for cl in range(n_classes):
                    mask = y == cl
                    faithfullness[cl] += (torch.argmax(preds[mask], dim=-1) == torch.argmax(preds_subnet[mask], dim=-1)).sum().item()
                    total[cl] += mask.sum().item()
            else:
                faithfullness += (torch.argmax(preds, dim=-1) == torch.argmax(preds_subnet, dim=-1)).sum().item()
                total += len(X)

    return faithfullness / total

def soft_pruning(network, strength, imp, ignored_layers, iterative_steps, example_inputs, not_prune_bn, pruning_ratio_dict, verbose = False):
    """ just zero the weight do not remove """

    pruner = tp.pruner.MagnitudePruner(
        network,
        example_inputs,
        importance=imp,
        global_pruning=True,
        iterative_steps=iterative_steps,
        pruning_ratio=strength, # remove strength*100% channels, f.e. ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        pruning_ratio_dict=pruning_ratio_dict,
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(network, example_inputs)
    base_zero_nparams = count_zero_parameters(network)
    for i in range(iterative_steps):
        # Soft Pruning
        for group in pruner.step(interactive=True):
            for dep, idxs in group:
                target_layer = dep.target.module
                pruning_fn = dep.handler
                if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                    target_layer.weight.data[:, idxs] *= 0
                elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                    target_layer.weight.data[idxs] *= 0
                    if target_layer.bias is not None:
                        target_layer.bias.data[idxs] *= 0
                elif not not_prune_bn and pruning_fn in [tp.prune_batchnorm_out_channels]:
                    target_layer.weight.data[idxs] *= 0
                    target_layer.bias.data[idxs] *= 0
                # group.prune() # <= disable hard pruning
                if verbose:
                    print(target_layer, f"{((count_zero_parameters(target_layer))/(sum(p.numel() for p in target_layer.parameters())+0.0001)):.4f}")
        if verbose:
            print(network.classifier[-2].weight)

        macs, nparams = tp.utils.count_ops_and_params(network, example_inputs)
        zero_nparams = count_zero_parameters(network)
        if verbose:
            print(network)
            print(network(example_inputs).shape)
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, 0 Params: %.2f M => %.2f M"
                % (i+1, iterative_steps, base_zero_nparams / 1e6, zero_nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )

        # finetune your model here
        # finetune(model)
        # ...

    return network, zero_nparams

def main_single(network_name, network_path, train_dataset_name, test_datasets_names, finetune_dataset_name, loss_name, train_batch_size, test_batch_size, finetune_batch_size, device, 
         pruning_p, pruning_importance_str, pruning_iterative_steps, pruning_ignore_bn, save_model, verbose):
    
    DIR = "/home/mateusz.pyla/stan/atelier/sharpness/pruning/"
    result_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.abspath(os.getcwd())
    seed = 8

    if not "RESULTS" in os.environ:
        os.environ["RESULTS"] = os.path.join(result_dir, "results")
    if not "DATASETS" in os.environ:
        os.environ["DATASETS"] = os.path.join(dataset_dir, "data")


    loss_fn, acc_fn = get_loss_and_acc(loss_name)
    train_dataset, train_dataloader, test_datasets, test_dataloaders = prepare_data(train_dataset_name, test_datasets_names, loss_name, train_batch_size, test_batch_size)
    finetune_dataset, finetune_dataloader = prepare_data_finetune(loss_name, finetune_dataset_name, finetune_batch_size)
    example_inputs = torch.randn_like(train_dataset[0][0].unsqueeze(dim=0)).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if network_name == "resnet-9":
        network = ResNet9(input_shape=[32, 32, 3], output_shape=10, softmax=True).to(device)
        network.train()
        ignored = [network.classifier[-2]]
    
    elif network_name == "lenet":
        network = LeNet(input_shape=[28, 28, 1], output_shape=10, softmax=True, pooling="max").to(device)
        network.train()
        ignored = [network.classifier[-2]]

    if pruning_ignore_bn:
        # for name, module in network.named_modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         ignored.append(module)
        pruning_ratio_dict = {network.feature_extractor[0][1]: 0.0,
                              network.feature_extractor[1][1]: 0.0,
                              network.feature_extractor[2].layers[0][1]: 0.0,
                              network.feature_extractor[2].layers[0][1]: 0.0,
                              network.feature_extractor[3][1]: 0.0,
                              network.feature_extractor[4][1]: 0.0,
                              network.feature_extractor[5].layers[0][1]: 0.0,
                              network.feature_extractor[5].layers[0][1]: 0.0}
    else:
        pruning_ratio_dict = None

    network.load_state_dict(torch.load(os.path.join(DIR, network_path)))
    network.eval()
    before_network = copy.deepcopy(network)
    before_zero_nparams = count_zero_parameters(network)

    before_train_rez = compute_losses(network, [loss_fn, acc_fn], train_dataset, test_batch_size, no_grad=True)
    before_test_rez = []
    for k, test_dataset in enumerate(test_datasets):
        before_test_rez.append(compute_losses(network, [loss_fn, acc_fn], test_dataset, test_batch_size, no_grad=True))
    print_compute_losses(before_train_rez, before_test_rez)

    imp = load_importance(pruning_importance_str)
    pruned_network, zero_nparams = soft_pruning(network, pruning_p, imp, ignored, pruning_iterative_steps, example_inputs, pruning_ignore_bn, pruning_ratio_dict, verbose)
    if save_model:
        torch.save(pruned_network.state_dict(), os.path.join(DIR, f"pruned_{pruning_importance_str}_{pruning_p}_nobn{'T' if pruning_ignore_bn else 'F'}_ON_" + network_path))

    pruned_network.eval()
    train_rez = compute_losses(pruned_network, [loss_fn, acc_fn], train_dataset, test_batch_size, no_grad=True)
    test_rez = []
    for k, test_dataset in enumerate(test_datasets):
        test_rez.append(compute_losses(pruned_network, [loss_fn, acc_fn], test_dataset, test_batch_size, no_grad=True))
    #print("BEFORE", before_train_rez, before_test_rez, before_zero_nparams)
    #print("AFTER", train_rez, test_rez, zero_nparams)
    print_compute_losses(before_train_rez, before_test_rez)
    print_compute_losses(train_rez, test_rez)

    faithfullness = compute_faithfullness(before_network, pruned_network, loss_name, test_dataset, test_batch_size, no_grad=True, per_class=True)
    print(faithfullness)

def main_many(network_name, network_path, train_dataset_name, test_datasets_names, finetune_dataset_name, loss_name, train_batch_size, test_batch_size, finetune_batch_size, device, 
         pruning_ps, pruning_importance_str, pruning_iterative_steps, pruning_ignore_bn, save_model, verbose):
    
    DIR = "/home/mateusz.pyla/stan/atelier/sharpness/pruning/"
    result_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.abspath(os.getcwd())
    seed = 8

    if not "RESULTS" in os.environ:
        os.environ["RESULTS"] = os.path.join(result_dir, "results")
    if not "DATASETS" in os.environ:
        os.environ["DATASETS"] = os.path.join(dataset_dir, "data")


    loss_fn, acc_fn = get_loss_and_acc(loss_name)
    train_dataset, train_dataloader, test_datasets, test_dataloaders = prepare_data(train_dataset_name, test_datasets_names, loss_name, train_batch_size, test_batch_size)
    finetune_dataset, finetune_dataloader = prepare_data_finetune(loss_name, finetune_dataset_name, finetune_batch_size)
    example_inputs = torch.randn_like(train_dataset[0][0].unsqueeze(dim=0)).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if network_name == "resnet-9":
        network = ResNet9(input_shape=[32, 32, 3], output_shape=10, softmax=True).to(device)
        network.train()
        ignored = [network.classifier[-2]]
    
    elif network_name == "lenet":
        network = LeNet(input_shape=[28, 28, 1], output_shape=10, softmax=True, pooling="max").to(device)
        network.train()
        ignored = [network.classifier[-2]]

    if pruning_ignore_bn:
        for name, module in network.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                ignored.append(module)

    network.load_state_dict(torch.load(os.path.join(DIR, network_path)))

    before_train_rez = compute_losses(network, [loss_fn, acc_fn], train_dataset, test_batch_size, no_grad=True)
    before_test_rez = []
    for k, test_dataset in enumerate(test_datasets):
        before_test_rez.append(compute_losses(network, [loss_fn, acc_fn], test_dataset, test_batch_size, no_grad=True))
    #print("BEFORE", before_train_rez, before_test_rez)
    print_compute_losses(before_train_rez, before_test_rez)
    n_params = sum(p.numel() for p in network.parameters())

    pruning_rez = {}
    pruning_rez[0.0] = before_train_rez, before_test_rez, count_zero_parameters(network) / n_params
    faithfullness_rez = {}
    faithfullness_rez[0.0] = 1.0, [1.0] * len(test_datasets_names)

    for pruning_p in pruning_ps:

        main_network = copy.deepcopy(network)

        imp = load_importance(pruning_importance_str)

        pruned_network, base_zero_nparams = soft_pruning(main_network, pruning_p, imp, ignored, pruning_iterative_steps, example_inputs, pruning_ignore_bn, verbose)
        if save_model:
            torch.save(pruned_network.state_dict(), os.path.join(DIR, f"pruned_{pruning_importance_str}_{pruning_p}_nobn{'T' if pruning_ignore_bn else 'F'}_ON_" + network_path))

        train_rez = compute_losses(pruned_network, [loss_fn, acc_fn], train_dataset, test_batch_size, no_grad=True)
        faithfullness_train = compute_faithfullness(network, pruned_network, loss_name, train_dataset, test_batch_size, no_grad=True, per_class=False)
        test_rez = []
        faithfullness_test = []
        for k, test_dataset in enumerate(test_datasets):
            test_rez.append(compute_losses(pruned_network, [loss_fn, acc_fn], test_dataset, test_batch_size, no_grad=True))
            faithfullness_test.append(compute_faithfullness(network, pruned_network, loss_name, test_dataset, test_batch_size, no_grad=True, per_class=False))
        print_compute_losses(train_rez, test_rez)
        # print("AFTER", train_rez, test_rez)
        pruning_rez[pruning_p] = train_rez, test_rez, base_zero_nparams / n_params

        faithfullness_rez[pruning_p] = faithfullness_train, faithfullness_test

    fig = simple_plot_rez(pruning_rez, pruning_ps, train_dataset_name, test_datasets_names, network_path)
    fig.savefig(os.path.join(DIR, f"{network_path[:-3]}_rez_pruning_simple.png"))

    fig = plot_rez(pruning_rez, faithfullness_rez, pruning_ps, train_dataset_name, test_datasets_names, network_path, pruning_importance_str)
    fig.savefig(os.path.join(DIR, f"{network_path[:-3]}_{pruning_importance_str}_rez_pruning.png"))


if __name__ == "__main__":
    network_name = "resnet-9"
    network_path = "resnet-9_cifar10-10k_sgd_0.01_50_mse_s8.pt"
    train_dataset_name = "cifar10-10k"
    test_datasets_names = ["cifar10t", "cifar10c_brightness2", "cifar10c_gaussian_noise4"]
    finetune_dataset_name = "cifar10-10k"
    loss_name = "mse"
    train_batch_size = 64
    test_batch_size = 1000
    finetune_batch_size = 64 
    device = "cuda"
    pruning_p = 0.5
    pruning_ps = np.logspace(-10.0, 0.0, base=2, endpoint=False, num=50) # [0.001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
    pruning_importance_str = "magnitude_min" # magnitude min max
    pruning_iterative_steps = 1
    pruning_ignore_bn = True # currently counts the whole block
    pruning_smaller_bn = 0.0001

    single = True
    save_model = True
    verbose = True

    if single:
        main_single(network_name, network_path, train_dataset_name, test_datasets_names, finetune_dataset_name, loss_name, train_batch_size, test_batch_size, finetune_batch_size, device,
            pruning_p, pruning_importance_str, pruning_iterative_steps, pruning_ignore_bn, save_model, verbose)

    else:
        main_many(network_name, network_path, train_dataset_name, test_datasets_names, finetune_dataset_name, loss_name, train_batch_size, test_batch_size, finetune_batch_size, device,
            pruning_ps, pruning_importance_str, pruning_iterative_steps, pruning_ignore_bn, save_model, verbose)
