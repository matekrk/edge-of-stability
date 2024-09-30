import copy
import os
import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
import wandb
import torch
from torch.nn import Conv2d, Linear

wandb.init(mode="disabled")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch_pruning

from main import train_gd
from mnist import load_mnist
from data_fashion import load_fashion
from lenet import LeNet
from utilities import iterate_dataset, get_loss_and_acc

loss = "mse"
loss_fn, acc_fn = get_loss_and_acc(loss)
loss_fn.__setattr__("individual", False)
acc_fn.__setattr__("individual", False)
batch_size = 200

main_path = "/home/mateusz.pyla/stan/atelier/sharpness"

if not "RESULTS" in os.environ:
    os.environ["RESULTS"] = os.path.join(main_path, "results")
if not "DATASETS" in os.environ:
    os.environ["DATASETS"] = os.path.join(main_path, "data")

train, test = load_mnist(loss, os.environ["DATASETS"])
train_fashion, test_fashion = load_fashion(loss, os.environ["DATASETS"])

lst_seeds = [10, 11, 12, 13, 14]

def get_model_path(dataset, seed, step):
    return f"/home/mateusz.pyla/stan/atelier/sharpness/results/{dataset}/lenet/mse/sgd/lr_0.01_delta0.0/seed_{seed}/freq_1/start_0/model_snapshot_step_{step}"

def count_zero_parameters(model):
    zero_count = 0
    for param in model.parameters():
        if param is not None:
            zero_count += torch.sum(param == 0).item()
    return zero_count

def finetune(model, loss_fn, train_dataset, batch_size, lr, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for e in range(epochs):
        running_loss = []
        for (X, y) in iterate_dataset(train_dataset, batch_size):
            opt.zero_grad()
            X.to(device)
            out = model(X)
            loss = loss_fn(out, y) / len(X)
            running_loss.append(loss.item())
            loss.backward()
            opt.step()

def evaluate(model, test_dataset, batch_size=1000):
    with torch.no_grad():
        running_loss, running_acc = [], []
        for (X, y) in iterate_dataset(test_dataset, batch_size):
            X, y = X.to(device), y.to(device)

            loss = loss_fn(model(X), y) / len(X)
            acc = acc_fn(model(X), y) / len(X)

            running_loss.append(loss)
            running_acc.append(acc)

        return sum(running_loss)/len(running_loss), sum(running_acc)/len(running_acc)
    

lst_steps = [0, 5, 10, 20, 30, 50, 75, 99]
lst_percents = [0, 5, 10, 20, 30, 40, 50, 75, 90]
dataset = "fashion"
lst_types = [[LeNet]] # [[LeNet],[Conv2d, Linear], [Conv2d], [Linear]]

def is_one_of(value, types):
    """Checks if a value is one of the given types."""
    return any(isinstance(value, t) for t in types)

lst_pruning_methods = [torch_pruning.pruner.MagnitudePruner]
iterative_steps = 5
ex = test[0][0].unsqueeze(0).to(device)

def main():
    d = {}
    for seed in lst_seeds:
        for step in lst_steps:
            for percent in lst_percents:
                for i, types in enumerate(lst_types):
                    for j, pruning_method in enumerate(lst_pruning_methods):
                        model = LeNet()
                        model.load_state_dict(torch.load(get_model_path(dataset, seed, step)))
                        model.to(device)

                        base_macs, base_nparams = torch_pruning.utils.count_ops_and_params(model, ex)

                        ignored_layers = []
                        for m in model.named_parameters():
                            if not is_one_of(m, types):
                                ignored_layers.append(m)
                    
                        imp = torch_pruning.importance.MagnitudeImportance(p=2, group_reduction='mean')
                        this_pruning_method = pruning_method(model, ex, importance=imp, iterative_steps=iterative_steps, 
                                                            pruning_ratio=percent*0.01, global_pruning=False,
                                                            ignored_layers=ignored_layers)

                        for k in range(iterative_steps):
                            for group in this_pruning_method.step(interactive=True):
                                for dep, idxs in group:
                                    target_layer = dep.target.module
                                    pruning_fn = dep.handler
                                    if pruning_fn in [torch_pruning.prune_conv_in_channels, torch_pruning.prune_linear_in_channels]:
                                        target_layer.weight.data[:, idxs] *= 0
                                    elif pruning_fn in [torch_pruning.prune_conv_out_channels, torch_pruning.prune_linear_out_channels]:
                                        target_layer.weight.data[idxs] *= 0
                                        if target_layer.bias is not None:
                                            target_layer.bias.data[idxs] *= 0
                                    elif pruning_fn in [torch_pruning.prune_batchnorm_out_channels]:
                                        target_layer.weight.data[idxs] *= 0
                                        target_layer.bias.data[idxs] *= 0
                                    # group.prune() # <= disable hard pruning

                            pruned_macs, pruned_nparams = torch_pruning.utils.count_ops_and_params(model, ex)

                            nparams = count_zero_parameters(model)

                            sparsity = 1.0 - nparams / base_nparams

                            loss, acc = evaluate(model, test_fashion, batch_size)

                            model_ft = copy.deepcopy(model)

                            # finetune
                            finetune(model_ft, loss_fn, train, batch_size, 0.01, 3)
                            loss_ft, acc_ft = evaluate(model_ft, test, batch_size)

                            exp_name = f"s{seed}_c{step}_p{percent}_t{i}_m{j}_i{k}"
                            d[exp_name] = (acc, loss, sparsity, loss_ft, acc_ft)
                            print(exp_name)
            
    with open('saved_dictionary_soft_transferFtoM.pkl', 'wb') as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    main()
