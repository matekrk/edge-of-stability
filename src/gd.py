from os import makedirs, path
from matplotlib import pyplot as plt
import wandb

import torch
from torch.nn.utils import parameters_to_vector
import torch.functional as F

from archs import load_architecture
from plot import plot_gd
from sam import SAM
from cifar import CIFAR_LABELS, predict_particular
from gradcam import do_gradcam
from selective_neurons import get_class_selectivity
from utilities import compute_empirical_sharpness, compute_grad_norm, compute_loss_for_single_instance, get_gd_optimizer, get_gd_directory, get_gd_params, get_loss_and_acc, compute_losses, num_parameters, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, obtained_eos, split_batch, str_to_layers, weights_init
from data import load_dataset, take_first, DATASETS

def train_epoch(network, train_dataset, physical_batch_size, loss_fn, acc_fn, 
                optimizer, optimizer_outliners_data, optimizer_outliners_features,
                swa, swa_model, swa_scheduler, ema, ema_model, ema_decay,
                grad_ind_data, grad_ind_data_strategy, grad_ind_data_gamma,
                grad_ind_grad, grad_ind_grad_strategy, grad_ind_grad_gamma,
                keep_random_layers, keep_random_neurons, last_layer = None):

    if last_layer is None:
        if network._get_name() == "LeNet":
            last_layer = network.fc3
        elif network._get_name() == "ResNet9":
            last_layer = network.classifier[3]
        elif network._get_name() == "ResNet":
            last_layer = network.fc

    network.train()

    gradients = 0
    optimizer.zero_grad()
    others = []

    for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
        X, y = X.cuda(), y.cuda()

        if grad_ind_data:
            # grad_ind_data_strategy == "norm_grad"
            optimizer.zero_grad()
            network.eval()
            vmap_loss = torch.vmap(compute_loss_for_single_instance, in_dims=(None, None, 0, 0))
            losses = vmap_loss(network, loss_fn, X, y)
            norm_gradients = [compute_grad_norm(torch.autograd.grad(loss, network.parameters(), retain_graph=True)).cpu().numpy() for loss in losses]
            network.train()
            gradients += 1
            X_inliners, y_inliners, X_outliners, y_outliners = split_batch(X, y, norm_gradients, grad_ind_data_gamma)

            if len(X_inliners) == 0:
                print("WOW")

            wandb.log({"train/running/data_outliners_ratio": len(X_outliners)/len(X),
                       "train/running/data_inliners_loss": loss_fn(network(X_inliners), y_inliners)/len(X_inliners),
                       "train/running/data_inliners_acc": acc_fn(network(X_inliners), y_inliners)/len(X_inliners),
                       "train/running/data_outliners_loss": loss_fn(network(X_outliners), y_outliners)/len(X_outliners),
                       "train/running/data_outliners_acc": acc_fn(network(X_outliners), y_outliners)/len(X_outliners),
                       "train/running/data_inliners_e1": compute_empirical_sharpness(network, loss_fn, X_inliners, y_inliners),
                       "train/running/data_outliners_e1": compute_empirical_sharpness(network, loss_fn, X_outliners, y_outliners)})

            """
            others.append()
            others.append()
            others.append()
            others.append()
            others.append()
            others.append()
            others.append() """
        else:
            X_inliners, y_inliners = X, y
            X_outliners, y_outliners = None, None

        mask_outweights = torch.zeros(last_layer.weight.shape[1], dtype=torch.bool)
        if grad_ind_grad:
            if grad_ind_grad_strategy == "grad":
                optimizer.zero_grad()
                loss = loss_fn(network(X_inliners), y_inliners) / len(X_inliners)
                loss.backward()
                gradients += 1
                grads = last_layer.weight.grad
                _, outweight = torch.topk(grads.abs(), k=int(grad_ind_grad_gamma), dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights = torch.zeros(grads.shape[1], dtype=torch.bool)
                mask_outweights[outweight] = 1
                #if grad_ind_grad_lr is not None:
                #    optimizer.param_groups[-2]['lr'] = grad_ind_grad_lr
                #TODO: mask

                """
                # def f(network, x):
            #     _, f = network(x, return_features=True)

                #vmap_loss = torch.vmap(compute_loss_for_single_instance, in_dims=(None, None, 0))
                #losses = vmap_loss(network, f, X)

                prev_grad = optimizer.param_groups[0]['params'][-2].grad.copy()
                optimizer.param_groups[0]['params'][-2].grad = 0.0
                inweight = torch.inverse(outweight)
                if grad_ind_grad_lr is not None:
                    optimizer.param_groups["fc3.weight"]["lr"] = grad_ind_grad_lr
                optimizer.add_param_group({'params': optimizer.param_groups[0]['params'][-2]})"""

            elif grad_ind_grad_strategy == "norm":
                optimizer.zero_grad()
                outputs, features = network(X_inliners, return_features=True)
                loss = loss_fn(outputs, y_inliners) / len(X_inliners)
                loss.backward()
                gradients += 1
                mean_feat_norm = features.norm(dim=0)
                grads = last_layer.weight.grad
                _, outweight = torch.topk(mean_feat_norm.abs(), k=int(grad_ind_grad_gamma), dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights[outweight] = 1

                torch.autograd.grad(loss, network.parameters(), retain_graph=True)
                gradients += 1

        # TODO: keep_random_neurons
        frozen_layers = []
        if keep_random_layers:
            for name, param in network.named_parameters():
                if torch.rand(1).item() < keep_random_layers:
                    param.requires_grad = False
                    frozen_layers.append(param)

        optimizer.zero_grad()
        loss = loss_fn(network(X_inliners), y_inliners) / len(X_inliners)
        loss.backward()
        last_layer.weight.grad[:, mask_outweights] = 0.0
        if isinstance(optimizer, SAM):
            def closure():
                loss = loss_fn(network(X_inliners), y_inliners) / len(X_inliners)
                loss.backward()
                last_layer.weight.grad[:, mask_outweights] = 0.0
                return loss
            optimizer.step(closure)
            gradients += 2
        else:
            optimizer.step()
            gradients += 1

        if optimizer_outliners_data:
            optimizer_outliners_data.zero_grad()
            loss = loss_fn(network(X_outliners), y_outliners) / len(X_outliners)
            loss.backward()
            last_layer.weight.grad[:, mask_outweights] = 0.0
            if isinstance(optimizer_outliners_data, SAM):
                def closure():
                    loss = loss_fn(network(X_inliners), y_inliners) / len(train_dataset)
                    loss.backward()
                    last_layer.weight.grad[:, mask_outweights] = 0.0
                    return loss
                optimizer_outliners_data.step(closure)
                gradients += 2
            else:
                optimizer_outliners_data.step()
                gradients += 1

        if optimizer_outliners_features: #FIXME
            optimizer_outliners_features.zero_grad()
            loss = loss_fn(network(X_outliners), y_outliners) / len(train_dataset)
            loss.backward()
            last_layer.weight.grad[:, ~mask_outweights] = 0.0
            if isinstance(optimizer_outliners_features, SAM):
                def closure():
                    loss = loss_fn(network(X_outliners), y_outliners) / len(train_dataset)
                    loss.backward()
                    last_layer.weight.grad[:, ~mask_outweights] = 0.0
                    return loss
                optimizer_outliners_features.step(closure)
                gradients += 2
            else:
                optimizer_outliners_features.step()
                gradients += 1

        for layer in frozen_layers:
            layer.requires_grad = True

        if swa:
          if swa_model is None:
              swa_model = torch.optim.swa_utils.AveragedModel(network)
          else:
              swa_model.update_parameters(network)
              if swa_scheduler is not None:
                swa_scheduler.step()

        if ema:
            if ema_model is None:
              ema_model = torch.optim.swa_utils.AveragedModel(network, 
                                                              torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay), 
                                                              use_buffers=True)
            else:
              ema_model.update_parameters(network)
    
    optimizer.zero_grad()
    return gradients # , others

def do_minirestart(network, train_dataset, physical_batch_size, loss_fn, optimizer,
                minirestart_tricks,
                minirestart_tricks_layers,
                minirestart_tricks_topk: int = 10,
                minirestart_tricks_reducenorm_param: float = 0.1,
                minirestart_tricks_addnoise_param: float = 0.01,
                minirestart_tricks_addneurons_param: float = 1.1):
    
    gradients = 0
    relevant_layers = str_to_layers(network, minirestart_tricks_layers)

    for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
        X, y = X.cuda(), y.cuda()

        if "reducenorm" in minirestart_tricks:
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X), y) / len(X)
                loss.backward()
                gradients += 1
            for module in relevant_layers:
                grads = module.weight.grad
                _, outweight = torch.topk(grads.abs(), k=minirestart_tricks_topk, dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights = torch.zeros(grads.shape[1], dtype=torch.bool)
                mask_outweights[outweight] = 1
                module.weight[:,~mask_outweights].data *= minirestart_tricks_reducenorm_param

        if "backtoinit" in minirestart_tricks:
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X), y) / len(X)
                loss.backward()
                gradients += 1
            for module in relevant_layers:
                grads = module.weight.grad
                _, outweight = torch.topk(grads.abs(), k=minirestart_tricks_topk, dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights = torch.zeros(grads.shape[1], dtype=torch.bool)
                mask_outweights[outweight] = 1
                to_save = module.weight[:,~mask_outweights]
                module.reset_parameters()
                module.weight[:,~mask_outweights].data = to_save

        if "addnoise" in minirestart_tricks:
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X), y) / len(X)
                loss.backward()
                gradients += 1
            for module in relevant_layers:
                grads = module.weight.grad
                _, outweight = torch.topk(grads.abs(), k=minirestart_tricks_topk, dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights = torch.zeros(grads.shape[1], dtype=torch.bool)
                mask_outweights[outweight] = 1
                module.weight[:,mask_outweights].data = (minirestart_tricks_addnoise_param**0.5)*torch.randn(*module.weight[:,mask_outweights].data.shape)
            optimizer.zero_grad()

        if "addneurons" in minirestart_tricks:
            for str_module in minirestart_tricks_layers:
                if str_module == "conv1":
                    network.resize_conv1_layer(int(minirestart_tricks_addneurons_param*network.filter1))
                if str_module == "conv2":
                    network.resize_conv2_layer(int(minirestart_tricks_addneurons_param*network.filter2))
                if str_module == "fc1":
                    network.resize_fc1_layer(int(minirestart_tricks_addneurons_param*network.neurons1))
                if str_module == "fc2":
                    network.resize_fc2_layer(int(minirestart_tricks_addneurons_param*network.neurons2))
                if str_module == "fc3":
                    network.resize_fc3_layer(int(minirestart_tricks_addneurons_param*network.neurons3))

    optimizer.zero_grad()
    return gradients
    

def train(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, 
          loss_goal: float = None, acc_goal: float = None, load_step: int = 0,
          physical_batch_size: int = 1000, abridged_size: int = 5000,
          save_model: bool = False, seed: int = 0, 
          beta: float = 0.0, delta: float = 0.0, sam: bool = False, sam_out: bool = False, sam_rho: float = 0.0, 
          swa: bool = False, swa_lr: float = 0.0, swa_start: int = 0, ema: bool = False, ema_decay: float = 0.9,
          neigs: int = 0, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1, nproj: int = 0,
          minirestart: bool = False, minirestart_freq: int = -1, minirestart_start: int = 0, 
          minirestart_tricks = [], minirestart_tricks_layers = [], minirestart_tricks_topk: int = 0,
          minirestart_tricks_reducenorm_param: float = 0.1, minirestart_tricks_addnoise_param: float = 0.01,
          minirestart_tricks_addneurons_param: float = 1.1,
          eliminate_outliners_data: bool = False, eliminate_outliners_data_strategy: str = None, 
          eliminate_outliners_data_gamma: str = 0.0, eliminate_outliners_data_lr: float = 0.0,
          eliminate_outliners_features: bool = False, eliminate_outliners_features_strategy: str = None, 
          eliminate_outliners_features_gamma: str = 0.0, eliminate_outliners_features_lr: float = 0.0,
          keep_random_layers: bool = False, keep_random_neurons: bool = False, cifar_analysis = False, gradcam = False
          ):
    
    directory = get_gd_directory(dataset, arch_id, loss, opt, lr, eig_freq, seed, beta, delta, load_step)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    torch.manual_seed(seed)
    print(f"set torch seed: {seed}")

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    non_standardized = dataset.endswith("nonst")

    max_iters = (max_steps-load_step) * len(train_dataset)
    loss_fn, acc_fn = get_loss_and_acc(loss)
    loss_fn.__setattr__("individual", False)
    acc_fn.__setattr__("individual", False)
    loss_fn_ind, acc_fn_ind = get_loss_and_acc(loss)
    loss_fn_ind.__setattr__("individual", True)

    dynamic = False
    if minirestart and "addneurons" in minirestart_tricks:
        dynamic = True
    
    network = load_architecture(arch_id, dataset, dynamic).cuda()
    wandb.run.summary["num_parameters"] = num_parameters(network)

    if load_step:
        assert path.isdir(directory)
        load_directory = get_gd_directory(dataset, arch_id, loss, opt, lr, eig_freq, seed, beta, delta, 0)
        network.load_state_dict(torch.load(f"{load_directory}/model_snapshot_final_{load_step}"))
        print("Loaded model successfully")

    params = filter(lambda p: p.requires_grad, network.parameters())
    optimizer = get_gd_optimizer(params, opt, lr, beta, delta, sam, sam_rho)
    print(f"train {arch_id} of {sum(p.numel() for p in network.parameters() if p.requires_grad)} params",
          f"with {opt} on {dataset} using {loss} objective")
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))
    
    if eliminate_outliners_data and eliminate_outliners_data_lr:
        optimizer_outliners = \
            get_gd_optimizer(filter(lambda p: p.requires_grad, network.parameters()), opt, eliminate_outliners_data_lr, beta, delta, sam_out, sam_rho)
    else:
        optimizer_outliners = None
    if eliminate_outliners_features and eliminate_outliners_features_lr:
        optimizer_features = \
            get_gd_optimizer(filter(lambda p: p.requires_grad, network.parameters()), opt, eliminate_outliners_features_lr, beta, delta)
    else:
        optimizer_features = None

    ema_model, swa_model, swa_scheduler = None, None, None
    if swa:
        assert swa_start > 0 and swa_start < max_steps
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr) # CosineAnnealingLR(optimizer, T_max=100)
    else:
        swa_start = -1
    swa = False
    if ema:
        ema_model = None
        assert ema_decay > 0.0 and ema_decay < 1.0

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), \
        torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step)
    if eliminate_outliners_data:
        train_outliners_ratio = torch.zeros(max_steps)
        if eliminate_outliners_data_lr:
            train_loss_inliners, test_loss_inliners, train_acc_inliners, test_acc_inliners = \
                torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step)
            train_sharpness_inliners, train_sharpness_outliners = torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step)
            train_loss_outliners, test_loss_outliners, train_acc_outliners, test_acc_outliners = \
                torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step)
    if eliminate_outliners_features:
        train_feat_outliners_ratio = torch.zeros(max_iters)
        if eliminate_outliners_data_lr:
            train_loss_feat_outliners, test_loss_feat_outliners, train_acc_feat_outliners, test_acc_feat_outliners = \
                torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step), torch.zeros(max_steps-load_step)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    gradients = 0
    step_eos = -1
    last_step = max_steps
    save_files(directory, [("last_time", last_step)])

    wandb.define_metric("train/step")
    wandb.define_metric("train/grads")
    wandb.define_metric("test/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("test/*", step_metric="test/step")

    for step in range(load_step, max_steps):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset, physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")
        wandb.log({'train/step': step, 'train/acc': train_acc[step], 'train/loss': train_loss[step]})
        wandb.log({'test/step': step, 'test/acc': test_acc[step], 'test/loss': test_loss[step]})

        if eig_freq != -1 and step % eig_freq == 0 and cifar_analysis and dataset.startswith("cifar"):
            logits_dict = {'logits/step': step}
            logits = predict_particular(network, standardize=~non_standardized)
            for k, v in logits.items():
                f = plt.figure(figsize=(5, 5), dpi=100)
                plt.pie(torch.softmax(v, dim=0), labels=CIFAR_LABELS, labeldistance=None)
                plt.legend()
                logits_dict[f"logits/{k}"] = wandb.Image(f)
                plt.close()
            
            if not network._get_name().startswith("Fully_connected"):
                some_visualizations, model_outputs = do_gradcam(network, test_dataset, batch_size=physical_batch_size, targets=None, standardized=~non_standardized)
                gradcam_dict = {f"gradcam/step": step}
                for i, (img, imggradcam, v) in enumerate(some_visualizations):
                    gradcam_dict.update({f"gradcam/img_{i}": [wandb.Image(img), wandb.Image(imggradcam)]})
                    f = plt.figure(figsize=(5, 5), dpi=100)
                    plt.pie(torch.softmax(v, dim=0), labels=CIFAR_LABELS, labeldistance=None)
                    plt.legend()
                    logits_dict[f"logits/img_{i}"] = wandb.Image(f)
                    plt.close()
                wandb.log(gradcam_dict)
            wandb.log(logits_dict)

            class_selectivity, neurons_predictions, class_activations = get_class_selectivity(network, test_dataset)
            class_selectivity_dict = {'selectivity/step':step}
            for layer_k, dict_layer_k in class_selectivity.items():
                for bottleneck_j, class_selectivity_val in dict_layer_k.items():
                    class_selectivity_dict.update({f"selectivity/layer_{layer_k}_bottleneck_{bottleneck_j}": wandb.Histogram(class_selectivity_val)})

            #for layer_k, dict_layer_k in class_activations.items():
            #    for class_i, dict_class_i in dict_layer_k.items():
            #        for bottleneck_j, var in dict_class_i.items():
            #            class_selectivity_dict.update()
            wandb.log(class_selectivity_dict)

            # TODO: bar plot https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            print(f"{step}\teigenvalues:\t{eigs[step//eig_freq, :]}")

        wandb.log({'train/step': step, 'train/e1': eigs[step // eig_freq, 0], 'train/e2': eigs[step // eig_freq, 1], 'train/e_2divlr': 2/lr})

        if save_freq != -1 and step % save_freq == 0:
            torch.save(network.state_dict(), f"{directory}/model_snapshot_step_{step}")

        if eig_freq != -1 and step_eos == -1 and obtained_eos(eigs[:step // eig_freq, 0], lr):
            print(f"{step}\tEOS")
            step_eos = step
            save_files(directory, [("eos_time", step_eos)])


        if (loss_goal != None and test_loss[step] < loss_goal) or (acc_goal != None and test_acc[step] > acc_goal):
            last_step = step
            save_files(directory, [("last_time", last_step)])
            break

        if swa_start >= step:
            swa = True

        if minirestart and step >= minirestart_start and step % minirestart_freq == 0:
            gradients += do_minirestart(network, train_dataset, physical_batch_size, loss_fn, optimizer,
                                        minirestart_tricks, minirestart_tricks_layers, minirestart_tricks_topk,
                                        minirestart_tricks_reducenorm_param, 
                                        minirestart_tricks_addnoise_param, 
                                        minirestart_tricks_addneurons_param)

        gradients_epoch = train_epoch(network, train_dataset, physical_batch_size, loss_fn, acc_fn, 
                                         optimizer, optimizer_outliners, optimizer_features,
                                         swa, swa_model, swa_scheduler, ema, ema_model, ema_decay,
                                        grad_ind_data=eliminate_outliners_data,
                                        grad_ind_data_strategy=eliminate_outliners_data_strategy,
                                        grad_ind_data_gamma=eliminate_outliners_data_gamma,
                                        grad_ind_grad=eliminate_outliners_features, 
                                        grad_ind_grad_strategy=eliminate_outliners_features_strategy,
                                        grad_ind_grad_gamma=eliminate_outliners_features_gamma,
                                        keep_random_layers=keep_random_layers, keep_random_neurons=keep_random_neurons)
        gradients += gradients_epoch
        wandb.log({'train/step': step, 'train/gradients': gradients})
        """others
        if eliminate_outliners_data:
            train_outliners_ratio[step] = others.pop(0)

            train_loss_inliners[step] = others.pop(0)
            train_acc_inliners[step] = others.pop(0)
            train_loss_outliners[step] = others.pop(0)
            train_acc_outliners[step] = others.pop(0)

            train_sharpness_inliners[step] = others.pop(0)
            train_sharpness_outliners[step] = others.pop(0)

        if eliminate_outliners_features:
            train_feat_outliners_ratio[step] = others.pop(0)

            train_loss_feat_outliners[step] = others.pop(0)
            train_acc_feat_outliners[step] = others.pop(0)
        """

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])], step=last_step)
    if save_model:
        torch.save(network.state_dict(), f"{directory}/model_snapshot_final_{step}")
    print(f"{step}\tFINISHED\tEOS at {step_eos}")

    f = plot(train_loss, train_acc, eigs[:,0], eig_freq, lr, step_eos, load_step, last_step, directory, "plot")
    wandb.log({"summary": wandb.Image(f)})

def plot(train_loss, train_acc, sharpness, eig_freq, lr, eos, first_step, last_step, directory, file):

    f = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(train_loss[:last_step+1])
    plt.xlim(left=first_step)
    plt.title("train loss")
    plt.subplots_adjust(hspace=0)

    plt.subplot(3, 1, 2)
    plt.plot(train_acc[:last_step+1])
    plt.xlim(left=first_step)
    plt.title("train accuracy")
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 3)
    x_range = min(len(sharpness), last_step+1)
    plt.scatter(torch.arange(x_range//eig_freq), sharpness[:x_range//eig_freq], s=5)
    plt.xlim(left=first_step)
    plt.axhline(2. / lr, linestyle='dotted')
    if eos != -1:
        plt.axvline(eos, alpha=0.5, color='orange')
    plt.title("sharpness")
    plt.xlabel("iteration")

    plt.savefig(f"{directory}/{file}.png")
    return f
