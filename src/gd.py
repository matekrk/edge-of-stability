from os import makedirs, path
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import wandb

import torch
from torch.nn.utils import parameters_to_vector
import torch.functional as F

from archs import load_architecture
from utilities import get_dataloader, iterate_dataset, split_batch, get_gd_params, get_gd_optimizer, get_gd_directory_extra, get_gd_directory, \
    get_hessian_eigenvalues, compute_empirical_sharpness, obtained_eos, \
    compute_grad_norm, get_loss_and_acc, compute_losses, compute_loss_for_single_instance, compute_metrics_per_class, \
    save_files, save_files_final, split_batch_trak, str_to_layers, weights_init, num_parameters
from plot import plot_gd
from sam import SAM

from data_generic import load_dataset, take_first, get_labels, get_predict_particular
from data_cifar import predict_particular_cifar
from data_coloured_mnist import load_coloured_mnist_mask, predict_particular_coloured_mnist
from data_mnist import load_mnist_mask, predict_particular_mnist
from data_fashion import load_fashion_mask, predict_particular_fashion
from data_svhn import load_svhn_mask, predict_particular_svhn

from gradcam import do_gradcam, get_target_layers
from selective_neurons import get_class_selectivity
from traker import trak, trak_onebatch, visualize
from debug import do_trak_traintrain

from epoch import train_epoch

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
          loss_goal: float = None, acc_goal: float = None, load_step: int = 0, load_file: str = None,
          physical_batch_size: int = 1000, abridged_size: int = 5000, trakked_data: bool = False, trakked_models = None,
          save_model: bool = False, seed: int = 0, path_extra = None,
          beta: float = 0.0, delta: float = 0.0, sam: bool = False, sam_out: bool = False, sam_rho: float = 0.0, 
          swa: bool = False, swa_lr: float = 0.0, swa_start: int = 0, ema: bool = False, ema_decay: float = 0.9,
          omega_wd_0: float = 0.0, omega_wd_1: float = 0.0, omega_wd_2: float = 0.0,
          neigs: int = 0, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1, nproj: int = 0, 
          minirestart: bool = False, minirestart_freq: int = -1, minirestart_start: int = 0, 
          minirestart_tricks = [], minirestart_tricks_layers = [], minirestart_tricks_topk: int = 0,
          minirestart_tricks_reducenorm_param: float = 0.1, minirestart_tricks_addnoise_param: float = 0.01,
          minirestart_tricks_addneurons_param: float = 1.1,
          eliminate_outliners_data: bool = False, eliminate_outliners_data_strategy: str = None, 
          eliminate_outliners_data_gamma: str = 0.0, eliminate_outliners_data_lr: float = 0.0,
          eliminate_outliners_features: bool = False, eliminate_outliners_features_strategy: str = None, 
          eliminate_outliners_features_gamma: str = 0.0, eliminate_outliners_features_lr: float = 0.0,
          keep_random_layers: bool = False, keep_random_neurons: bool = False, 
          analysis_freq: int = -1, analysis_sparsity = False, analysis_class_selectivity = False, analysis_gradcam = False, 
          analysis_cifar = False, analysis_coloured_mnist = False, analysis_svhn = False, analysis_mnist = False, analysis_fashion = False, 
          analysis_trak = False, analysis_perclass = False, analysis_eigenvalues = 0
          ):

    directory_extra = get_gd_directory_extra(eliminate_outliners_data, eliminate_outliners_data_strategy, eliminate_outliners_data_gamma, eliminate_outliners_data_lr, sam_out)
    directory = get_gd_directory(dataset, arch_id, loss, opt, lr, sam, eig_freq, seed, beta, delta, load_step, directory_extra, path_extra)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    torch.manual_seed(seed)
    print(f"set torch seed: {seed}")

    if trakked_data:
        mask = do_trak_traintrain(trakked_models)
        train_dataset, test_dataset = load_mnist_mask(dataset, loss, mask) # FIXME
    else:
        train_dataset, test_dataset = load_dataset(dataset, loss)
    
    abridged_train = take_first(train_dataset, abridged_size)
    standardized = dataset.endswith("st")

    max_iters = (max_steps-load_step) * (len(train_dataset) // physical_batch_size)
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
        if load_file is None:
            load_dir = get_gd_directory(dataset, arch_id, loss, opt, lr, sam, eig_freq, seed, beta, delta, 0, directory_extra, path_extra)
            assert path.isdir(load_dir)
            load_file = f"{load_dir}model_snapshot_step_{load_step-1}.pt"
        assert path.isfile(load_file)
        network.load_state_dict(torch.load(load_file))
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
    iterates = torch.zeros((max_steps-load_step) // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros((max_steps-load_step) // eig_freq if eig_freq >= 0 else 0, neigs)

    target_layers = get_target_layers(arch_id, network)

    LABELS = get_labels(dataset)

    predict_fns = get_predict_particular(analysis_cifar, analysis_coloured_mnist, analysis_svhn, analysis_mnist, analysis_fashion)
    true_label = False if "coloured" in dataset and "TRrandom" in dataset else True
    plot_logits_softmax = True

    if analysis_cifar or analysis_mnist or analysis_fashion:
        logits_dict = {}
        for i in range(13):
            logits_dict[i] = {}
            for c in LABELS:
                logits_dict[i][c] = []
    elif analysis_coloured_mnist or analysis_svhn:
        logits_dict = {}
        for i in range(110):
            logits_dict[i] = {}
            for c in LABELS:
                logits_dict[i][c] = []
    if analysis_eigenvalues:
        neigs_to_compute_max = 2 * len(LABELS)
        eigs_cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "green"], N=neigs_to_compute_max)
        eigs_max = torch.zeros((max_steps-load_step) // analysis_freq if analysis_freq >= 0 else 0, neigs_to_compute_max)
    else:
        neigs_to_compute_max = neigs
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
        train_loss[step-load_step], train_acc[step-load_step] = compute_losses(network, [loss_fn, acc_fn], train_dataset, physical_batch_size)
        test_loss[step-load_step], test_acc[step-load_step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
        print(f"{step}\t{train_loss[step-load_step]:.3f}\t{train_acc[step-load_step]:.3f}\t{test_loss[step-load_step]:.3f}\t{test_acc[step-load_step]:.3f}")
        wandb.log({'train/step': step, 'train/acc': train_acc[step-load_step], 'train/loss': train_loss[step-load_step]}, step=step)
        wandb.log({'test/step': step, 'test/acc': test_acc[step-load_step], 'test/loss': test_loss[step-load_step]}, step=step)
            
        if analysis_freq != -1 and step % analysis_freq == 0:
            if analysis_trak and step > save_freq:
                train_dataloader = get_dataloader(train_dataset, physical_batch_size) 
                test_dataloader = get_dataloader(test_dataset, physical_batch_size)
                scores = trak(deepcopy(network), directory, train_dataloader, test_dataloader, len(train_dataset), len(test_dataset))
                figs = visualize(scores, train_dataset, test_dataset, directory)
                for i,f in enumerate(figs):
                    wandb.log({f"trak/{i}": wandb.Image(f)}, step=step)

            for predict_fn in predict_fns:
                logits_dict_wandb = {}
                logits_dict_wandb['logits/step'] = step
                step_logits = predict_fn(network, standardize=standardized, return_dict=True, softmax=plot_logits_softmax, true_label=true_label)
                for i, k in enumerate(sorted(step_logits)):
                    for c, l in step_logits[k].items():
                        logits_dict[10+i][c].append(l.item())
                    f = plt.figure(figsize=(5, 5), dpi=100)
                    if step == load_step:
                        bot = 0.0
                        for single_v, single_l in zip(logits_dict[10+i].values(), logits_dict[10+i].keys()):
                            plt.bar(x=0, height=single_v[0], label=single_l, bottom=bot)
                            bot += single_v[0]
                    else:
                        plt.stackplot(range((step-load_step)//analysis_freq+1), logits_dict[10+i].values(), labels=logits_dict[10+i].keys(), alpha=0.8)
                    plt.legend(reverse=True)
                    logits_dict_wandb[f"logits/{k}"] = wandb.Image(f)
                    plt.close()

            calculated_logits = False
            if analysis_gradcam:
                if not network._get_name().startswith("Fully_connected"):
                    some_visualizations, model_outputs = do_gradcam(network, test_dataset, batch_size=physical_batch_size, targets=None, standardized=standardized, target_layers=target_layers)
                    gradcam_dict = {f"gradcam/step": step}
                    for i, (img, imggradcam, v) in enumerate(some_visualizations):
                        if i == 10:
                            break

                        gradcam_dict.update({f"gradcam/img_{i}": [wandb.Image(img), wandb.Image(imggradcam)]})

                        if analysis_cifar or analysis_coloured_mnist or analysis_svhn or analysis_mnist or analysis_fashion:
                            for c, l in zip(LABELS, v):
                                if plot_logits_softmax:
                                    l = torch.softmax(l, dim=0)
                                logits_dict[i][c].append(l)
                
                        f = plt.figure(figsize=(5, 5), dpi=100)
                        if step == load_step:
                            bot = 0.0
                            for single_v, single_l in zip(logits_dict[i].values(), logits_dict[i].keys()):
                                plt.bar(x=0, height=single_v[0], label=single_l, bottom=bot)
                                bot += single_v[0]
                        else:
                            plt.stackplot(range((step-load_step)//analysis_freq+1), logits_dict[i].values(), labels=logits_dict[i].keys(), alpha=0.8)
                        plt.legend(reverse=True)
                        logits_dict_wandb[f"logits/img_{i}"] = wandb.Image(f)
                        plt.close()

                    wandb.log(gradcam_dict, step=step)
                    calculated_logits = True
        
            if not calculated_logits and (analysis_cifar or analysis_coloured_mnist or analysis_svhn or analysis_mnist or analysis_fashion):
                with torch.no_grad():
                    for (X, y) in iterate_dataset(test_dataset, physical_batch_size):
                        logits = network(X)
                        for i in range(10):
                            img = X[i]
                            out = logits[i]
                            if standardized:
                                img = torch.clamp((img+1)/2, a_max=1, a_min=0).permute((1,2,0)).cpu().numpy()
                            if plot_logits_softmax:
                                l = torch.softmax(l, dim=0)
                            for c, l in zip(LABELS, out):
                                logits_dict[i][c].append(l.cpu())
                            if step == load_step:
                                bot = 0.0
                                for single_v, single_l in zip(logits_dict[i].values(), logits_dict[i].keys(), alpha=0.8):
                                    plt.bar(x=0, height=single_v[0], label=single_l, bottom=bot)
                                    bot += single_v[0]
                            else:
                                plt.stackplot(range(step//analysis_freq+1), logits_dict[i].values(), labels=logits_dict[i].keys(), alpha=0.8)
                            plt.legend(reverse=True)
                            logits_dict_wandb[f"logits/img_{i}"] = wandb.Image(f)
                        break
                    
            if analysis_cifar or analysis_coloured_mnist or analysis_svhn or analysis_mnist or analysis_fashion:
                wandb.log(logits_dict_wandb, step=step)

            if analysis_class_selectivity:
                class_selectivity, neurons_predictions, class_activations = get_class_selectivity(network, test_dataset, loss_fn)
                class_selectivity_dict = {'selectivity/step':step}
                for layer_k, dict_layer_k in class_selectivity.items():
                    for bottleneck_j, class_selectivity_val in dict_layer_k.items():
                        class_selectivity_dict.update({f"selectivity/layer_{layer_k}_bottleneck_{bottleneck_j}": wandb.Histogram(class_selectivity_val)})

                #for layer_k, dict_layer_k in class_activations.items():
                #    for class_i, dict_class_i in dict_layer_k.items():
                #        for bottleneck_j, var in dict_class_i.items():
                #            class_selectivity_dict.update()
                wandb.log(class_selectivity_dict, step=step)

            if analysis_perclass:
                outs = compute_metrics_per_class(network, test_dataset, physical_batch_size)
                precisions, recalls, f1s = outs
                labels_classes = LABELS if LABELS is not None else range(len(precisions))
                for metric, out in zip(["precision", "recall", "f1"], outs):
                    data = [[name, m] for (name, m) in zip(labels_classes, out)]
                    table = wandb.Table(data=data, columns=["class_name", metric])
                    wandb.log({f"metrics_perclass/{metric}" : wandb.plot.bar(table, "class_name", metric, f"{metric} per class")}, step=step)

                data = [[name, precision, recall, f1] for (name, precision, recall, f1) in zip(labels_classes, precisions, recalls, f1s)]
                table = wandb.Table(data=data, columns=["class_name", "precision", "recall", "f1"])
                wandb.log({"metrics_perclass/all" : wandb.plot.bar(table, "class_name", "all", "Metrics Per Class")}, step=step)

            if analysis_sparsity:

                sparsity_dict_wandb = dict()
                eps = 1e-3
                total_params_with_grad = 0
                all_norms = []
                weights_close_to_zero = 0
                for name, param in network.named_parameters():
                    if not param.requires_grad:
                        continue
                    norm_value = torch.norm(param).item()
                    all_norms.append(norm_value)
                    weights_close_to_zero += torch.count_nonzero((torch.abs(param.data) < eps))
                    total_params_with_grad += param.numel()

                f = plt.figure(figsize=(5, 5), dpi=100)
                plt.hist(all_norms)
                #plt.xlabel("L2 Norm")
                #plt.ylabel("Number of Weights")
                sparsity_dict_wandb[f"sparsity/L2_norm_hist"] = wandb.Image(f)
                plt.close()
                sparsity_dict_wandb["sparsity/close_to_zero"] = weights_close_to_zero / total_params_with_grad
                wandb.log(sparsity_dict_wandb, step=step)

        if (eig_freq != -1 and (step-load_step) % eig_freq == 0) or (analysis_freq != -1 and (step-load_step) % analysis_freq == 0):
            current_eigs = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs_to_compute_max,
                                                                physical_batch_size=physical_batch_size)
            eigs[(step-load_step) // eig_freq, :] = current_eigs[:neigs]
            print(f"{step}\teigenvalues:\t{eigs[(step-load_step)//eig_freq, :]}")

            eigs_dict_wandb = {'train/e1': eigs[(step-load_step) // eig_freq, 0], 'train/e1_scaled': eigs[(step-load_step) // eig_freq, 0] * lr * 0.5, 'train/e2': eigs[(step-load_step) // eig_freq, 1], 'train/e_2divlr': 2/lr}

            if analysis_eigenvalues and analysis_freq != -1 and (step-load_step) % analysis_freq == 0:
                eigs_max[(step-load_step) // analysis_freq, :] = current_eigs
                f = plt.figure(figsize=(5, 5), dpi=100)
                for i in range(neigs_to_compute_max):
                    plt.scatter(torch.arange(load_step, load_step+1 + step//eig_freq), eigs_max[:(step-load_step+1) // eig_freq, i], color=eigs_cmap(i/(neigs_to_compute_max-1)))
                top_eigs = wandb.Image(f)
                plt.close()
                eigs_dict_wandb['eigs/top_eigs'] = top_eigs
                
            wandb.log(eigs_dict_wandb, step=step)

        if save_model and save_freq != -1 and (step-load_step) % save_freq == 0:
            torch.save(network.state_dict(), f"{directory}/model_snapshot_step_{step}.pt")

        if eig_freq != -1 and step_eos == -1 and obtained_eos(eigs[:(step-load_step) // eig_freq, 0], lr):
            print(f"{step}\tEOS")
            step_eos = step
            save_files(directory, [("eos_time", step_eos)])


        if (loss_goal != None and test_loss[step-load_step] < loss_goal) or (acc_goal != None and test_acc[step-load_step] > acc_goal):
            last_step = step
            save_files(directory, [("last_time", last_step)])
            break

        if swa_start >= step:
            swa = True

        if minirestart and step >= minirestart_start and (step-load_step) % minirestart_freq == 0:
            gradients += do_minirestart(network, train_dataset, physical_batch_size, loss_fn, optimizer,
                                        minirestart_tricks, minirestart_tricks_layers, minirestart_tricks_topk,
                                        minirestart_tricks_reducenorm_param, 
                                        minirestart_tricks_addnoise_param, 
                                        minirestart_tricks_addneurons_param)

        gradients_epoch = train_epoch(network, train_dataset, physical_batch_size, loss_fn, acc_fn, 
                                        optimizer, optimizer_outliners, optimizer_features,
                                        swa, swa_model, swa_scheduler, ema, ema_model, ema_decay,
                                        omega_wd_0, omega_wd_1, omega_wd_2,
                                        eliminate_outliners_data=eliminate_outliners_data,
                                        eliminate_outliners_data_strategy=eliminate_outliners_data_strategy,
                                        eliminate_outliners_data_gamma=eliminate_outliners_data_gamma,
                                        # eliminate_outliners_data_lr=eliminate_outliners_data_lr,
                                        eliminate_outliners_features=eliminate_outliners_features, 
                                        eliminate_outliners_features_strategy=eliminate_outliners_features_strategy,
                                        eliminate_outliners_features_gamma=eliminate_outliners_features_gamma,
                                        # eliminate_outliners_features_lr=eliminate_outliners_features_lr,
                                        keep_random_layers=keep_random_layers, keep_random_neurons=keep_random_neurons, 
                                        log_epoch=step if step%eig_freq == 0 else None)
        gradients += gradients_epoch
        wandb.log({'train/step': step, 'train/gradients': gradients}, step=step)
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

    last_step = step
    save_files_final(directory,
                     [("eigs", eigs[:(step-load_step + 1) // eig_freq]), ("iterates", iterates[:(step-load_step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step-load_step + 1]), ("test_loss", test_loss[:step-load_step + 1]),
                      ("train_acc", train_acc[:step-load_step + 1]), ("test_acc", test_acc[:step-load_step + 1])], step=last_step)
    if save_model:
        torch.save(network.state_dict(), f"{directory}/model_snapshot_final_{step}.pt")
    print(f"{step}\tFINISHED\tEOS at {step_eos}")

    f = plot(train_loss, train_acc, eigs[:,0], eig_freq, lr, step_eos, load_step, last_step, directory, "plot")
    wandb.log({"summary": wandb.Image(f)}, step=step)

def plot(train_loss, train_acc, sharpness, eig_freq, lr, eos, first_step, last_step, directory, file):

    f = plt.figure(figsize=(5, 5), dpi=100)

    xrange = range(first_step, (last_step-first_step)+1)

    plt.subplot(3, 1, 1)
    plt.plot(xrange, train_loss[:last_step+1])
    #plt.xlim(left=first_step)
    plt.title("train loss")
    plt.subplots_adjust(hspace=0)

    plt.subplot(3, 1, 2)
    plt.plot(xrange, train_acc[:last_step+1])
    #plt.xlim(left=first_step)
    plt.title("train accuracy")
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 3)
    xrangetop = min(len(sharpness), last_step-first_step+1)
    plt.scatter(torch.arange(first_step, first_step + xrangetop//eig_freq), sharpness[:xrangetop//eig_freq], s=5)
    #plt.xlim(left=first_step)
    plt.axhline(2. / lr, linestyle='dotted')
    if eos != -1:
        plt.axvline(eos, alpha=0.5, color='orange')
    plt.title("sharpness")
    plt.xlabel("iteration")

    plt.savefig(f"{directory}/{file}.png")
    return f
