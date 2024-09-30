from math import sqrt
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import numpy as np
import torch

from new_opti import get_name_optimizer, get_optimizer
from new_gradcam import do_gradcam
from new_archs import get_target_layers, get_last_layer, load_network
from new_data_main import prepare_data
from new_objective import compute_metrics_per_class, get_loss_and_acc, compute_losses_dataloader
from new_selective_neurons import get_class_selectivity
from new_sharpness import compute_sharpness
from new_log import log

def prepare(args):
    
    results_dir = os.environ["RESULTS"]
    directory = os.path.join(results_dir, args.load_checkpoint_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"set torch seed: {args.seed}")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    network_kwargs = {}
    network = load_network(args.arch_id, [int(x) for x in args.input_shape], args.output_shape, args.softmax, network_kwargs).to(device)

    loss_fn, acc_fn = get_loss_and_acc(args.loss)
    loss_fn_ind, acc_fn_ind = get_loss_and_acc(args.loss, individual=True)

    train_dataloader, train_dataset, test_datasets, test_dataloaders = prepare_data(args.train_dataset, args.test_datasets, args.loss, args.train_batch_size, args.test_batch_size)

    optimizer = get_optimizer(filter(lambda p: p.requires_grad, network.parameters()), args.opti_type, args.lr, args.aux_type, args.aux_lr, args.data_type, args.data_lr,
                              args.momentum, args.dampening, args.wp, args.rho, args.adaptive)

    prepared_dict = {"directory": directory,
                     "device": device,
                     "network": network,
                     "loss_fn": loss_fn,
                     "acc_fn": acc_fn,
                     "train_dataloader": train_dataloader,
                     "train_dataset": train_dataset,
                     "test_daloaders": test_dataloaders,
                     "test_datasets": test_datasets,
                     "optimizer": optimizer}
    return prepared_dict

def do_eval(current_iter, network, train_dataset, train_dataloader, test_datasets, test_dataloaders, generic_compute, loss_fn, acc_fn, hessian_compute, fisher_compute, sharpness_eval_dict):
    network.eval()
    results = dict()
    if generic_compute:
        train_loss, train_acc = compute_losses_dataloader(network, [loss_fn, acc_fn], train_dataloader, no_grad=True)
        test_loss_acc_s = [compute_losses_dataloader(network, [loss_fn, acc_fn], test_dataloader, no_grad=True) for test_dataloader in test_dataloaders]
        results.update({"train/iter": current_iter, "train/loss": train_loss, "train/acc": train_acc})
        for i, (test_loss, test_acc) in enumerate(test_loss_acc_s):
            results.update({f"test_{i}/iter": current_iter, f"test_{i}/loss": test_loss, f"test_{i}/acc": test_acc})
    if hessian_compute:
        # sharpness_dict = {"method": "lanczos", "batch_size": sharpness_batch_size, "repeats": sharpness_repeats, "neigs": sharpness_neigs, "num_iterations": sharpness_num_iterations, "only_top": False}
        evals, evecs = compute_sharpness(network, loss_fn, "lanczos", dataset=train_dataset, only_top=False, return_evecs=True, **sharpness_eval_dict)
        sharpness_results = {"sharpness/train/iter": current_iter, "sharpness/train/lanczos_e1": evals[0], "sharpness/train/lanczos_e2": evals[1]}  
        sharpness_results.update({"sharpness/train/evals": evals, "sharpness/train/evecs": evecs})
        results.update(sharpness_results)

    if fisher_compute:
        trace_fim = compute_sharpness(network, loss_fn, "fim", dataset=train_dataset, **sharpness_eval_dict)
        sharpness_results = {"sharpness/train/iter": current_iter, "sharpness/train/fim": trace_fim}  
        results.update(sharpness_results)
    return results

def prepare_logits(dataset_name: str, LABELS, from_gradcam_examples: int):
    if dataset_name.startswith("cifar100") or dataset_name.startswith("coloured"):
        from_particular = 100
    if dataset_name.startswith("cifar10") or dataset_name.startswith("mnist") or dataset_name.startswith("fashion"):
        from_particular = 10

    from_particular = 0 # tmp

    n_inspect = from_particular + from_gradcam_examples

    logits_dict = {}
    for i in range(n_inspect):
        logits_dict[i] = {}
        for c in LABELS:
            logits_dict[i][c] = []
    return logits_dict

def do_analysis(current_iter, network, arch_id, loss_fn, 
                train_dataset_name, test_datasets_names, labels, test_datasets, test_dataloaders, test_labels,
                analysis_fn, analysis_freq, skip_analysis_start,
                plot_logits_softmax, 
                gradcam_batch_size, gradcam_counter, gradcam_rescale,
                selectivity_batch_size, selectivity_epsilon,
                perclass_batch_size, sparsity_epsilon,
                all_logits_dict,
                evals, all_evals):

    network.eval()
    target_layers = get_target_layers(arch_id, network)
    last_layer = get_last_layer(arch_id, network)

    analysis_dict = {}

    for j, dataset in enumerate(test_datasets):
        # GRADCAM + LOGITS
        gradcam_dict = {f"gradcam/test_{j}/step": current_iter}
        some_visualizations, model_outputs = do_gradcam(network, dataset, gradcam_batch_size, gradcam_counter, rescale=gradcam_rescale, target_layers=target_layers)
        for i, (img, imggradcam, v) in enumerate(some_visualizations):
            gradcam_dict[f"gradcam/test_{j}/img_{i}"] = [img, imggradcam] # gradcam_dict[f"gradcam/test_{j}_img_{i}"] = [img, imggradcam]
            if plot_logits_softmax:
                v = torch.softmax(v, dim=0)
            for c, l in zip(test_labels[j], v):
                all_logits_dict[j][i][c].append(l)

            f = plt.figure(figsize=(5, 5), dpi=100)
            if int(skip_analysis_start) + current_iter == 0:
                bot = 0.0
                for single_v, single_l in zip(all_logits_dict[j][i].values(), all_logits_dict[j][i].keys()):
                    plt.bar(x=0, height=single_v[0], label=single_l, bottom=bot)
                    bot += single_v[0]
            else:
                plt.stackplot(torch.arange(len(list(all_logits_dict[j][i].values())[0]))*analysis_freq+int(skip_analysis_start), all_logits_dict[j][i].values(), labels=all_logits_dict[j][i].keys(), alpha=0.8)
            plt.legend(reverse=True)
            analysis_dict[f"logits/test_{j}/img_{i}"] = f
            plt.close()
        analysis_dict.update(gradcam_dict)

        # SELECTIVITY
        class_selectivity, neurons_predictions, class_activations = get_class_selectivity(arch_id, network, dataset, loss_fn, selectivity_batch_size, selectivity_epsilon, last_layer)
        class_selectivity_dict = {f"selectivity/step": current_iter}
        for layer_k, dict_layer_k in class_selectivity.items():
            for bottleneck_j, class_selectivity_val in dict_layer_k.items():
                class_selectivity_dict.update({f"selectivity/test_{j}/layer_{layer_k}_bottleneck_{bottleneck_j}_hist": class_selectivity_val})
                f = plt.figure(figsize=(5, 5), dpi=100)
                n_bins = int(sqrt(len(v))) if len(v) > 4000 else 64
                plt.hist(class_selectivity_val, n_bins)
                plt.close()
                class_selectivity_dict.update({f"selectivity/test_{j}/layer_{layer_k}_bottleneck_{bottleneck_j}_histplot": f})
        analysis_dict.update(class_selectivity_dict)

        # PERCLASS
        perclass_dict = {f"perclass/test_{j}/step": current_iter}
        outs = compute_metrics_per_class(network, dataset, perclass_batch_size)
        precisions, recalls, f1s = outs
        labels_classes = test_labels[j] if test_labels[j] is not None else range(len(precisions))
        for metric, out in zip(["precision", "recall", "f1"], outs):
            data = [[name, m] for (name, m) in zip(labels_classes, out)]
            perclass_dict[f"perclass/test_{j}/{metric}"] = (data, ["class_name", metric], "class_name", metric, f"test_{j} / {metric} per class")

        data = [[name, precision, recall, f1] for (name, precision, recall, f1) in zip(labels_classes, precisions, recalls, f1s)]
        perclass_dict[f"perclass/test_{j}/all"] = (data, ["class_name", "precision", "recall", "f1"], "class_name", "all", "test_{j} / Metrics Per Class")
        analysis_dict.update(perclass_dict)

    # SPARSITY
    sparsity_dict = dict()
    total_params_with_grad = 0
    all_params_norms, all_weights_norms = [], []
    weights_close_to_zero = 0
    for name, param in network.named_parameters():
        if param is None or not param.requires_grad:
            continue
        norm_value = torch.norm(param).item()
        all_params_norms.append(norm_value)
        weights_close_to_zero += torch.sum(abs(param) < sparsity_epsilon).item()
        all_weights_norms.extend(param.data.cpu().numpy().flatten())
        total_params_with_grad += param.numel()

    f = plt.figure(figsize=(5, 5), dpi=100)
    plt.hist(all_params_norms)
    plt.xlabel("L2 Norm")
    plt.ylabel("Number of weight params")
    sparsity_dict[f"sparsity/L2_norm_hist_params"] = f
    plt.close()
    f = plt.figure(figsize=(5, 5), dpi=100)
    plt.hist(all_weights_norms, bins=int(sqrt(total_params_with_grad)))
    plt.xlabel("Weights")
    plt.ylabel("Number of weight params")
    plt.axvline(-sparsity_epsilon, color="yellow")
    plt.axvline(sparsity_epsilon, color="yellow", label=f"eps: {sparsity_epsilon} near 0: {(weights_close_to_zero / total_params_with_grad):.5f}")
    sparsity_dict[f"sparsity/L2_norm_hist_weights"] = f
    plt.close()
    sparsity_dict["sparsity/close_to_zero"] = weights_close_to_zero / total_params_with_grad
    analysis_dict.update(sparsity_dict)

    # TOP EIGENVALUES
    if evals is not None:
        all_evals.append(evals)
        eigs_cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "green"], N=len(evals))
        f = plt.figure(figsize=(5, 5), dpi=100)
        for i in range(len(evals)):
            plt.scatter(torch.arange(len(all_evals))*analysis_freq+int(skip_analysis_start), np.array(all_evals)[:, i], color=eigs_cmap(i/(len(evals)-1)))
        plt.close()
        analysis_dict.update({"sharpness/train/topeigen": f, "sharpness/train/iter": current_iter})

    # log(analysis_dict, current_iter)
    return analysis_dict
