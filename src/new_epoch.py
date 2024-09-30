import math
from copy import deepcopy
import torch
from new_sharpness import flatten_gradients, get_hessian_eigenvalues, project_gradient, obtained_eos
from new_log import log
from new_eval import do_eval, do_analysis
from new_main_utils import save_checkpoint
from new_objective import compute_grad_norm, compute_loss_for_single_instance, omega_penalty, sharpness_penalty

# requires grads computed
def do_perturb(network, random_perturbation_alpha, random_perturbation_std, random_perturbation_threshold):
    with torch.no_grad():
        for p in network.parameters():

            if p.grad is not None:
                gradients = p.grad.flatten()
                threshold = torch.quantile(gradients, random_perturbation_threshold)
                mask = (gradients >= threshold).reshape_as(p.grad)

                p.data[mask] = random_perturbation_alpha * p.data[mask] + torch.randn_like(p.data[mask]) * random_perturbation_std

def initialize(network, method, mask = None):

    def weights_init(module, mask):
        methods[method](module.weight[mask].data)

    methods = {
        "xavier_uniform": torch.nn.init.xavier_uniform,
        "xavier_normal": torch.nn.init.xavier_normal,
        "kaiming_uniform": torch.nn.init.kaiming_uniform,
        "kaiming_normal": torch.nn.init.kaiming_normal
    }

    parameters = []
    for param in network.parameters():
        parameters.append(param.view(-1))
    parameters = torch.cat(parameters)

    if mask is None:
        mask = torch.ones_like(parameters, dtype=torch.bool)

    parameters[mask] = methods[method](parameters[mask]) # does it work?s

    idx = 0
    for param in network.parameters():
        param.data.copy_(parameters[idx:idx + param.numel()]).view(param.shape)
        idx += param.numel()

# requires grads computed
def do_reinitialize(network, method, threshold):

    parameters = []
    gradients = []
    for param in network.parameters():
        if param.grad is not None:
            parameters.append(param.view(-1))
            gradients.append(param.grad.view(-1))

    parameters = torch.cat(parameters)
    gradients = torch.cat(gradients)

    _, indices = torch.abs(gradients).sort(descending=True)
    num_weights = int(threshold * parameters.numel())

    mask = indices[:num_weights]
    initialize(network, method, mask)

def do_random_shock(network, random_perturbation, random_perturbation_alpha, random_perturbation_std, random_perturbation_threshold,
                    random_reinitialization, random_reinitialization_threshold):
    if random_perturbation:
        do_perturb(network, random_perturbation_alpha, random_perturbation_std, random_perturbation_threshold)
    if random_reinitialization:
        do_reinitialize(network, random_reinitialization_threshold)

# requires grads computed
def split_network(network, network_copy, separate_weights, separate_weights_threshold, separate_weights_gradient_large, separate_weights_gradient_small, 
                            separate_features, separate_features_threshold, separate_features_gradient_large, separate_features_gradient_small):
    
    if int(separate_weights) + int(separate_features) == 0:
        return network.parameters(), None, 0.0
    assert int(separate_weights) + int(separate_features) == 1
    assert int(separate_weights) == int(separate_weights_gradient_large) + int(separate_weights_gradient_small)
    assert int(separate_features) == int(separate_features_gradient_large) + int(separate_features_gradient_small)
    # FIXME add support to features
    assert separate_weights
    # THINK: should it be 10% of each param, or 10% of the whole net?

    flattened_params = torch.cat([param.view(-1) for param in network_copy.parameters()])
    flattened_grads = flattened_params.grad
    flattened_grad_norms = flattened_grads.norm(dim=0)
    sorted_indices = torch.argsort(flattened_grad_norms, descending=False)
    how_many_indices = int(separate_weights_threshold * len(flattened_grads))
    if separate_weights_gradient_large:
        sorted_indices = sorted_indices[::-1]
    main_indices = sorted_indices[how_many_indices:]
    aux_indices = sorted_indices[:how_many_indices]

    aux_ratio_grad_norm = torch.norm(flattened_grad_norms[aux_indices]) / torch.norm(flattened_grad_norms)

    main_network, aux_network = [], []
    for index, param in flattened_params:
        if index in main_indices:
            main_network.append(param)
        else:
            aux_network.append(param)
    return main_network, aux_network, aux_ratio_grad_norm


def split_data(network, loss_fn, X, y, separate_data, separate_data_threshold, separate_data_gradient_large, separate_data_gradient_small):
    if not separate_data:
        return (X,y), (None, None)
    assert int(separate_data) == int(separate_data_gradient_large) + int(separate_data_gradient_small)
    vmap_loss = torch.vmap(compute_loss_for_single_instance, in_dims=(None, None, 0, 0))
    losses = vmap_loss(network, loss_fn, X, y)
    norm_gradients = [compute_grad_norm(torch.autograd.grad(loss, network.parameters(), retain_graph=True)).cpu().numpy() for loss in losses]
    sorted_indices = torch.argsort(torch.tensor(norm_gradients), descending=False)
    how_many_indices = int(separate_data_threshold * len(norm_gradients))
    if separate_data_gradient_large:
        sorted_indices = sorted_indices[::-1]
    main_indices = sorted_indices[how_many_indices:]
    aux_indices = sorted_indices[:how_many_indices]
    return (X[main_indices], y[main_indices]), (X[aux_indices], y[aux_indices])
    


# TO DO : trak, 
def do_iter(current_iter, device, network, batch, dataset, loss_fn, acc_fn,
                optimizer, swa_model, swa_scheduler, ema_model, complex_compute,
                gradient_proj_compute, evecs, n_leading_eigenvalues, projectors, 
                iterates_full, iterates_bulk, iterates_orthogonal, gradient_freq,
                sharpness_dict, sharpness_quick_dict,
                omega_wd_0, omega_wd_1, omega_wd_2,
                tau_0, tau_1,
                separate_data, separate_data_threshold, separate_data_gradient_large, separate_data_gradient_small, 
                separate_weights, separate_weights_threshold, separate_weights_gradient_large, separate_weights_gradient_small, 
                separate_features, separate_features_threshold, separate_features_gradient_large, separate_features_gradient_small,
                perturb, reinit, random_shock_dict):

    iter_stats = {"train/step": current_iter}
    gradient_computation = 0

    network.to(device)
    (X, y) = batch
    X, y = X.to(device), y.to(device) # if next(network.parameters()).is_cuda: #    X, y = X.cuda(), y.cuda()
    optimizer.zero_grad()
    network.train()

    if complex_compute:

        total_weighted_norm, total_L0_norm, total_L1_norm, total_L2_norm = omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2, return_absolute=True)
        iter_stats["train/gradient/step"] = current_iter
        iter_stats["train/gradient/norm_running_l0"] = total_L0_norm
        iter_stats["train/gradient/norm_running_l1"] = total_L1_norm
        iter_stats["train/gradient/norm_running_l2"] = total_L2_norm
        iter_stats["train/gradient/norm_running_lcomb"] = total_weighted_norm
        total_weighted_penalty, fisher_penalty, hessian_penalty = sharpness_penalty(network, loss_fn, X, y, sharpness_quick_dict, tau_0, tau_1, return_absolute=True)
        iter_stats["train/sharpness/step"] = current_iter
        iter_stats["train/sharpness/fisher_running"] = fisher_penalty
        iter_stats["train/sharpness/hessian_running"] = hessian_penalty
        iter_stats["train/sharpness/penalty_running"] = total_weighted_penalty
        loss_regularization = total_weighted_norm + total_weighted_penalty
        gradient_computation += int(tau_0) + int(tau_1)

        network_copy = None
        if gradient_proj_compute or separate_data or separate_weights or separate_features:
            network_copy = deepcopy(network)
            loss = loss_fn(network_copy(X), y) / len(X) # + loss_regularization ? # compute gradients of the whole batch for splits - how do i calculate?
            loss.backward()
            gradient_computation += 1

        if gradient_proj_compute:
            if evecs is None or evecs.shape[1] < n_leading_eigenvalues:
                evals, evecs = get_hessian_eigenvalues(network, loss_fn, dataset, batch = None, neigs=n_leading_eigenvalues, return_evecs=True, batch_size=sharpness_dict.get("batch_size", 1000), repeats=sharpness_dict.get("repeats", None)) # dataset of batch??
                gradient_computation += sharpness_dict["repeats"] if sharpness_dict["repeats"] is not None else len(dataset) // sharpness_dict["batch_size"]

            flatten_grad = flatten_gradients(network_copy)
            flatten_grad_norm = torch.norm(flatten_grad, p=2) ** 2
            projected_gradient, orthogonal_component, bulk_projection = project_gradient(flatten_grad, evecs[:,:n_leading_eigenvalues])
        
            iterates_full[current_iter // gradient_freq, :] = projectors.mv(flatten_grad.cpu().detach())
            iterates_bulk[current_iter // gradient_freq] = projectors.mv(projected_gradient.cpu().detach())
            iterates_orthogonal[current_iter // gradient_freq] = projectors.mv(orthogonal_component.cpu().detach())
            iter_stats["train/gradient/gradient_norm"] = flatten_grad_norm.item()
            iter_stats["train/gradient/gradient_norm_bulk_percent"] = bulk_projection.item()

        # assign data splits to appropriate optimizers
        (main_X, main_y), (auxdata_X, auxdata_y) = split_data(network, loss_fn, X, y, separate_data, separate_data_threshold, separate_data_gradient_large, separate_data_gradient_small)

        # assign to optimizers appropriate parts
        main_network, aux_network, aux_ratio_grad_norm = split_network(network, network_copy, separate_weights, separate_weights_threshold, separate_weights_gradient_large, separate_weights_gradient_small, 
                                                        separate_features, separate_features_threshold, separate_features_gradient_large, separate_features_gradient_small)
        optimizer.main_optimizer.param_groups[0]['params'] = main_network
        if aux_network is not None:
            optimizer.aux_optimizer.param_groups[0]['params'] = aux_network
        iter_stats["train/ratio_grad_norm_aux"] = aux_ratio_grad_norm
        del network_copy


        with torch.no_grad():
            iter_stats["train/acc_running"] = acc_fn(network(X), y) / len(y)

        main_loss = loss_fn(network(main_X), main_y) / len(main_X) + loss_regularization #loss_reg?
        if separate_data:
            auxdata_loss = loss_fn(network(auxdata_X), auxdata_y) / len(auxdata_X) #loss_reg?
            iter_stats["train/loss_running_auxdata"] = auxdata_loss.item()
            iter_stats["train/ratio_auxdata"] = len(auxdata_X) / len(X)

    else:
        main_loss = loss_fn(network(X), y) / len(X)
        with torch.no_grad():
            iter_stats["train/acc_running"] = acc_fn(network(X), y) / len(y)

    iter_random_shock_dict = random_shock_dict.copy()
    iter_random_shock_dict["random_perturbation"] = perturb and random_shock_dict["random_perturbation"]
    iter_random_shock_dict["random_reinitialization"] = perturb and random_shock_dict["random_reinitialization"]
    do_random_shock(network, **iter_random_shock_dict)

    optimizer.zero_grad()
    main_loss.backward()
    gradient_computation += 1
    optimizer.step(data=False)
    if separate_data:
        optimizer.zero_grad()
        auxdata_loss.backward()
        gradient_computation += 1
        optimizer.step(main=False, aux=False)

    if swa_model is None:
        swa_model = torch.optim.swa_utils.AveragedModel(network)
        if swa_scheduler is not None:
            swa_scheduler.step()
    if ema_model is not None:
        ema_model.update_parameters(network)

    iter_stats["train/gradient_computation"] = gradient_computation
    iter_stats["train/loss_running"] = main_loss.item()

    return iter_stats

def do_epoch(current_iter, device, network, loss_fn, acc_fn, 
             train_dataset, train_dataloader, test_datasets, test_dataloaders,
             max_iters, train_min_acc, train_max_loss, optimizer, swa_model, swa_scheduler, ema_model,
             complex_freq, generic_freq, gradient_freq, n_leading_eigenvalues, projectors, 
             iterates, iterates_loss, iterates_full, iterates_bulk, iterates_orthogonal, iterates_freq,
             step_eos_e1, sharpness_dict, hessian_freq, step_eos_fim, sharpness_quick_dict, fisher_freq,
             random_shock_dict, perturb_freq, reinit_freq, analysis_freq, skip_analysis_start, 
             analysis_fn, analysis_dict, all_logits_dict, all_evals, save_dir, save_freq, info_freq, extra_hyperparams):

    for batch in train_dataloader:
        current_results = dict()
        complex_compute = int(skip_analysis_start) + current_iter % complex_freq == 0 and complex_freq > 0
        generic_compute = int(skip_analysis_start) + current_iter % generic_freq == 0 and generic_freq > 0
        gradient_proj_compute = int(skip_analysis_start) + current_iter % gradient_freq == 0 and gradient_freq > 0
        projection_compute = int(skip_analysis_start) + current_iter % iterates_freq == 0 and iterates_freq > 0
        analysis = int(skip_analysis_start) + current_iter % analysis_freq == 0 and analysis_freq > 0
        hessian_compute = int(skip_analysis_start) + current_iter % hessian_freq == 0 and hessian_freq > 0
        fisher_compute = int(skip_analysis_start) + current_iter % fisher_freq == 0 and fisher_freq > 0
        perturb = int(skip_analysis_start) + current_iter % perturb_freq == 0 and perturb_freq > 0
        reinit = int(skip_analysis_start) + current_iter % reinit_freq == 0 and reinit_freq > 0 
        save = int(skip_analysis_start) + current_iter % save_freq == 0 and save_freq > 0
        info = int(skip_analysis_start) + current_iter % info_freq == 0 and info_freq > 0

        if info:
            print(f"*****iter {current_iter}*****")
        
        eval_results = do_eval(current_iter, network,
                               train_dataset, train_dataloader, test_datasets, test_dataloaders, 
                               generic_compute, loss_fn, acc_fn, 
                               hessian_compute, fisher_compute, sharpness_dict)
        
        if info and generic_compute:
            print(f"train_acc {eval_results[f'train/acc'].item():.5f} | train_loss {eval_results[f'train/loss'].item():.5f}")
            print("test_acc,test_loss ", "".join([f"| {eval_results[f'test_{i}/acc'].item():.5f} | {eval_results[f'test_{i}/loss'].item():.5f} |" for i in range(len(test_datasets))]))

        # dumb
        # eval_results["sharpness/train/evals"] = torch.randn(10, device="cpu")
        # eval_results["sharpness/train/evecs"] = torch.randn(sum(p.numel() for p in network.parameters()), 10, device="cpu")
        current_results.update(eval_results)

        if step_eos_e1 != -1 and hessian_compute:
            sh = eval_results.get("sharpness/train/lanczos_e1", -1.0) #sharpness_e1.append()
            if obtained_eos(sh, optimizer.get_lr(), window=5, relative_error=0.1):
                step_eos_e1 = current_iter
        if step_eos_fim != -1 and fisher_compute:
            sh = eval_results.get("sharpness/train/fim", -1.0) #sharpness_fim.append()
            if obtained_eos(sh, optimizer.get_lr(), window=5, relative_error=0.1):
                step_eos_fim = current_iter

        if projection_compute:
            iterates[current_iter // iterates_freq] = projectors.mv(torch.nn.utils.parameters_to_vector(network.parameters()).cpu().detach())
            iterates_loss[current_iter // iterates_freq] = eval_results.get("train/loss", -1)

        if analysis:
            analysis_results = do_analysis(current_iter, network, loss_fn=loss_fn, analysis_fn=analysis_fn,
                                           test_datasets=test_datasets, test_dataloaders=test_dataloaders,
                                           all_logits_dict=all_logits_dict,
                                           evals = eval_results.get("sharpness/train/evals", None), all_evals=all_evals, **analysis_dict)
            current_results.update(analysis_results)

        iter_results = do_iter(current_iter, device, network, batch, train_dataset, loss_fn, acc_fn, optimizer, 
                               swa_model, swa_scheduler, ema_model, complex_compute,
                               gradient_proj_compute, eval_results.get("sharpness/train/evecs", None), n_leading_eigenvalues, projectors, 
                               iterates_full, iterates_bulk, iterates_orthogonal, gradient_freq,
                               sharpness_dict, sharpness_quick_dict, **extra_hyperparams,
                               perturb=perturb, reinit=reinit, random_shock_dict=random_shock_dict)
        current_results.update(iter_results)
        # potential remove "sharpness/train/evecs"

        log(current_results, current_iter)
        if save:
            save_checkpoint(network, save_dir, current_iter)
        current_iter += 1
        if current_iter >= max_iters or current_results.get("train/acc_running", 0.0) >= train_min_acc or current_results.get("train/loss_running", math.inf) <= train_max_loss:
            return -1, current_iter

    return current_iter, current_iter
