
import os
import numpy as np
import torch

from new_opti import get_name_optimizer, get_optimizer
from new_archs import get_target_layers, get_last_layer, load_network
from new_objective import get_loss_and_acc
from new_data_main import prepare_data
from new_log import log
from new_data_labels import get_labels
from new_epoch import do_epoch
from new_main_utils import save_checkpoint, save_files, visualize_grads, visualize_iters
from new_eval import prepare_logits

def get_gd_directory_extra(eliminate_outliners_data, eliminate_outliners_data_strategy, eliminate_outliners_data_gamma, eliminate_outliners_data_lr, sam_out):
    if eliminate_outliners_data:
        s = ""
        if sam_out:
            s += "sam_"
        s += f"{eliminate_outliners_data_strategy}_g{eliminate_outliners_data_gamma}_lr{eliminate_outliners_data_lr}"
        return s
    return None

def get_gd_directory(dataset: str, arch_id: str, loss: str, opt: str, lr: float, sam: bool,eig_freq: int, seed: int, 
                     beta: float = None, delta: float = None, start_step: int = 0, algo_extra = None, extra = None):
    """Return the directory in which the results should be saved."""
    results_dir = os.environ["RESULTS"]

    if sam:
        opt_str = f"{opt}_sam"
    else:
        opt_str = opt

    directory = f"{results_dir}/{dataset}/{arch_id}/{loss}/{opt_str}/"
    if algo_extra is not None:
        directory += f"{algo_extra}/"

    if opt == "sgd":
        directory += f"lr_{lr}"
    elif opt == "polyak" or opt == "nesterov":
        directory += f"{directory}/lr_{lr}_beta_{beta}"
    if delta is None:
        directory += "/"
    else:
        directory += f"_delta{delta}/"
    directory += f"seed_{seed}/"

    path = f"{directory}freq_{eig_freq}/start_{start_step}/"

    if extra is not None:
        path += f"{extra}/"

    return path

def prepare(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"set torch seed: {args.seed}")

    results_dir = os.environ["RESULTS"]
    opt_str = get_name_optimizer(args.opti_type, args.lr, args.aux_type, args.aux_lr, args.data_type, args.data_lr, args.beta_momentum, args.delta_dampening, args.omega_wd, args.rho_sam, args.adaptive_sam)
    args.optimizer_str = opt_str
    beg_dir = "new/" + args.train_dataset
    directory = os.path.join(results_dir, beg_dir, opt_str, args.path_extra, f"s_{args.seed}")
    print(f"output directory: {directory}")
    os.makedirs(directory, exist_ok=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    network_kwargs = {}
    network = load_network(args.arch_id, [int(x) for x in args.input_shape], args.output_shape, args.softmax, **network_kwargs).to(device)

    loss_fn, acc_fn = get_loss_and_acc(args.loss)
    loss_fn_ind, acc_fn_ind = get_loss_and_acc(args.loss, individual=True)

    train_dataset, train_dataloader, test_datasets, test_dataloaders = prepare_data(args.train_dataset, args.test_datasets, args.loss, args.train_batch_size, args.test_batch_size) # trak
    
    optimizer = get_optimizer(list(filter(lambda p: p.requires_grad, network.parameters())), args.opti_type, args.lr, args.aux_type, args.aux_lr, args.data_type, args.data_lr,
                              args.beta_momentum, args.delta_dampening, args.omega_wd, args.rho_sam, args.adaptive_sam)
    
    sharpness_dict = {"batch_size": args.sharpness_batch_size, "neigs": args.sharpness_neigs, "repeats": args.sharpness_repeats} # "num_iterations": args.sharpness_num_iterations, "only_top": args.sharpness_only_top
    sharpness_quick_dict = {"repeats": args.sharpness_repeats, "batch_size": args.sharpness_quick_batch_size}

    random_shock_dict = {"random_perturbation": args.random_perturbation, 
                         "random_perturbation_alpha": args.random_perturbation_alpha, 
                         "random_perturbation_std": args.random_perturbation_std, 
                         "random_perturbation_threshold": args.random_perturbation_threshold, 
                         "random_reinitialization": args.random_reinitialization, 
                         "random_reinitialization_threshold": args.random_reinitialization_threshold}

    extra_hyperparams = {"omega_wd_0": args.omega_wd_0, 
                         "omega_wd_1": args.omega_wd_1, 
                         "omega_wd_2": args.omega_wd_2, 
                         "tau_0": args.tau_0, 
                         "tau_1": args.tau_1, 
                         "separate_data": args.separate_data, 
                         "separate_data_threshold": args.separate_data_threshold, 
                         "separate_data_gradient_large": args.separate_data_gradient_large, 
                         "separate_data_gradient_small": args.separate_data_gradient_small,
                         "separate_weights": args.separate_weights, 
                         "separate_weights_threshold": args.separate_weights_threshold, 
                         "separate_weights_gradient_large": args.separate_weights_gradient_large, 
                         "separate_weights_gradient_small": args.separate_weights_gradient_small, 
                         "separate_features": args.separate_features, 
                         "separate_features_threshold": args.separate_features_threshold, 
                         "separate_features_gradient_large": args.separate_features_gradient_large, 
                         "separate_features_gradient_small": args.separate_features_gradient_small}

    prepared_dict = {"directory": directory,
                     "device": device,
                     "network": network,
                     "loss_fn": loss_fn,
                     "acc_fn": acc_fn,
                     "loss_fn_ind": loss_fn_ind,
                     "acc_fn_ind": acc_fn_ind,
                     "train_dataset": train_dataset,
                     "train_dataloader": train_dataloader,
                     "test_datasets": test_datasets,
                     "test_dataloaders": test_dataloaders,
                     "optimizer": optimizer,
                     "sharpness_dict": sharpness_dict,
                     "sharpness_quick_dict": sharpness_quick_dict,
                     "random_shock_dict": random_shock_dict,
                     "extra_hyperparams": extra_hyperparams}
    return prepared_dict

def train(args, directory, device, network, loss_fn, acc_fn, loss_fn_ind, acc_fn_ind, train_dataset, train_dataloader, test_datasets, test_dataloaders, optimizer, sharpness_dict, sharpness_quick_dict, random_shock_dict, extra_hyperparams):

    TEST_LABELS = [get_labels(test_dataset) for test_dataset in args.test_datasets]
    LABELS = get_labels(args.train_dataset)
    n_leading_eigenvalues = len(LABELS)
    neigs_to_compute_max = 2 * n_leading_eigenvalues if args.log_top_eigs else 2
    
    descr = f"train {args.arch_id} of {sum(p.numel() for p in network.parameters() if p.requires_grad)} params " +  \
          f"with {args.optimizer_str} on {args.train_dataset} using {args.loss} objective"
    print(descr)
    
    log({"num_parameters": len(torch.nn.utils.parameters_to_vector(network.parameters())),
         "description": descr}, step=None, summary=True)

    projectors = torch.randn(args.nproj, len(torch.nn.utils.parameters_to_vector(network.parameters())))
    iterates = torch.zeros(args.max_iters // args.iterate_freq if args.iterate_freq > 0 else 0, args.nproj)
    iterates_loss = torch.zeros(args.max_iters // args.iterate_freq if args.iterate_freq > 0 else 0)
    iterates_full = torch.zeros(args.max_iters // args.gradient_freq if args.gradient_freq > 0 else 0, args.nproj)
    iterates_bulk = torch.zeros(args.max_iters // args.gradient_freq if args.gradient_freq > 0 else 0, args.nproj)
    iterates_orthogonal = torch.zeros(args.max_iters // args.gradient_freq if args.gradient_freq > 0 else 0, args.nproj)

    ema_model, swa_model, swa_scheduler = None, None, None
    if args.swa:
        assert swa_start >= 0 and swa_start < args.max_iters
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.swa_lr) # CosineAnnealingLR(optimizer, T_max=100)
    else:
        swa_start = -1
    swa = False # FIXME
    if args.ema:
        ema_model = None
        assert args.gamma_ema > 0.0 and args.gamma_ema < 1.0

    train_loss, train_acc = torch.zeros(args.max_iters), torch.zeros(args.max_iters)
    test_loss, test_acc = torch.zeros(len(test_dataloaders), args.max_iters), torch.zeros(len(test_dataloaders), args.max_iters)
    all_evals = []
    eigs = torch.zeros(args.max_iters // args.hessian_freq if args.hessian_freq >= 0 else 1, neigs_to_compute_max)
    eigs_fim = torch.zeros(args.max_iters // args.fisher_freq if args.fisher_freq >= 0 else 1, 1)
    step_eos_e1, step_eos_fim = -1, -1
    all_logits_dict = [prepare_logits(dataset_name, labels, args.gradcam_counter) for dataset_name, labels in zip(args.test_datasets, TEST_LABELS)]

    analysis_fn = None
    analysis_dict = {"arch_id": args.arch_id, "train_dataset_name": args.train_dataset, "test_datasets_names": args.test_datasets, "labels": LABELS, "test_labels": TEST_LABELS, "plot_logits_softmax": False,
                     "gradcam_batch_size": args.gradcam_batch_size, "gradcam_counter": args.gradcam_counter, "gradcam_rescale": args.gradcam_rescale,
                     "selectivity_batch_size": args.selectivity_batch_size, "selectivity_epsilon": args.selectivity_epsilon, "perclass_batch_size": args.perclass_batch_size, 
                     "sparsity_epsilon": args.sparsity_epsilon, "analysis_freq": args.analysis_freq, "skip_analysis_start": args.skip_analysis_start}

    current_iter = 0
    last_time = 0
    for epoch in range(args.max_iters // len(train_dataloader) + 1):
        current_iter, last_time = do_epoch(current_iter, device, network, loss_fn, acc_fn, 
                                train_dataset, train_dataloader, test_datasets, test_dataloaders,
                                args.max_iters, args.train_min_acc, args.train_max_loss, optimizer, swa_model, swa_scheduler, ema_model,
                                args.complex_freq, args.generic_freq, args.gradient_freq, n_leading_eigenvalues, projectors,
                                iterates, iterates_loss, iterates_full, iterates_bulk, iterates_orthogonal, args.iterate_freq, 
                                step_eos_e1, sharpness_dict, args.hessian_freq, step_eos_fim, sharpness_quick_dict, args.fisher_freq, 
                                random_shock_dict, args.perturb_freq, args.reinit_freq, args.analysis_freq, args.skip_analysis_start, 
                                analysis_fn, analysis_dict, all_logits_dict, all_evals, directory, args.save_freq, args.info_freq, extra_hyperparams)
        if current_iter < 0 or current_iter >= args.max_iters:
            break

    print("Ended training at ", last_time)

    after_training = dict()
    # args.iterate_freq = 10
    # iterates = torch.rand(last_time//args.iterate_freq, args.nproj)
    # iterates_loss = torch.rand(last_time//args.iterate_freq)
    # iterates_full= torch.rand(last_time//args.iterate_freq, args.nproj)
    # iterates_bulk = torch.rand(last_time//args.iterate_freq, args.nproj)
    # iterates_orthogonal = iterates_full - iterates_bulk
    if args.iterate_freq != -1:
        torch.save(iterates, "/home/mateuszpyla/stan/sharpness/results/iterates.pt")
        if last_time > args.iterate_freq:
            after_training["viz/iterates"] = visualize_iters(iterates[:last_time//args.iterate_freq], iterates_loss[:last_time//args.iterate_freq])

    if args.gradient_freq != -1:
        torch.save(iterates_full, "/home/mateuszpyla/stan/sharpness/results/iterates_full.pt")
        torch.save(iterates_bulk, "/home/mateuszpyla/stan/sharpness/results/iterates_bulk.pt")
        torch.save(iterates_orthogonal, "/home/mateuszpyla/stan/sharpness/results/iterates_ortho.pt")
        if last_time > args.iterate_freq:
            f, g = visualize_grads(iterates_full[:last_time//args.iterate_freq], iterates_bulk[:last_time//args.iterate_freq], iterates_orthogonal[:last_time//args.iterate_freq], iterates_loss[:last_time//args.iterate_freq])
            after_training["viz/gradients"] = f
            after_training["viz/gradients_traj"] = g

    log(after_training, step=last_time)


    if args.save_end:
        save_checkpoint(network, directory)
        save_files(directory, [("last_time", last_time)])
        #iterates_full, iterates_bulk, iterates_orthogonal # VIZ
        # sharpness_e1, sharpness_fim
