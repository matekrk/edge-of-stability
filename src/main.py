import os
import argparse
import wandb

from data_generic import DATASETS as supported_DATASETS

from gd import train as train_gd
from gd import plot as plot_gd
from flow import train as train_flow
from flow import plot as plot_flow
from aggregate import plot_average, plot_vslr
from plot import plot_proj_traj

if __name__ == "__main__":

    ###
    parser = argparse.ArgumentParser(description="Train using gradient descent / gradient flow")
    # TYPE
    parser.add_argument("type", choices=["train", "test", "aggregate"], help="what to do")
    # TRAIN - GENERAL - MANDATORY
    parser.add_argument("method", choices=["gd", "flow"], help="Which learning algorithm to choose")
    parser.add_argument("dataset", type=str, choices=supported_DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="Which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    # TRAIN - GENERAL - AUXILIARY
    parser.add_argument("--seed", type=int, help="The random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--physical_batch_size", type=int, help="The maximum number of examples on the GPU at once", 
                        default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="Terminate training if the train accuracy ever crosses this value",
                        default=None)
    parser.add_argument("--loss_goal", type=float, help="Terminate training if the train loss ever crosses this value",
                        default=None)
    parser.add_argument("--load_step", type=int, help="At which step start training (for > 0 required loading)",
                        default=0)
    parser.add_argument("--load_file", type=str, help="How to initialize the model", default=None)
    parser.add_argument("--trakked_data", action=argparse.BooleanOptionalAction, help="Whether to apply mask on data")
    parser.add_argument("--trakked_models", type=str, default=None, help="Path dir to models on which select mask to data")
    # TRAIN - GD
    parser.add_argument("--opt", type=str, choices=["sgd", "polyak", "nesterov"], help="Which optimization algorithm to use",
                        default="gd")
    parser.add_argument("--beta", type=float, help="Momentum parameter (used if opt = polyak or nesterov)",
                        default=None, required="opt"=="polyak" or "opt"=="nesterov")
    parser.add_argument("--delta", type=float, help="Dumpening parameter (not used with nesterov)",
                        default=None, required="opt"=="polyak" or "opt"=="nesterov")
    parser.add_argument("--sam", action=argparse.BooleanOptionalAction, help="Sharpness-Aware Minimization", 
                        default=False)
    parser.add_argument("--sam_out", action=argparse.BooleanOptionalAction, help="Sharpness-Aware Minimization for outliners", 
                        default=False)
    parser.add_argument("--sam_rho", type=float, help="Rho parameter for sharpness-Aware Minimization",
                        default=None)
    parser.add_argument("--swa", action=argparse.BooleanOptionalAction, help="Stochastic Weight Averaging", 
                        default=False)
    parser.add_argument("--swa_lr", type=float, help="LR parameter for stochastic weight averaging",
                        default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, help="Exponential Moving Average", 
                        default=False)
    parser.add_argument("--lr", type=float, help="The learning rate", required="method"=="gd")
    parser.add_argument("--max_steps", type=int, help="the maximum number of steps to train for", required="method"=="gd")
    parser.add_argument("--omega-wd-0", type=float, help="regularization penalty for weight norm L0", default=0.0)
    parser.add_argument("--omega-wd-1", type=float, help="regularization penalty for weight norm L1", default=0.0)
    parser.add_argument("--omega-wd-2", type=float, help="regularization penalty for weight norm L2", default=0.0)
    parser.add_argument("--project_gradient", action=argparse.BooleanOptionalAction, help="Project gradient onto Dom", 
                        default=False)
    # TRAIN - FLOW
    parser.add_argument("--tick", type=float, help="The train/test losses/accuracies computed and saved every tick units of time",
                        default=1.0, required="method"=="flow")
    parser.add_argument("--max_time", type=float, help="The maximum time (ODE time, not wall clock time) to train for",
                        default=1000, required="method"=="flow")
    parser.add_argument("--alpha", type=float, help="The Runge-Kutta step size min(alpha/[estimated sharpness],max_step_size)",
                        default=1.0, required="method"=="flow")
    parser.add_argument("--max_step_size", type=float, help="As for alpha: step = min(alpha/[estimated sharpness],max_step_size)",
                        default=999, required="method"=="flow")
    # TRAIN - EOS - ANALYSIS
    parser.add_argument("--neigs", type=int, help="The number of top eigenvalues to compute",
                        default=6)
    parser.add_argument("--eig_freq", type=int, help="The frequency at which compute top Hessian eigenvalues (-1 never)",
                        default=-1)
    parser.add_argument("--nproj", type=int, help="The dimension of random projections",
                        default=0)
    parser.add_argument("--iterate_freq", type=int, help="The frequency at which save random projections of the iterates",
                        default=-1)
    parser.add_argument("--abridged_size", type=int, help="Computing top Hessian eigenvalues, use an abridged size dataset",
                        default=5000)
    # TRAIN - AROUND
    parser.add_argument("--result_dir", type=str, default=os.path.abspath(os.getcwd())) # os.pardir
    parser.add_argument("--path_extra", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=os.path.abspath(os.getcwd()))
    parser.add_argument("--no_wandb", action=argparse.BooleanOptionalAction, help="No logging to wandb", 
                        default=False)
    parser.add_argument("--ManualGroup", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="save model weights at end of training",
                        default=False)
    parser.add_argument("--save_freq", type=int, help="The frequency at which save resuls (-1 never)", 
                        default=-1)
    # TRAIN - EOS - REACT
    parser.add_argument("--minirestart", action=argparse.BooleanOptionalAction, help="Minirestart (not only at EOS)", 
                        default=False)
    parser.add_argument("--minirestart_freq", type=int, help="The frequency at which perform minirestart",
                        default=-1)
    parser.add_argument("--minirestart_start", type=int, help="Start step for minirestarts",
                        default=-1)
    parser.add_argument("--minirestart_tricks_topk", type=int, help="How many top outlining weights to take",
                        default=-1)
    parser.add_argument("--minirestart_tricks", nargs='+', help="What to do at EOS: addneurons, reducenorm, addnoise, backtoinit")
    parser.add_argument("--minirestart_tricks_reducenorm_param", type=float, help="How much reduce the norm in minirestart")
    parser.add_argument("--minirestart_tricks_addnoise_param", type=float, help="How much Gaussian noise add in minirestart")
    parser.add_argument("--minirestart_tricks_layers", nargs='+', default=[], help="On which layers of the net perform minirestart")
    # TRAIN - EOS - OUTLINERS
    parser.add_argument("--eliminate_outliners_data", action=argparse.BooleanOptionalAction, 
                        default=False, help="Eliminate data outliners contributing to EOS")
    parser.add_argument("--eliminate_outliners_data_strategy", type=str, help="How to remove data outliners",
                        choices=["trak", "gradient_vmap", "gradient", "fisher_vmap", "fisher", 
                                 "activation_vmap", "activation", "representation_vmap", "representation"])
    parser.add_argument("--eliminate_outliners_data_gamma", type=float, help="Determine how many std (criterion on outliners)",
                        default=1.0)
    parser.add_argument("--eliminate_outliners_data_lr", type=float, default=0.0, help="The learning rate for outliners")
    parser.add_argument("--eliminate_outliners_features", action=argparse.BooleanOptionalAction, 
                        default=False, help="Eliminate feature outliners contributing to EOS")
    parser.add_argument("--eliminate_outliners_features_strategy", type=str, help="How to remove feature outliners",
                        choices=["gradient_vmap", "gradient", "fisher_vmap", "fisher", 
                                 "activation_vmap", "activation", "representation_vmap", "representation"])
    parser.add_argument("--eliminate_outliners_features_gamma", type=float, help="Determine how many std (criterion on outliners)",
                        default=1.0)
    parser.add_argument("--eliminate_outliners_features_lr", type=float, default=0.0, help="The learning rate for outliners")
    # TRAIN - EOS - RANDOMNESS
    parser.add_argument("--keep_random_layers", type=float, default=0.0, help="With given prob, every layer is kept random")
    parser.add_argument("--keep_random_neurons", type=float, default=0.0, help="With given prob, every neuron is kept random")
    # TRAIN - ANALYSIS
    parser.add_argument("--analysis_freq", type=int, help="The frequency at which do analysis_* (-1 never)", 
                        default=-1)
    parser.add_argument("--analysis_sparsity", action=argparse.BooleanOptionalAction, 
                        default=False, help="Look at the sparisity of the model.")
    parser.add_argument("--analysis_class_selectivity", action=argparse.BooleanOptionalAction, 
                        default=False, help="Class selectivity. Support for: Resnets, VGGs, Convnets")
    parser.add_argument("--analysis_gradcam", action=argparse.BooleanOptionalAction, 
                        default=False, help="Gradcam. Support for: cifar and mnist/fashion CNN-based")
    parser.add_argument("--analysis_mnist", action=argparse.BooleanOptionalAction, 
                        default=False, help="Plotting logits for mnist")
    parser.add_argument("--analysis_fashion", action=argparse.BooleanOptionalAction, 
                        default=False, help="Plotting logits for fashion")
    parser.add_argument("--analysis_coloured_mnist", action=argparse.BooleanOptionalAction, 
                        default=False, help="Plotting logits for coloured mnist")
    parser.add_argument("--analysis_cifar", action=argparse.BooleanOptionalAction, 
                        default=False, help="Plotting logits for cifar-10")
    parser.add_argument("--analysis_trak", action=argparse.BooleanOptionalAction, 
                        default=False, help="Adding trak")
    parser.add_argument("--analysis_perclass", action=argparse.BooleanOptionalAction, 
                        default=False, help="Metrics (acc, prec, recall, f1) per class")
    parser.add_argument("--analysis_eigenvalues", type=int, 
                        default=0, help="How many top eigs to plot")
    # TEST - PLOT
    parser.add_argument("--plot_classic", action=argparse.BooleanOptionalAction, help="Plot as in the paper")
    parser.add_argument("--plot_trajectory", action=argparse.BooleanOptionalAction, help="Plot projected traj concat test reprs")
    parser.add_argument("--plot_trajectory_first_steps", type=int, default=50, help="How many first steps of trajectory to plot")
    parser.add_argument("--plot_trajectory_first_examples", type=int, default=50, help="How many reprs of test set to concat")
    parser.add_argument("--plot_rgb_activations", action=argparse.BooleanOptionalAction, help="Plot activations of RGB images")

    # AGGREGATE
    parser.add_argument("--aggregate_models", nargs='+', help="Which trained results to aggregate (provide paths)")
    # AGGREGATE - AVERAGE OVER SEEDS
    parser.add_argument("--aggregate_plot_seeds", action=argparse.BooleanOptionalAction, help="Plot results averaged over seeds")
    # AGGREGATE - VS LR
    parser.add_argument("--aggregate_plot_vslr", action=argparse.BooleanOptionalAction, help="Plot results varied over LR")

    args = parser.parse_args()
    ###

    # AROUND
    if not "RESULTS" in os.environ:
        os.environ["RESULTS"] = os.path.join(args.result_dir, "results")
    if not "DATASETS" in os.environ:
        os.environ["DATASETS"] = os.path.join(args.dataset_dir, "data")

    f = open("wandb_key.txt", "r")
    wandb_key = f.read()
    wandb.login(key=wandb_key)

    f = open("wandb_entity.txt", "r")
    wandb_entity = f.read()

    f = open("wandb_project.txt", "r")
    wandb_project = f.read()
    wandb_tag = ["train"] if args.type == "train" else []

    wandb_group = args.group_name

    run = wandb.init(project=wandb_project, entity=wandb_entity, config=args, dir=os.environ["RESULTS"], 
                     tags=wandb_tag, group=wandb_group, mode="disabled" if args.no_wandb else None)

    # TRAIN
    if args.type == "train":
        if args.method == "gd":
            train_gd(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
                     loss_goal=args.loss_goal, acc_goal=args.acc_goal, load_step=args.load_step,
                     physical_batch_size=args.physical_batch_size, abridged_size=args.abridged_size,
                     seed=args.seed, path_extra=args.path_extra, save_model=args.save_model, beta=args.beta, delta=args.delta, 
                     sam=args.sam, sam_out=args.sam_out,sam_rho=args.sam_rho, 
                     swa=args.swa, swa_lr=args.swa_lr, ema=args.ema,
                     omega_wd_0=args.omega_wd_0, omega_wd_1=args.omega_wd_1, omega_wd_2=args.omega_wd_2,
                     project_gradient=args.project_gradient,
                     neigs=args.neigs, eig_freq=args.eig_freq, iterate_freq=args.iterate_freq, save_freq=args.save_freq, nproj=args.nproj,
                     minirestart=args.minirestart, minirestart_freq=args.minirestart_freq, minirestart_start=args.minirestart_start,
                     minirestart_tricks=args.minirestart_tricks, minirestart_tricks_layers=args.minirestart_tricks_layers, 
                     minirestart_tricks_topk=args.minirestart_tricks_topk,
                     minirestart_tricks_reducenorm_param=args.minirestart_tricks_reducenorm_param,
                     minirestart_tricks_addnoise_param=args.minirestart_tricks_addnoise_param,
                     eliminate_outliners_data=args.eliminate_outliners_data,
                     eliminate_outliners_data_strategy=args.eliminate_outliners_data_strategy,
                     eliminate_outliners_data_gamma=args.eliminate_outliners_data_gamma,
                     eliminate_outliners_data_lr=args.eliminate_outliners_data_lr,
                     eliminate_outliners_features=args.eliminate_outliners_features,
                     eliminate_outliners_features_strategy=args.eliminate_outliners_features_strategy,
                     eliminate_outliners_features_gamma=args.eliminate_outliners_features_gamma,
                     eliminate_outliners_features_lr=args.eliminate_outliners_features_lr,
                     keep_random_layers=args.keep_random_layers, keep_random_neurons=args.keep_random_neurons, analysis_freq=args.analysis_freq,
                     analysis_sparsity=args.analysis_sparsity, analysis_class_selectivity=args.analysis_class_selectivity, analysis_gradcam=args.analysis_gradcam,
                     analysis_cifar=args.analysis_cifar, analysis_coloured_mnist=args.analysis_coloured_mnist, analysis_mnist=args.analysis_mnist, analysis_fashion = args.analysis_fashion,
                     analysis_trak=args.analysis_trak, analysis_perclass=args.analysis_perclass, analysis_eigenvalues=args.analysis_eigenvalues
                     )
        
            #trajectories = args.plot_trajectory, trajectories_first = args.trajectory_first_steps,
            #rgb_activations = args.plot_rgb_activations,
            #ministart_addneurons = args.minirestart_addneurons, minirestart_reducenorm=args.minirestart_reducenorm, 
            #minirestart_addnoise = args.minirestart_addnoise, minirestart_backtoinit = args.minirestart_backtoinit,
            #eliminate_outliners = args.eliminate_outliners, eliminate_outliners_strategy = args.eliminate_outliners_strategy, 
            #eliminate_outliners_gamma = args.eliminate_outliners_gamma, lr_outliners=args.lr_outliners,
            #eliminate_features=args.eliminate_features, eliminate_features_gamma=args.eliminate_features_gamma, lr_features=args.lr_features

    if args.method == "flow":
        train_flow()

    if args.plot_classic:
        if args.method == "gd":
            plot_gd(os.environ["RESULTS"], dataset = args.dataset, arch = args.arch_id, loss = args.loss, seed = args.seed, 
                    gd_lr = args.lr, gd_eig_freq = args.eig_freq, save=True)
        if args.method == "flow":
            plot_flow(os.environ["RESULTS"], dataset = args.dataset, arch = args.arch_id, loss = args.loss, seed = args.seed,
                      flow_tick=args.tick, save=True)

    if args.plot_trajectory:
        traj_directory = os.path.join(os.environ["RESULTS"], args.dataset, args.arch_id, f"seed_{args.seed}", args.loss, args.method, f"lr_{args.lr}", "traj")
        plot_proj_traj() # (os.path.join(traj_directory, "all.pt"), os.path.join(traj_directory, "losses.pt"), traj_directory, take_first=args.plot_trajectory_first_examples, save=True)

    # AGGREGATE
    if args.type == "aggregate":
        if args.aggregate_plot_seeds:
            plot_average()
        if args.aggregate_plot_vslr:
            plot_vslr()
