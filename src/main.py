import os
import argparse
import wandb

from data import DATASETS
from plot import plot_gd, plot_flow, plot_proj_traj

from gd import main as main_gd
from flow import main as main_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent / gradient flow.")
    # GENERAL
    parser.add_argument("method", choices=["gd", "flow"], help="which method to choose")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--result_dir", type=str, default=os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    parser.add_argument("--dataset_dir", type=str, default=os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    parser.add_argument("--minirestart_addneurons", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_reducenorm", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_addnoise", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_backtoinit", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")

    # GD
    #parser.add_argument("--gd", type=bool, default=False, help="if 'true', gradient descent")
    parser.add_argument("--lr", type=float, help="the learning rate", required='method'=="gd")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for", required='method'=="gd")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")

    # FLOW
    #parser.add_argument("--flow", type=bool, default=False, help="if 'true', flow gradient")
    parser.add_argument("--tick", type=float, default=1.0,
                        help="the train / test losses and accuracies will be computed and saved every tick units of time",
                        required='method'=="flow")
    parser.add_argument("--max_time", type=float, default=1000,
                        help="the maximum time (ODE time, not wall clock time) to train for", required='method'=="flow")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help=" the Runge-Kutta step size is min(alpha / [estimated sharpness], max_step_size).")
    parser.add_argument("--max_step_size", type=float, default=999,
                        help=" the Runge-Kutta step size is min(alpha / [estimated sharpness], max_step_size)")

    # PLOT
    parser.add_argument("--plot_classic", action=argparse.BooleanOptionalAction, help="plot as in the paper")
    parser.add_argument("--plot_trajectory", action=argparse.BooleanOptionalAction, help="plot projected trajectory by concat of test feat repr")
    parser.add_argument("--trajectory_first_steps", type=int, default=50, help="how many first steps take into plotting")
    parser.add_argument("--trajectory_first_examples", type=int, default=50, help="how many first images from test set to concat")

    args = parser.parse_args()

    os.environ["RESULTS"] = os.path.join(args.result_dir, "rez")
    os.environ["DATASETS"] = os.path.join(args.dataset_dir, "data")

    f = open("wandb_key.txt", "r")
    wandb_key = f.read()
    f = open("wandb_entity.txt", "r")
    wandb_entity = f.read()
    f = open("wandb_project.txt", "r")
    wandb_project = f.read()
    wandb.login(key=wandb_key)
    run = wandb.init(project=wandb_project, entity=wandb_entity, config=args, dir=os.environ["RESULTS"])

    if args.method == "gd":
        main_gd(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
            neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
            iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
            nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size, seed=args.seed,
            trajectories=args.plot_trajectory, trajectories_first=args.trajectory_first_steps,
            ministart_addneurons = args.minirestart_addneurons, minirestart_reducenorm=args.minirestart_reducenorm, 
            minirestart_addnoise = args.minirestart_addnoise, minirestart_backtoinit = args.minirestart_backtoinit)

    if args.method == "flow":
        main_flow(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, max_time=args.max_time, tick=args.tick,
            neigs=args.neigs, physical_batch_size=args.physical_batch_size, abridged_size=args.abridged_size,
            eig_freq=args.eig_freq, iterate_freq=args.iterate_freq, save_freq=args.save_freq, nproj=args.nproj,
            loss_goal=args.loss_goal, acc_goal=args.acc_goal, seed=args.seed)

    if args.plot_classic:
        if args.method == "gd":
            plot_gd(os.environ["RESULTS"], dataset = args.dataset, arch = args.arch_id, loss = args.loss, seed = args.seed, 
                    gd_lr = args.lr, gd_eig_freq = args.eig_freq, save=True)
        if args.method == "flow":
            plot_flow(os.environ["RESULTS"], dataset = args.dataset, arch = args.arch_id, loss = args.loss, seed = args.seed,
                      flow_tick=args.tick, save=True)

    if args.plot_trajectory:
        traj_directory = os.path.join(os.environ["RESULTS"], args.dataset, args.arch_id, f"seed_{args.seed}", args.loss, args.method, f"lr_{args.lr}", "traj")
        plot_proj_traj(os.path.join(traj_directory, "all.pt"), os.path.join(traj_directory, "losses.pt"), traj_directory, take_first=args.trajectory_first_examples, save=True)
