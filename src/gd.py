from os import makedirs, path
import wandb
import torch
from torch.nn.utils import parameters_to_vector

import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, save_features, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, compute_space, calculate_trajectory_point, obtained_eos
from utilities import compute_logits, reduced_batch
from relative_space import transform_space
from plot import plot_pca_space, plot_proj_traj
from data import load_dataset, take_first, DATASETS


def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0,
         trajectories: bool = False, trajectories_first: int = -1,
         ministart_addneurons: bool = False, minirestart_reducenorm: bool = False, 
         minirestart_addnoise: bool = False, minirestart_backtoinit: bool = False, 
         eliminate_outliners: bool = False, eliminate_outliners_gamma: float = 1.0):
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    if trajectories:
        traj_directory = path.join(directory, "traj")
        makedirs(traj_directory, exist_ok=True)
        assert trajectories_first > 0

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)

    exploding = -1

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    train_outliners = torch.zeros(max_steps)
    train_outliners_histogram = torch.zeros((max_steps, 100))
    sky_activations, red_activations, green_activations = torch.zeros((max_steps, 10)), torch.zeros((max_steps, 10)), torch.zeros((max_steps, 10))
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("test/*", step_metric="test/step")

    if trajectories:
        dim_feature = network.get_representation_dim()
        trajectories_full = torch.zeros((trajectories_first//eig_freq, 1000, dim_feature))
        y_loss = torch.zeros(trajectories_first//eig_freq)
    #perm = torch.randperm(1000)
    #ind_anchors = perm[:num_anchors]

    for step in range(0, max_steps):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        (test_loss[step], test_acc[step]), features = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size, return_features=True)
        wandb.log({'train/step': step, 'train/acc': train_acc[step], 'train/loss': train_loss[step]})
        wandb.log({'test/step': step, 'test/acc': test_acc[step], 'test/loss': test_loss[step]})

        sky_activations[step], red_activations[step], green_activations[step] = compute_logits(network, [loss_fn, acc_fn])
        for i_class in range(len(sky_activations[step])):
            wandb.log({'artificial/step': step, f'artificial/sky/{i_class}': sky_activations[step, i_class], f'artificial/red/{i_class}': red_activations[step, i_class], f'artificial/green/{i_class}': green_activations[step, i_class]})

        if eig_freq != -1 and step % eig_freq == 0:
            if trajectories and step < trajectories_first:
                
                save_features(features, step, traj_directory)
                trajectories_full[step//eig_freq] = features
                y_loss[step//eig_freq] = test_loss[step]
            
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            """
            if obtained_eos(eigs, lr):
                num_network_layers = network.n_layers 
                active_layers = []
                if ministart_addneurons:
                    network.addneurons(active_layers)
                if minirestart_addnoise:
                    network.addnoise(active_layers)
                if minirestart_reducenorm:
                    network.reducenorm(active_layers)
                if minirestart_backtoinit:
                    network.restartweight(active_layers)
            """

            wandb.log({'train/step': step, 'train/e1': eigs[step // eig_freq, 0], 'train/e2': eigs[step // eig_freq, 1]})

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break

        if exploding > -1 or (step > 3 and train_loss[step] > 2 * train_loss[step-1]):
            print("exploding")
            save_features(features, step, traj_directory)
            exploding = step
            if train_loss[step] > 5 * train_loss[step-4]:
                print("stabilized")
                exploding = -1

        optimizer.zero_grad()
        to_cal_mean = []
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            if eliminate_outliners:
                X_r, y_r = reduced_batch(network, loss_fn, X.cuda(), y.cuda(), strategy="norm", gamma=eliminate_outliners_gamma)
            else:
                X_r, y_r = X.cuda(), y.cuda()
            to_cal_mean.append(len(X_r)/len(X))
            loss = loss_fn(network(X_r), y_r) / len(train_dataset)
            loss.backward()
        optimizer.step()
        train_outliners[step] = sum(to_cal_mean)/len(to_cal_mean)

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1]),
                      ("train_outliners", train_outliners[:step + 1]), ("train_outliners_histogram", train_outliners_histogram[:step + 1]),
                      ("sky_activations", sky_activations[:step + 1]), ("red_activations", red_activations[:step + 1]), ("green_activations", green_activations[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")

    if trajectories:
        torch.save(trajectories_full, path.join(traj_directory, "all.pt"))
        torch.save(y_loss, path.join(traj_directory, "losses.pt"))
    
    print("***DONE***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
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
                        help="save model weights at end of training")
    parser.add_argument("--trajectories", action=argparse.BooleanOptionalAction,
                        help="plot projected trajectory by concat of test feat repr")
    parser.add_argument("--trajectories_first", type=int, default=50,
                        help="how many first steps take into plotting")
    parser.add_argument("--minirestart_addneurons", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_reducenorm", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_addnoise", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--minirestart_backtoinit", action=argparse.BooleanOptionalAction, help="Minirestart at EOS.")
    parser.add_argument("--eliminate_outliners", action=argparse.BooleanOptionalAction, help="Eliminate outliners contributing to EOS.")
    parser.add_argument("--eliminate_outliners_gamma", type=float, default=1.0,
                        help="how many std to remove when computing the criterion on outliners.")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, trajectories=args.trajectories, trajectories_first=args.trajectories_first,
         ministart_addneurons = args.minirestart_addneurons, minirestart_reducenorm=args.minirestart_reducenorm, 
         minirestart_addnoise = args.minirestart_addnoise, minirestart_backtoinit = args.minirestart_backtoinit,
         eliminate_outliners=args.eliminate_outliners, eliminate_outliners_gamma=args.eliminate_outliners_gamma)
