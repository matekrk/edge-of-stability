import wandb
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import numpy as np
import torch

from relative_space import transform_space

def plot_pca_space(x, y, path, i="-1", new_dim=2, save=True):

    if len(x.shape) == 3:
        x = np.reshape(x, (-1, 512))
        y = y.flatten()

    pca = PCA(n_components=new_dim)
    x_transformed = pca.fit_transform(x)

    print(x_transformed.shape, y.shape)

    principal_cifar_Df = pd.DataFrame(data = x_transformed
             , columns = ['principal component 1', 'principal component 2'])
    principal_cifar_Df['y'] = y
    
    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=principal_cifar_Df,
        legend="full",
        alpha=0.3
    )

    wandb.log({"representation_pca": wandb.Image(plt)})
    if save:
        plt.savefig(os.path.join(path, f"pca_rel_{i}.png"))

def plot_proj_traj(trajectory, y_loss, directory, take_first=50, save=True):

    if isinstance(trajectory, str):
        trajectory = torch.load(trajectory)
    if isinstance(y_loss, str):
        y_loss = torch.load(y_loss)

    print(trajectory.shape, y_loss.shape)

    feat_dim = trajectory[0].shape[1]
    DIMS = (take_first, feat_dim)
    assert len(trajectory) == len(y_loss)
    for el in trajectory:
        assert el.shape == (1000, DIMS[1])


    TOTAL = DIMS[0] # DIMS[0] * DIMS[1]
    NUM_ANCH = TOTAL//10
    ind_anchors = np.random.choice(np.arange(TOTAL), NUM_ANCH, replace=False)

    transformed = np.zeros((len(trajectory), DIMS[0] * NUM_ANCH))
    
    for i, X in enumerate(trajectory):
        X = X[:DIMS[0], :]
        transformed[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
        

    pca = PCA(n_components=2, random_state=0)
    visio = pca.fit_transform(transformed)

    principal_cifar_Df = pd.DataFrame(data = visio
             , columns = ['principal component 1', 'principal component 2'])
    principal_cifar_Df['y_loss'] = y_loss.numpy()
    
    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue="y_loss",
        palette=sns.color_palette("crest", as_cmap=True),
        data=principal_cifar_Df,
        legend="brief",
        # legend="full",
        alpha=0.3
    )

    wandb.log({"trajectory_loss": wandb.Image(plt)})

    if save:
        plt.savefig(os.path.join(directory, f"trajectory.png"))

def plot_gd(path_results, dataset, arch, seed, loss, gd_lr, gd_eig_freq, save = False):

    gd_directory = f"{path_results}/{dataset}/{arch}/seed_{seed}/{loss}/gd/lr_{gd_lr}"

    gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
    gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
    gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]

    fig = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(gd_train_loss)
    plt.title("train loss")

    plt.subplot(3, 1, 2)
    plt.plot(gd_train_acc)
    plt.title("train accuracy")

    plt.subplot(3, 1, 3)
    plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
    plt.axhline(2. / gd_lr, linestyle='dotted')
    plt.title("sharpness")
    plt.xlabel("iteration")

    plt.tight_layout()

    wandb.log({"all": wandb.Image(plt)})

    if save:
        plt.savefig(os.path.join(gd_directory, "gd.jpg"))


def plot_flow(path_results, dataset, arch, seed, loss, flow_tick = 1.0, flow_eig_freq = 1, save=False):


    flow_directory = f"{path_results}/{dataset}/{arch}/seed_{seed}/{loss}/flow/tick_{flow_tick}"

    flow_train_loss = torch.load(f"{flow_directory}/train_loss_final")
    flow_train_acc = torch.load(f"{flow_directory}/train_acc_final")
    flow_sharpness = torch.load(f"{flow_directory}/eigs_final")[:, 0]

    plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(torch.arange(len(flow_train_loss)) * flow_tick, flow_train_loss)
    plt.title("train loss")

    plt.subplot(3, 1, 2)
    plt.plot(torch.arange(len(flow_train_acc)) * flow_tick, flow_train_acc)
    plt.title("train accuracy")

    plt.subplot(3, 1, 3)
    plt.scatter(torch.arange(len(flow_sharpness)) * flow_tick * flow_eig_freq, flow_sharpness, s=5)
    plt.title("sharpness")
    plt.xlabel("time")

    plt.show()

    if save:
        plt.savefig(os.path.join(flow_directory, "flow.jpg"))

"""

def main(mode="gd"):
    
    # "/home/mateusz.pyla/stan/plots_rez"

    path_results = "/home/mateusz.pyla/stan/rez_eos"

    dataset = "cifar10-5k"
    arch = "vgg11" # "fc-tanh"
    loss = "mse"
    seed = 0
    lr = 0.01
    tick = 1.0

    # path_results = os.path.join(path_results, dataset, arch, f"seed_{seed}", loss)

    assert mode in ["gd", "gdflow"] #TODO: sgd
    if mode == "gd":
        plot_gd(path_results = path_results, dataset = dataset, arch = arch, loss = loss, seed = seed, gd_lr = lr, gd_eig_freq = 100, save=True)
    else:
        plot_flow(path_results = path_results, dataset = dataset, arch = arch, loss = loss, seed = seed, tick = tick, save=True)

if __name__ == "__main__":
    mode="gd"
    main(mode=mode)
"""
