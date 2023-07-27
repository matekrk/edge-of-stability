
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

import torch

from relative_space import transform_space

STEPS = 50

TAKE_FIRST = 100
vgg_dim = 512
DIMS = (TAKE_FIRST, vgg_dim)

NUM_ANCH = TAKE_FIRST//10
IND_ANCH = np.random.choice(np.arange(TAKE_FIRST), NUM_ANCH, replace=False)

def numb_nonzero(t):
    return len((t.reshape(t.shape[0],-1)).any(1).nonzero()) # works for 3d

def p_ce(lr, seed):
    return f"/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_{seed}/ce/gd/lr_{lr}/traj"

def feat_ce(lr, seed, return_steps=False):
    if return_steps:
        feats = torch.load(os.path.join(p_ce(lr, seed), "all.pt"))
        num_steps = numb_nonzero(feats)
        return feats, num_steps
    return torch.load(os.path.join(p_ce(lr, seed), "all.pt"))

def loss_ce(lr, seed):
    return torch.load(os.path.join(p_ce(lr, seed), "losses.pt"))

def normalized_loss_ce(lr, seed):
    return torch.nn.functional.normalize(loss_ce(lr, seed), dim=0)

def take_first_flattened_representations_ce(lr, seed, take_steps=None, take_first=TAKE_FIRST, num_anch=NUM_ANCH, ind_anch=IND_ANCH):
    feat = feat_ce(lr, seed, return_steps=False)
    if take_steps is None:
        take_steps = len(feat)
    feat = feat[:take_steps]
    feat = feat[:,:take_first]
    transf = np.zeros((take_steps, take_first * num_anch))
    for i, X in enumerate(feat):
        p = torch.flatten(transform_space(X, ind_anchors=ind_anch)[0])
        transf[i] = p
    return transf

def avg_seeds_take_first_flattened_representations_ce(lr, seeds, take_first=TAKE_FIRST, num_anch=NUM_ANCH, ind_anch=IND_ANCH, return_loss=True):
    min_steps = 50000
    for seed in seeds:
        _, num_steps = feat_ce(lr, seed, return_steps=True)
        if num_steps < min_steps:
            min_steps = num_steps

    transfs = np.zeros((len(seeds), min_steps, take_first * NUM_ANCH))
    losss = np.zeros((len(seeds), min_steps))
    for i, seed in enumerate(seeds):
        transfs[i] = take_first_flattened_representations_ce(lr, seed, min_steps, take_first, num_anch, ind_anch)
        losss[i] = normalized_loss_ce(lr, seed)[:min_steps]
    if return_loss:
        return np.mean(transfs, axis=0), np.mean(losss, axis=0)
    else:
        return np.mean(transfs, dim=0)

def concat_lrs(lrs, seeds, take_first=TAKE_FIRST, num_anch=NUM_ANCH, ind_anch=IND_ANCH, return_loss=True):
    repr_multi_dim = []
    losses = []
    num_traj_per_lr = []
    for i, lr in enumerate(lrs):
        avg_repr, avg_loss = avg_seeds_take_first_flattened_representations_ce(lr, seeds, take_first, num_anch, ind_anch, return_loss=True)
        repr_multi_dim.append(avg_repr)
        num_traj_per_lr.append(len(avg_repr))
        losses.append(avg_loss)
        assert len(avg_repr) == len(avg_loss)
    concat_repr_multi_dim = np.concatenate(repr_multi_dim)
    concat_losses = np.concatenate(losses)
    assert sum(num_traj_per_lr) == len(concat_repr_multi_dim)
    if return_loss:
        return concat_repr_multi_dim, num_traj_per_lr, concat_losses
    return concat_repr_multi_dim, num_traj_per_lr

def pca_repr(lrs, concat_repr_multi_dim, num_traj_per_lr, losses=None):
    pca = PCA(n_components=2, random_state=0)
    visio = pca.fit_transform(concat_repr_multi_dim)

    dict_repr = {}
    if losses is not None:
        dict_loss = {}
    ind_i = 0
    ind_j = 0
    while ind_i < len(visio):
        dict_repr[lrs[ind_j]] = visio[ind_i:ind_i+num_traj_per_lr[ind_j]]
        if losses is not None:
            dict_loss[lrs[ind_j]] = losses[ind_i:ind_i+num_traj_per_lr[ind_j]]
        ind_i += num_traj_per_lr[ind_j]
        ind_j += 1
    if losses is not None:
        return dict_repr, dict_loss
    return dict_repr

def plot_repr(dict_repr, dict_loss, markers, names, path):

    sets = []
    for k, v in dict_repr.items():
        visio_df = pd.DataFrame(data=v, columns = ['principal component 1', 'principal component 2'])
        visio_df['y_loss'] = dict_loss[k]
        sets.append(visio_df)

    concatenated = pd.concat([sets[i].assign(dataset=names[i]) for i in range(len(dict_repr))])
    concatenated.rename(columns={'dataset': 'method'}, inplace=True)

    fig = plt.figure(figsize=(16,10))
    sns.scatterplot(x="principal component 1", y="principal component 2", data=concatenated,
                    hue='y_loss', style='method', markers=markers, palette=sns.color_palette("hot", as_cmap=True), alpha=0.3)

    plt.savefig(path)

def main():
    lrs = [0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005] #0.001 
    seeds = [10, 11, 12]
    NUM_METHODS = len(lrs)

    my_markers = ["o", "s", "D", "<", "v", "^", ">", "*", "X"] # "P"
    assert len(my_markers) == NUM_METHODS

    names = [f"vgg11-ce-lr{lr}" for lr in lrs]

    concat_repr_multi_dim, num_traj_per_lr, losses = concat_lrs(lrs, seeds, take_first=TAKE_FIRST, num_anch=NUM_ANCH, ind_anch=IND_ANCH, return_loss=True)
    dict_repr, dict_loss = pca_repr(lrs, concat_repr_multi_dim, num_traj_per_lr, losses)

    plot_repr(dict_repr, dict_loss, my_markers, names, "/home/mateusz.pyla/stan/edge-of-stability/figures/fig.png")

if __name__ == "__main__":
    main()