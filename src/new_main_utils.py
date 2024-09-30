import os
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
import umap
from sklearn.decomposition import PCA

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

def load_checkpoint(network, load_file: str):
    if load_file is None: return
    assert os.path.isfile(load_file)
    network.load_state_dict(torch.load(load_file))
    print("Loaded model successfully")

def save_checkpoint(network, save_dir: str, iter: int = None):
    save_file = "final.pt" if iter is None else f"iter_{iter}.pt"
    torch.save(network.state_dict(), os.path.join(save_dir, save_file))

def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")

def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]], step: int = None):
    """Save a bunch of tensors."""
    suffix = f"final_{step}" if step is not None else "final"
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_{suffix}")

def visualize_iters(iters, losses):
    embedding = umap.UMAP(n_neighbors=min(10, len(iters)-1), min_dist=0.1).fit_transform(iters)
    marker_base_size = len(embedding) * (-0.5) + 51 # up to 100 points
    times = [marker_base_size * i for i in range(len(embedding),0,-1)]
    f = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], s=times, c=losses.squeeze(), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Loss value")
    for i in range(len(iters)):
        plt.text(embedding[i, 0], embedding[i, 1], str(i), fontsize=8)
    plt.title('UMAP Embedding with Color Coding')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    # plt.legend()
    plt.savefig("outputs/fig_h.png")
    plt.close()
    return f

def visualize_grads(x, y, z, losses):
    # x = y+z

    n = len(x)
    strengths = [(x+1) for x in range(n)]
    alphas = [0.5 for _ in range(n)]

    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x)
    y_2d = pca.transform(y)
    z_2d = pca.transform(z)
    x_cmap = cm.get_cmap('Blues')
    y_cmap = cm.get_cmap('Greens')
    z_cmap = cm.get_cmap('Greys')

    f, ax = plt.subplots(figsize=(10, 6))

    sc1 = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=strengths, cmap=x_cmap, label='whole gradients', alpha=alphas)
    sc2 = plt.scatter(y_2d[:, 0], y_2d[:, 1], c=strengths, cmap=y_cmap, label='bulk gradients', alpha=alphas)
    sc3 = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=strengths, cmap=z_cmap, label='ortho gradients', alpha=alphas)

    cbar1 = f.colorbar(sc1, ax=ax, label='Whole Gradients Iter')
    cbar2 = f.colorbar(sc2, ax=ax, label='Bulk Gradients Iter')
    cbar3 = f.colorbar(sc3, ax=ax, label='Ortho Gradients Iter')

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA 2D of gradients")
    plt.legend()

    #for i in range(len(x_2d)):
    #    plt.text(x_2d[i, 0], x_2d[i, 1], str(i), fontsize=9)
    #    plt.text(x_2d[i, 0], x_2d[i, 1], str(i), fontsize=9)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA 2D of gradients')
    # plt.colorbar(label="Iter")
    plt.savefig("outputs/fig_f.png")
    plt.close()
    plt.clf()

    ######################

    g = plt.figure(figsize=(10, 6))
    plt.title("Gradients trajectory using 2d")
    losses = losses.detach().cpu().numpy()
    loss_normalized = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
    cmap = cm.get_cmap('Reds', len(losses)) # vmin=np.min(losses), vmax=np.max(losses)
    # norm = plt.Normalize(min(losses), max(losses))
    start_point = np.random.uniform(-10*np.abs(x_2d).max(), 10*np.abs(x_2d).max(), 2)
    start_size, start_marker = 50, "*"
    cur_point = start_point
    for i in range(len(x_2d)):
        x_cur, y_cur = cur_point
        if i == 0:
            plt.arrow(x_cur, y_cur, 0.5*x_2d[i, 0], 0.5*x_2d[i, 1], color='blue', head_width=0.1, head_length=0.1, alpha=0.75, label="whole")
            plt.arrow(x_cur, y_cur, 0.5*y_2d[i, 0], 0.5*y_2d[i, 1], color='green', head_width=0.1, head_length=0.1, alpha=0.5, label="bulk")
            plt.arrow(x_cur, y_cur, 0.5*z_2d[i, 0], 0.5*z_2d[i, 1], color='grey', head_width=0.1, head_length=0.1, alpha=0.5, label="ortho")
            plt.scatter([x_cur], [y_cur], color=cmap(i), alpha=1.0, marker=start_marker, s=start_size)
        else:
            plt.arrow(x_cur, y_cur, 0.5*x_2d[i, 0], 0.5*x_2d[i, 1], color='blue', head_width=0.1, head_length=0.1, alpha=0.75)
            plt.arrow(x_cur, y_cur, 0.5*y_2d[i, 0], 0.5*y_2d[i, 1], color='green', head_width=0.1, head_length=0.1, alpha=0.5)
            plt.arrow(x_cur, y_cur, 0.5*z_2d[i, 0], 0.5*z_2d[i, 1], color='grey', head_width=0.1, head_length=0.1, alpha=0.5)
            plt.scatter([x_cur], [y_cur], color=cmap(i), alpha=1.0, marker="o", s=(-30/len(x_2d) * i + start_size))
        new_cur_point = x_cur + x_2d[i, 0], y_cur + x_2d[i, 1]
        if i < len(x_2d) - 1:
            plt.plot([x_cur, new_cur_point[0]], [y_cur, new_cur_point[1]], color="yellow", alpha=0.25)
        cur_point = new_cur_point
    plt.legend()
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(losses), vmax=np.max(losses))), label="Loss", cax=cax)
    plt.grid(True)
    plt.savefig("outputs/fig_g.png")
    plt.close()
    plt.clf()

    return f, g

