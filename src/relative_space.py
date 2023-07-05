import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

def transform_space(x, num_anchors=None, ind_anchors=None):
    if ind_anchors is None:
        N, dim = x.shape
        if num_anchors is None:
            num_anchors = dim // 8
        ind_anchors = np.random.choice(np.arange(N), num_anchors, replace=False)
    X_transformed = relative_projection(x, x[ind_anchors])
    return X_transformed, ind_anchors

def relative_projection(x, anchors):
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)

    return torch.einsum("nd, ad -> na", x, anchors)

def demo_test():
    x_1 = torch.tensor([1.5, -1.0])
    x_2 = torch.tensor([-1.5, 2.0])
    x_3 = torch.tensor([-1.5, 0.5])
    x_4 = torch.tensor([0.0, 1.0])

    y_1 = torch.tensor([-0.5, 1.5])
    y_2 = torch.tensor([1.5, -0.5])

    xs = torch.stack([x_1, x_2, x_3, x_4])
    ys = torch.stack([y_1, y_2])

    xs_rel = relative_projection(xs, ys)
    ys_rel = relative_projection(ys, ys) # assert 1 in some dimension

    normz_xs = F.normalize(xs, p=2, dim=-1)
    normz_ys = F.normalize(ys, p=2, dim=-1)
    

    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Demo representation')
    left = -2
    right = 2
    xs_n = xs.numpy()
    ys_n = ys.numpy()
    axs[0].scatter(xs_n[:, 0], xs_n[:, 1], c='black')
    for i in range(len(xs_n)):
        axs[0].text(xs_n[i, 0], xs_n[i, 1], f"{i}")
    axs[0].scatter(ys_n[0, 0], ys_n[0, 1], c='red')
    axs[0].scatter(ys_n[1, 0], ys_n[1, 1], c='blue')
    axs[0].set_xlim(left, right)
    axs[0].set_ylim(left, right)
    axs[0].axis('equal')

    row_sums = np.linalg.norm(xs_n, axis=1)
    xs_norm_n = xs_n / row_sums[:, np.newaxis]
    row_sums = np.linalg.norm(ys_n, axis=1)
    ys_norm_n = ys_n / row_sums[:, np.newaxis]
    axs[1].scatter(xs_norm_n[:, 0], xs_norm_n[:, 1], c='black')
    for i in range(len(xs_norm_n)):
        axs[1].text(xs_norm_n[i, 0], xs_norm_n[i, 1], f"{i}", va='bottom', ha='center')
    axs[1].scatter(ys_norm_n[0, 0], ys_norm_n[0, 1], c='red')
    axs[1].scatter(ys_norm_n[1, 0], ys_norm_n[1, 1], c='blue')
    axs[1].set_xlim(left, right)
    axs[1].set_ylim(left, right)
    axs[1].axis('equal')

    xs_rel_n = xs_rel.numpy()
    ys_rel_n = ys_rel.numpy()
    axs[2].scatter(xs_rel_n[:, 0], xs_rel_n[:, 1], c='black')
    for i in range(len(xs_rel_n)):
        axs[2].text(xs_rel_n[i, 0], xs_rel_n[i, 1], f"{i}", va='bottom', ha='center')
    axs[2].scatter(ys_rel_n[0, 0], ys_rel_n[0, 1], c='red')
    axs[2].scatter(ys_rel_n[1, 0], ys_rel_n[1, 1], c='blue')
    axs[2].set_xlim(left, right)
    axs[2].set_ylim(left, right)
    axs[2].axis('equal')

    plt.savefig("/home/mateusz.pyla/stan/plots_rez/rel.png")

    print("xs", xs)
    print("ys", ys)
    print("normalized xs", normz_xs)
    for i in range(len(xs_n)):
        print(np.linalg.norm(xs_n[i]), np.linalg.norm(xs_norm_n[i]), np.linalg.norm(normz_xs[i].numpy()))
    print("normalized ys", normz_ys)
    for i in range(len(ys_n)):
        print(np.linalg.norm(ys_n[i]), np.linalg.norm(ys_norm_n[i]), np.linalg.norm(normz_ys[i].numpy()))
    print("relative xs", xs_rel)
    for i in range(len(xs_rel_n)):
        print(np.linalg.norm(xs_rel_n[i]))
    # print("first_sample", xs_rel_n[0] - ys_n[0], xs_rel_n[0] - ys_n[1])
    print("first sample", np.clip(np.dot(xs_norm_n[0], ys_norm_n[0]), -1.0, 1.0), np.clip(np.dot(xs_norm_n[0], ys_norm_n[1]), -1.0, 1.0))
    print("second sample", np.clip(np.dot(xs_norm_n[1], ys_norm_n[0]), -1.0, 1.0), np.clip(np.dot(xs_norm_n[1], ys_norm_n[1]), -1.0, 1.0))
    print("third sample", np.clip(np.dot(xs_norm_n[2], ys_norm_n[0]), -1.0, 1.0), np.clip(np.dot(xs_norm_n[2], ys_norm_n[1]), -1.0, 1.0))
    print("forth sample", np.clip(np.dot(xs_norm_n[3], ys_norm_n[0]), -1.0, 1.0), np.clip(np.dot(xs_norm_n[3], ys_norm_n[1]), -1.0, 1.0))

"""
def main():
    demo_test()

if __name__ == "__main__":
    main()
"""

