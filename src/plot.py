import torch
import matplotlib.pyplot as plt

from utilities import get_gd_directory

def plot_gd(path_to_save, file_to_save, dataset, arch, loss, opt, lr, eig_freq, seed, beta, delta, start_step):

    gd_directory = get_gd_directory(dataset, arch, loss, opt, lr, eig_freq, seed, beta, delta, start_step)

    train_loss = torch.load(f"{gd_directory}/train_loss_final")
    train_acc = torch.load(f"{gd_directory}/train_acc_final")
    sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]
    eos = torch.load(f"{gd_directory}/eos_time")

    f = plt.figure(figsize=(5, 5), dpi=100)

    plt.subplot(3, 1, 1)
    plt.plot(train_loss)
    plt.title("train loss")
    plt.subplots_adjust(hspace=0)

    plt.subplot(3, 1, 2)
    plt.plot(train_acc)
    plt.title("train accuracy")
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 3)
    plt.scatter(torch.arange(len(sharpness)) * eig_freq, sharpness, s=5)
    plt.axhline(2. / lr, linestyle='dotted')
    plt.axvline(eos)
    plt.title("sharpness")
    plt.xlabel("iteration")

    plt.savefig(f"{path_to_save}/{file_to_save}.png")
    return f


def plot_proj_traj():
    pass