import torch
import matplotlib.pyplot as plt
from os import environ

from utilities import get_gd_directory

dataset = "mnist-5k"
arch_id = "lenet"
loss = "mse"
opt = "gd"
lr = 0.1
eig_freq = 10
seed = 1

gd_directory = get_gd_directory(dataset, arch_id, loss, opt, lr, eig_freq, seed)
# gd_directory = f"{environ['RESULTS']}/{dataset}/{arch}/seed_{seed}/{loss}/gd/lr_{gd_lr}"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]
gd_eos = torch.load(f"{gd_directory}/eos_point_final")

plt.figure(figsize=(5, 5), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(gd_train_loss)
plt.title("train loss")
plt.subplots_adjust(hspace=0)

plt.subplot(3, 1, 2)
plt.plot(gd_train_acc)
plt.title("train accuracy")
plt.subplots_adjust(hspace=0.5)

plt.subplot(3, 1, 3)
plt.scatter(torch.arange(len(gd_sharpness)) * eig_freq, gd_sharpness, s=5)
plt.axhline(2. / lr, linestyle='dotted')
plt.axvline(gd_eos)
plt.title("sharpness")
plt.xlabel("iteration")

plt.savefig("results/mnist_cur.png")
