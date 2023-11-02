import torch
import matplotlib.pyplot as plt
from os import environ

dataset = "cifar10-5k"
if True:
    # arch = "fc-tanh"
    arch = "cnn-relu"
    loss = "mse"
    gd_lr = 0.01
    gd_eig_freq = 100
    seed = 3

if False:
    arch = "resnet32"
    loss = "mse"
    gd_lr = 0.3
    gd_eig_freq = 100
    seed = 4


name = f"{dataset}/{arch}/seed_{seed}/{loss}/gd/lr_{gd_lr}"

gd_directory = f"{environ['RESULTS']}/{name}"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]
gd_train_outliners = torch.load(f"{gd_directory}/train_outliners_final")
gd_sky_activations = torch.load(f"{gd_directory}/sky_activations_final")
gd_red_activations = torch.load(f"{gd_directory}/red_activations_final")
gd_green_activations = torch.load(f"{gd_directory}/green_activations_final")

plt.figure(figsize=(20, 20), dpi=100)

total = 2 + 1 + 1 + 3 # train loss/acc ; sharpness ; outliners ; activations

plt.subplot(total, 1, 1)
plt.plot(gd_train_loss)
plt.title("train loss")

plt.subplot(total, 1, 2)
plt.plot(gd_train_acc)
plt.title("train accuracy")

plt.subplot(total, 1, 3)
plt.scatter(torch.arange(len(gd_sharpness)) * gd_eig_freq, gd_sharpness, s=5)
plt.axhline(2. / gd_lr, linestyle='dotted')
plt.title("sharpness")
plt.xlabel("iteration")

plt.subplot(total, 1, 4)
plt.plot(gd_train_outliners)
# plt.plot((1000 - gd_train_outliners)/1000)
plt.title("train outliners ratio")

# class assignment
classes = {0: "airplane",
           1: "automobile",
           2: "bird",
           3: "cat",
           4: "deer",
           5: "dog",
           6: "frog",
           7: "horse",
           8: "ship",
           9: "truck"}

plt.subplot(total, 1, 5)
for i_class, n_class in classes.items():
    plt.plot(gd_sky_activations[:, i_class], label=n_class)
plt.legend(fontsize = 'x-small')
plt.title("sky activation logit")

plt.subplot(total, 1, 6)
for i_class, n_class in classes.items():
    plt.plot(gd_red_activations[:, i_class], label=n_class)
plt.legend(fontsize = 'x-small')
plt.title("red activation logit")

plt.subplot(total, 1, 7)
for i_class, n_class in classes.items():
    plt.plot(gd_green_activations[:, i_class], label=n_class)
plt.legend(fontsize = 'x-small')
plt.title("green activation logit")

#for i_class in range(10):
#    plt.subplot(total, 1, 5+i_class)
#    plt.plot(gd_sky_activations[:, i_class])
#    plt.title(f"sky activation logit {i_class}")

plt.savefig(f"figures/outliners/{name.replace('/', '_')}_now.png")