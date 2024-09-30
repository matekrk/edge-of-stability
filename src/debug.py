import os
import numpy as np
import torch

from data_generic import load_dataset, take_first
from utilities import get_loss_and_acc, get_dataloader
from archs import load_architecture

from traker import visualize, trak

"""

models_path = "/home/mateuszpyla/stan/sharpness/results/cifar10/resnet32/mse/sgd/lr_1.0_vanilla/seed_0/freq_1/start_0"
models_path = "/home/mateuszpyla/stan/sharpness/results/cifar10/resnet32/mse/sgd/lr_1.0/seed_1/freq_1/start_0"
os.environ["DATASETS"] = os.path.join(os.path.abspath(os.getcwd()), "data")

dataset = "cifar10"
loss = "ce"
physical_batch_size = 1000
abridged_size = 5000
arch_id = "resnet32"
dynamic = False

max_models = 30

"""

os.environ["DATASETS"] = os.path.join(os.path.abspath(os.getcwd()), "data")



def default_model_mnist():

    device = "cuda"

    dataset = "mnist"
    loss = "ce"
    physical_batch_size = 1000
    abridged_size = 5000
    arch_id = "lenet2"
    dynamic = False

    network = load_architecture(arch_id, dataset, dynamic).to(device)
    path_to_models = "/home/mateuszpyla/stan/sharpness/results/mnist/lenet2/mse/sgd/lr_0.01/seed_10/freq_1/start_0"

    train_dataset, _ = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    train_dataloader = get_dataloader(abridged_train, physical_batch_size)

    return network, path_to_models, train_dataloader

def do_trak_traintrain(path_to_model = None):

    if path_to_model is None:
        network, path_to_models, dataloader = default_model_mnist()

    n_points = len(dataloader.dataset)
    neighbours = 5
    max_models = 10

    # loss_fn, acc_fn = get_loss_and_acc(loss)

    scores = trak(network, path_to_models, dataloader, dataloader, n_points, n_points, max_models=max_models)
    
    top_val = np.zeros((n_points, neighbours))
    bot_val = np.zeros((n_points, neighbours))

    scores -= scores.min()
    scores /= scores.max()
    for i in range(n_points):
        indices = scores[:, i].argsort()
        top_trak_scorers_indices = indices[-neighbours:][::-1]
        bot_trak_scorers_indices = indices[:neighbours][::-1]
        top_trak_scorers_values = scores[:, i][top_trak_scorers_indices]
        bot_trak_scorers_values = scores[:, i][bot_trak_scorers_indices]
        
        top_val[i] = top_trak_scorers_values
        bot_val[i] = bot_trak_scorers_values

    mean, std = top_val.mean(), top_val.std()
    ar_val_means = top_val.mean(axis=1)
    top_outliners = abs(ar_val_means - mean) > std

    mean, std = bot_val.mean(), bot_val.std()
    ar_val_means = bot_val.mean(axis=1)
    bot_outliners = abs(ar_val_means - mean) > std

    return np.logical_or(top_outliners, bot_outliners)
    #trainset = torch.utils.data.Subset(train_dataset, outliners)
    #return trainset
    # visualize(scores, train_dataset, test_dataset, models_path)


# def do_trak_traintest():

#     train_dataset, test_dataset = load_dataset(dataset, loss)
#     abridged_train = take_first(train_dataset, abridged_size)
#     train_dataloader = get_dataloader(train_dataset, physical_batch_size) 
#     test_dataloader = get_dataloader(test_dataset, physical_batch_size)

#     loss_fn, acc_fn = get_loss_and_acc(loss)

#     network = load_architecture(arch_id, dataset, dynamic).cuda()

#     scores = trak(network, models_path, train_dataloader, test_dataloader, len(train_dataloader.dataset), len(test_dataloader.dataset), max_models=max_models)
#     visualize(scores, train_dataset, test_dataset, models_path)

def main():

    do_trak_traintrain()

if __name__ == "__main__":
    main()
