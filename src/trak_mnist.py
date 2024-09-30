import os

from data_generic import load_dataset, take_first
from utilities import get_loss_and_acc, get_dataloader
from archs import load_architecture

from traker import visualize, trak


models_path_1 = "/home/mateuszpyla/stan/sharpness/results/mnist/lenet/mse/sgd/lr_0.5/seed_12/freq_1/start_0"
models_path_2 = "/home/mateuszpyla/stan/sharpness/results/mnist/lenet/mse/sgd/lr_0.1/seed_12/freq_1/start_0"

models_path_21 = "/home/mateuszpyla/stan/sharpness/results/mnist/lenet2/mse/sgd/lr_0.5/seed_12/freq_1/start_0"
models_path_22 = "/home/mateuszpyla/stan/sharpness/results/mnist/lenet2/mse/sgd/lr_0.1/seed_12/freq_1/start_0"

models_path = models_path_21

os.environ["DATASETS"] = os.path.join(os.path.abspath(os.getcwd()), "data")

dataset = "mnist"
loss = "ce"
physical_batch_size = 1000
abridged_size = 5000
arch_id = "lenet2"
dynamic = False

max_models = 20

def main():

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)
    train_dataloader = get_dataloader(train_dataset, physical_batch_size) 
    test_dataloader = get_dataloader(test_dataset, physical_batch_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    network = load_architecture(arch_id, dataset, dynamic).cuda()

    scores = trak(network, models_path, train_dataloader, test_dataloader, len(train_dataloader.dataset), len(test_dataloader.dataset), max_models=max_models)
    visualize(scores, train_dataset, test_dataset, models_path)

if __name__ == "__main__":
    main()
