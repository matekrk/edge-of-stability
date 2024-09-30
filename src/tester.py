import os
from matplotlib import pyplot as plt
import torch

from data_fashion import FashionMNIST, FASHION_LABELS
from data_generic import load_dataset
from utilities import iterate_dataset

def main():

    dir_path = os.path.abspath(os.getcwd())

    if not "RESULTS" in os.environ:
        os.environ["RESULTS"] = os.path.join(dir_path, "results")
    if not "DATASETS" in os.environ:
        os.environ["DATASETS"] = os.path.join(dir_path, "data")

    dataset_name = "fashion-5k"
    loss = "mse"
    train_dataset, test_dataset = load_dataset(dataset_name, loss)
    physical_batch_size = 1000


    for batch in iterate_dataset(train_dataset, physical_batch_size):
        (X, y) = batch
        X, y = X.cuda(), y.cuda()

        image = X[0].squeeze()
        label = y[0]
        sample_idx= torch.randint((label), size = (1,)).item()
        plt.title(f"Label: {label} - {FASHION_LABELS[sample_idx]}")
        plt.imshow(image, cmap="gray")
        plt.show()
        print(f"Labels batch shape: {y.size()}")

if __name__ == "__main__":
    main()
