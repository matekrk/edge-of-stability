import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import optim
import wandb

class SimpleNet(nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc(x)
    
class LeNet(nn.Module):

    def __init__(self, n_classes = 10):
        super(LeNet, self).__init__()

        self.content = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Conv2d(6, 16, 5),
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_classes)
        )
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, n_classes)

    def forward(self, x, return_features = False):
        return self.content(x)


def power_iteration_efficient(model, dataloader, criterion, num_iterations=10, n=10):
    device = next(model.parameters()).device
    vec = torch.randn(sum(p.numel() for p in model.parameters()), device=device)
    vec = vec / torch.norm(vec)

    for _ in range(num_iterations):
        Hv = torch.zeros_like(vec)
        for batch_id, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            grads = grad(loss, model.parameters(), create_graph=True, retain_graph=True)
            grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])

            Hv_contrib = compute_hessian_vector_product(grad_vec, model.parameters(), vec)
            Hv += Hv_contrib

            if batch_id > n:
              break
        Hv /= n
        lambda_ = torch.dot(Hv, vec)
        vec = Hv / torch.norm(Hv)
    return lambda_, vec

def compute_hessian_vector_product(grad_vec, parameters, vec):
    jacobian_vec_product = grad(grad_vec, parameters, grad_outputs=vec, retain_graph=True)
    return torch.cat([jvp.contiguous().view(-1) for jvp in jacobian_vec_product])


def do_perturb_and_shrink(model, alpha, noise_std, threshold_norm_level=0.8):
    with torch.no_grad():
        for p in model.parameters():

            if p.grad is not None:
                gradients = p.grad.flatten()
                threshold = torch.quantile(gradients, threshold_norm_level)
                mask = (gradients >= threshold).reshape_as(p.grad)

                p.data[mask] = alpha * p.data[mask] + (1 - alpha) * torch.randn_like(p.data[mask]) * noise_std

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, device=torch.device("cpu"),
                perturb_and_shrink=False, every_batch=100, perturb_threshold=1.0, alpha_perturb=0.9, noise_std=0.01, threshold_norm_level=0.8):
    train_accuracies = []
    test_accuracies = []
    top_eigenvalues = []
    train_losses = []

    print("Start training")

    for epoch in range(epochs):

        train_accuracy, train_loss, losses, eigens = do_epoch(epoch, model, train_loader, criterion, optimizer, device, every_batch, perturb_and_shrink,
                                                             perturb_threshold, alpha_perturb, noise_std, threshold_norm_level)
        train_accuracies.append(train_accuracy)
        top_eigenvalues.extend(eigens)

        step = (epoch+1) * len(train_loader)
        test_accuracy = test_model(step, model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
        train_losses.extend(losses)

        print(f'Epoch: {epoch} Training Loss: {train_loss} '
              f'Train Accuracy: {train_accuracy} Test Accuracy: {test_accuracy}')

        if test_accuracy > 0.85:
            break
        
    print(f"Ended training at {epoch}")

    return top_eigenvalues, train_losses, train_accuracies, test_accuracies

def do_epoch(epoch_idx, model, train_loader, criterion, optimizer, device, every_batch=100, 
             perturb_and_shrink=False, perturb_threshold=1.0, alpha_perturb=0.9, noise_std=0.01, threshold_norm_level=0.8):
    model.train()
    total_loss, correct, total = 0, 0, 0
    running_loss = []
    losses = []
    eigens = []
    exp_eos = 2 / optimizer.param_groups[-1]['lr']
    for batch_idx, (data, target) in enumerate(train_loader):

        step = epoch_idx * len(train_loader) + batch_idx

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        running_loss.append(loss.item())

        if every_batch and (batch_idx + 1) % every_batch == 0:
            eigen = do_action(model, train_loader, criterion, perturb_and_shrink, exp_eos, perturb_threshold, alpha_perturb, noise_std, threshold_norm_level)
            eigens.append(eigen.item())
            losses.append(np.mean(running_loss))
            print(f"Batch {batch_idx+1}: Top Eigenvalue = {eigen.item()}")
            wandb.log({"train/e": eigen.item(), "train/e_scaled": eigen.item() / exp_eos}, step=step)

        wandb.log({"train/running_loss": np.mean(running_loss), "train/running_accuracy": correct / total}, step=step)

    wandb.log({"train/loss": total_loss / len(train_loader), "train/accuracy": correct / total}, step=step)
    return correct / total, total_loss / len(train_loader), losses, eigens

def do_action(model, train_loader, criterion, perturb_and_shrink, exp_eos, perturb_threshold=1.0, alpha_perturb=0.9, noise_std=0.01, threshold_norm_level=0.8):

    print("Computing Hv")
    lambda_, _ = power_iteration_efficient(model, train_loader, criterion)

    if perturb_and_shrink and lambda_.item() > perturb_threshold * exp_eos:
        print("Peturb and shrink")
        do_perturb_and_shrink(model, alpha=alpha_perturb, noise_std=noise_std, threshold_norm_level=threshold_norm_level)

    return lambda_
    

def test_model(step_idx, model, test_loader, criterion, device):
    model.eval()
    test_loss, correct = 0, 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += 1

    test_loss /= n
    accuracy = correct / len(test_loader.dataset)
    step = step_idx
    wandb.log({"test/loss": test_loss, "test/accuracy": accuracy}, step=step)
    return accuracy

def repeat_experiment(train_loader, test_loader, device, seeds, lrs, every_batches, threshold_levels, alpha_perturbs, noise_stds, threshold_norm_levels):

    print(f"Training {len(seeds)}x{len(lrs)}x[1+{len(every_batches)}x{len(threshold_levels)}x{len(alpha_perturbs)}x{len(noise_stds)}x{len(threshold_norm_levels)}] models")

    R = {}
    epochs = 200

    hyper_dict = {
        "perturb_and_shrink": None,
        "lr": None,
        "every_batch": None,
        "threshold_level": None,
        "alpha_perturb": None,
        "noise_std": None,
        "threshold_norm_level": None,
        "seed": None,
        "epochs": epochs
    }

    for seed in seeds:

        for lr in lrs:

            torch.manual_seed(seed=seed)
            model = SimpleNet()
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)

            hyper_dict["lr"], hyper_dict["seed"], hyper_dict["perturb_and_shrink"] = lr, seed , False
            hyper_dict["threshold_level"], hyper_dict["alpha_perturb"], hyper_dict["noise_std"], hyper_dict["threshold_norm_level"] = None, None, None, None
            exp_name = f"vanilla lr={lr} s={seed}"
            hyper_dict["name"] = exp_name
            print(exp_name)
            run = setup_wandb(hyper_dict)
            
            top_eigenvalues, train_losses, train_accuracies, test_accuracies =  \
                train_model(
                    model,
                    train_loader,
                    test_loader,
                    optimizer,
                    criterion,
                    epochs,
                    device,
                    perturb_and_shrink=False,
                )
            run.finish()
            R[exp_name] = (top_eigenvalues, train_losses, train_accuracies, test_accuracies, hyper_dict.copy())

            hyper_dict["perturb_and_shrink"] = True

            for every_batch in every_batches:
                hyper_dict["every_batch"] = every_batch
                for threshold_level in threshold_levels:
                    hyper_dict["threshold_level"] = threshold_level
                    for alpha_perturb in alpha_perturbs:
                        hyper_dict["alpha_perturb"] = alpha_perturb
                        for noise_std in noise_stds:
                            hyper_dict["noise_std"] = noise_std
                            for threshold_norm_level in threshold_norm_levels:
                                hyper_dict["threshold_norm_level"] = threshold_norm_level
                                
                                torch.manual_seed(seed=seed)
                                model = SimpleNet()
                                model.to(device)
                                criterion = nn.CrossEntropyLoss()
                                optimizer = optim.SGD(model.parameters(), lr=lr)

                                exp_name = f"lr={lr} eb={every_batch} thr={threshold_level} a={alpha_perturb} std={noise_std} thr_norm={threshold_norm_level} s={seed}"
                                hyper_dict["name"] = exp_name
                                print(exp_name)

                                run = setup_wandb(hyper_dict)

                                top_eigenvalues, train_losses, train_accuracies, test_accuracies =  \
                                    train_model(
                                        model,
                                        train_loader,
                                        test_loader,
                                        optimizer,
                                        criterion,
                                        epochs,
                                        device,
                                        True, # perturb_and_shrink=
                                        every_batch, threshold_level, alpha_perturb, noise_std, threshold_norm_level
                                    )
                                run.finish()

                                R[exp_name] = (top_eigenvalues, train_losses, train_accuracies, test_accuracies, hyper_dict.copy())

    return R

def setup_wandb(config):
    wandb_project = "perturb_and_shrink"
    wandb_entity = "mateuj"
    wandb_tag = None
    wandb_group = "FC"
    run = wandb.init(project=wandb_project, entity=wandb_entity, config=config, 
                     dir="/home/mateusz.pyla/stan/atelier/sharpness/sp_results", 
                     tags=wandb_tag, group=wandb_group)
    
    return run

def plot_results(R):

    def do_colours(n_colours):
        if n_colours <= 7:
            return ['k', 'g', 'r', 'c', 'm', 'y', 'b']
        elif n_colours <= 20:
            cmap = plt.get_cmap('tab20')
            return [cmap(i) for i in range(n_colours)]
        else:
            cmap = plt.get_cmap('gist_rainbow')
            return [cmap(1.*i/n_colours) for i in range(n_colours)]

    # fig = plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(4, sharex=False, figsize=(12, 6))  # Create 3 subplots sharing x axis

    labels = []

    n_colours = len(R)
    colours = do_colours(n_colours)

    # Plot the accuracies and top eigenvalues
    for i, key in enumerate(R):
        top_eigenvalues, losses, train_accuracies, test_accuracies, hyper_dict = R[key]

        train_marker, test_marker = '-', '--'
        linewidth = 1.0
        if hyper_dict["perturb_and_shrink"] == False:
            train_marker, test_marker = (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))
            linewidth = 3.0
        colour_marker = colours[i]

        """
        plt.subplot(3, 1, 1)
        plt.scatter(range(len(train_accuracies)), train_accuracies, c=colour_marker, marker='+')
        plt.scatter(range(len(test_accuracies)), test_accuracies, c=colour_marker, marker='o')
        l, = plt.plot(train_accuracies, color=colour_marker, linestyle=train_marker, linewidth=linewidth, label='Train ' + key)
        labels.extend([l])
        l, = plt.plot(test_accuracies, color=colour_marker, linestyle=test_marker, linewidth=linewidth, label='Test ' + key)
        labels.extend([l])
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title('Training and Test Accuracies')

        plt.subplot(3, 1, 2)
        l, = plt.plot(losses, color=colour_marker, linewidth=linewidth, label=key)
        labels.extend([l])
        plt.title('Loss over Batches')
        plt.xlabel('Batch index (every 100th batch)')
        plt.ylabel('Top Eigenvalue')
        # plt.legend()

        plt.subplot(3, 1, 3)
        l, = plt.plot(top_eigenvalues, color=colour_marker, linewidth=linewidth, label=key)
        labels.extend([l])
        plt.title('Top Eigenvalues over Batches')
        plt.xlabel('Batch index (every 100th batch)')
        plt.ylabel('Top Eigenvalue')
        # plt.legend(fontsize = 'x-small')

        """
        axs[0].scatter(train_accuracies, range(len(train_accuracies)), c=colour_marker, marker='+', s=6)
        axs[0].scatter(test_accuracies, range(len(train_accuracies)), c=colour_marker, marker='o', s=6)
        axs[0].plot(range(len(train_accuracies)), train_accuracies, color=colour_marker, linestyle=train_marker, linewidth=linewidth, label='Train ' + key)  # Plot on first subplot
        axs[0].plot(range(len(train_accuracies)), test_accuracies, color=colour_marker, linestyle=test_marker, linewidth=linewidth, label='Test ' + key)
        axs[0].set_xlabel('Accuracy')
        # axs[0].set_yticks(np.arange(-1, len(train_accuracies)+2, step=1))
        axs[0].set_ylim([-1, len(train_accuracies)])
        axs[0].set_xticks(np.arange(0, 110, step=10))
        axs[0].set_ylabel('Epoch')
        axs[0].set_title('Training and Test Accuracies')
        axs[1].plot(losses, color=colour_marker, linewidth=linewidth, label=key)
        axs[1].set_xlabel('Batch index (every 100th batch)')
        axs[1].set_ylabel('Loss value')
        axs[1].set_title('Losses over batches')
        axs[2].plot(top_eigenvalues, color=colour_marker, linewidth=linewidth, label=key)
        axs[2].set_xlabel('Batch index (every 100th batch)')
        axs[2].set_ylabel('Top Eigenvalue')
        axs[1].set_title('Top Eigenvalues over batches')

    handles, labels = axs[0].get_legend_handles_labels()
    #handles += axs[1].get_legend_handles_labels()[0]
    #handles += axs[2].get_legend_handles_labels()[0]
    #labels += axs[1].get_legend_handles_labels()[1]
    #labels += axs[2].get_legend_handles_labels()[1]

    plt.tight_layout()
    # fig.legend(handles, labels, fontsize = 'x-small') # , bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[3].legend(handles, labels, loc='center', fontsize=4, ncol=4)
    axs[3].axis('off')

    # fig.legend(labels, [l.get_label() for l in labels], fontsize = 'x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig
    # wandb.log({"result": wandb.Image(fig)})

def main(seeds, lrs, every_batches):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=600, shuffle=True) # 100 batches
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False) # 100 batches

    device = "cuda"

    wandb.login(key="fbe8977ced9962ba4c826b6e012e35dad2c3f044")

    R = repeat_experiment(
        train_loader, test_loader, device, \
        seeds, lrs, every_batches, \
        threshold_levels=[0.5], #, 0.5, 1.0], \
        alpha_perturbs=[0.1, 0.5, 0.9], \
        noise_stds=[0.001, 0.01, 0.1], \
        threshold_norm_levels=[0.25, 0.5, 0.75]
    )

    with open(f'./sp_results/R_{lrs}_s{seeds}_eb{every_batches}.pkl', 'wb') as f:
        pickle.dump(R, f)

    f = plot_results(R)
    plt.savefig(f"./sp_results/R_{lrs}_s{seeds}_eb{every_batches}.png")

if __name__ == "__main__":

    # parse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='The first argument')
    parser.add_argument('lr', type=float, help='The second argument')
    parser.add_argument('every_batch', type=int, help='The third argument')

    args = parser.parse_args()

    seeds = [args.seed] # [0, 1, 2]
    lrs = [args.lr] # [0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    every_batches = [args.every_batch] # [20, 50, 100]
    main(seeds, lrs, every_batches)

    # with open(f"./sp_results/R_{[0.2]}_s{[0]}_eb{[20]}.pkl", 'rb') as f:
    #     R = pickle.load(f)
    #     f = plot_results(R)
    #     plt.savefig(f"./sp_results/R_{[0.2]}_s{[0]}_eb{[20]}.png")
