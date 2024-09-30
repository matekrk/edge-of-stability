import os
import numpy as np
from matplotlib import pyplot as plt
import torch

def simple_plot_rez(pruning_rez, pruning_ps, train_dataset_name, test_datasets_names, network_path):
    fig, axs = plt.subplots(3, 1, figsize=(30, 20))
    plt.tight_layout()
    cmap = plt.cm.get_cmap('viridis')
    markers = ["o", "x", "^", "*", "s"]
    plt.suptitle(f"{network_path} trained on {train_dataset_name} {[(test_dataset_name,markers[k]) for k,test_dataset_name in enumerate(test_datasets_names)]}", fontsize=20)    

    # IN ACC
    for i, p in enumerate([0.0] + pruning_ps):
        v = pruning_rez[p][0][1]
        axs[0].scatter([p], [v], color=cmap(i/len(pruning_ps)))
        axs[0].set_ylim(0, 1)
    #axs[0].text(0.0, -0.01, f"trained on {train_dataset_name}" , fontsize=15)
    # OUT ACC
    for i, p in enumerate(pruning_ps):
        for j, v in enumerate(pruning_rez[p][1]):
            axs[1].scatter([p], [v[1]], color=cmap(i/len(pruning_ps)), marker=markers[j])
        axs[1].set_ylim(0, 1)
    #axs[2].text(0.0, -0.01, f"{[(test_dataset_name,markers[k]) for k,test_dataset_name in enumerate(test_datasets_names)]}", fontsize=15)
    # RATIO
    for i, p in enumerate(pruning_ps):
        v = pruning_rez[p][2]
        axs[2].scatter([p], [v], color=cmap(i/len(pruning_ps)), label=str(p))

    axs[2].legend()
    
    # LEGEND
    #axs[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #axs[5].axis('off')  # Turn off the axis for the legend subplot

    axs[0].set_title('In accuracy')
    axs[1].set_title('Out accuracy')
    axs[2].set_title('Ratio')
    axs[0].set_xlabel('percent')
    axs[1].set_xlabel('percent')
    axs[2].set_xlabel('percent')

    # plt.subplots_adjust(wspace=1.0)

    return fig

def plot_rez(pruning_rez, faithfullness_ps, pruning_ps, train_dataset_name, test_datasets_names, network_path, pruning_importance_str):
    n_ps = len(pruning_ps)
    n_tests = len(test_datasets_names)
    fig, axs = plt.subplots(n_tests + 1, 3, figsize=(30, 20))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, n_ps+1))

    plt.suptitle(f"pruning {network_path} strategy {pruning_importance_str}")

    handles, labels = [], []

    for i, p in enumerate([0.0] + pruning_ps):
        in_loss, in_acc = pruning_rez[p][0][0], pruning_rez[p][0][1]
        for j in range(n_tests):
            out_loss, out_acc = pruning_rez[p][1][j][0], pruning_rez[p][1][j][1]
            out_faithfullness = faithfullness_ps[p][1][j]
            axs[j, 0].scatter([in_loss], [out_loss], color=colors[i], label=f"{p}")
            axs[j, 1].scatter([in_acc], [out_acc], color=colors[i])
            axs[j, 2].scatter([in_acc], [out_faithfullness], color=colors[i])
            if i == 0:
                axs[j, 0].set_title("Loss")
                axs[j, 0].set_xlabel(f"{train_dataset_name}")
                axs[j, 0].set_ylabel(f"{test_datasets_names[j]}")
                axs[j, 1].set_title("Acc")
                axs[j, 1].set_xlabel(f"{train_dataset_name}")
                axs[j, 1].set_ylabel(f"{test_datasets_names[j]}")
                axs[j, 2].set_title("Faithfullness")
                axs[j, 2].set_xlabel(f"acc IN {train_dataset_name}")
                axs[j, 2].set_ylabel(f"faith OUT {test_datasets_names[j]}")

        axs[n_tests, 0].scatter([p], [pruning_rez[p][2]], color=colors[i], label=f"{p:.5f}")
    handles, labels = axs[n_tests, 0].get_legend_handles_labels()
    #handles = handles + handles1
    #labels = labels + labels1
    
    axs[n_tests, 1].axis("off")
    axs[n_tests, 1].legend(handles, labels, fontsize=12, ncol=n_ps // 12)
    axs[n_tests, 2].axis("off")
    # fig.colorbar(colors, ax=axs[n_tests, 2])


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    return fig
