
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

import torch

from relative_space import transform_space

take_first = 100
vgg_dim = 512

names = ["CE-0.01-0", "CE-0.05-0", "CE-0.01-1", "CE-0.3-0", "MSE-0.01-0", "MSE-0.05-0", "MSE-0.1-0", "MSE-0.3-0"]

p_ce_001 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.01/traj"
p_ce_005 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.05/traj"
p_ce_01 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_1/ce/gd/lr_0.01/traj" # MODIFIED!!
p_ce_03 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.3/traj"
p_mse_001 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.01/traj"
p_mse_005 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.05/traj"
p_mse_01 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.1/traj"
p_mse_03 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.3/traj"

my_markers = ["<", "v", "^", ">", "o", "s", "*", "X"] # , "x", "P"]

feat_p_ce_001 = torch.load(os.path.join(p_ce_001, "all.pt"))
feat_p_ce_005 = torch.load(os.path.join(p_ce_005, "all.pt"))
feat_p_ce_01 = torch.load(os.path.join(p_ce_01, "all.pt"))
feat_p_ce_03 = torch.load(os.path.join(p_ce_03, "all.pt"))
feat_p_mse_001 = torch.load(os.path.join(p_mse_001, "all.pt"))
feat_p_mse_005 = torch.load(os.path.join(p_mse_005, "all.pt"))
feat_p_mse_01 = torch.load(os.path.join(p_mse_01, "all.pt"))
feat_p_mse_03 = torch.load(os.path.join(p_mse_03, "all.pt"))

loss_p_ce_001 = torch.load(os.path.join(p_ce_001, "losses.pt"))
loss_p_ce_005 = torch.load(os.path.join(p_ce_005, "losses.pt"))
loss_p_ce_01 = torch.load(os.path.join(p_ce_01, "losses.pt"))
loss_p_ce_03 = torch.load(os.path.join(p_ce_03, "losses.pt"))
loss_p_mse_001 = torch.load(os.path.join(p_mse_001, "losses.pt"))
loss_p_mse_005 = torch.load(os.path.join(p_mse_005, "losses.pt"))
loss_p_mse_01 = torch.load(os.path.join(p_mse_01, "losses.pt"))
loss_p_mse_03 = torch.load(os.path.join(p_mse_03, "losses.pt"))

"""
loss_p_ce_001 = torch.nn.functional.normalize(loss_p_ce_001, dim=0)
loss_p_ce_005 = torch.nn.functional.normalize(loss_p_ce_005, dim=0)
loss_p_ce_01 = torch.nn.functional.normalize(loss_p_ce_01, dim=0)
loss_p_ce_03 = torch.nn.functional.normalize(loss_p_ce_03, dim=0)
loss_p_mse_001 = torch.nn.functional.normalize(loss_p_mse_001, dim=0)
loss_p_mse_005 = torch.nn.functional.normalize(loss_p_mse_005, dim=0)
loss_p_mse_01 = torch.nn.functional.normalize(loss_p_mse_01, dim=0)
loss_p_mse_03 = torch.nn.functional.normalize(loss_p_mse_03, dim=0)
"""
loss_p_ce_001 -= torch.median(loss_p_ce_001)
loss_p_ce_005 -= torch.median(loss_p_ce_005)
loss_p_ce_01 -= torch.median(loss_p_ce_01)
loss_p_ce_03 -= torch.median(loss_p_ce_03)
loss_p_mse_001 -= torch.median(loss_p_mse_001)
loss_p_mse_005 -= torch.median(loss_p_mse_005)
loss_p_mse_01 -= torch.median(loss_p_mse_01)
loss_p_mse_03 -= torch.median(loss_p_mse_03)
#"""

STEPS = 50
DIMS = (take_first, vgg_dim)
NUM_ANCH = take_first//10
ind_anchors = np.random.choice(np.arange(take_first), NUM_ANCH, replace=False)

#p_ce_001
transf_p_ce_001 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_001):
    X = X[:take_first, :]
    transf_p_ce_001[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_ce_001 = loss_p_ce_001[:take_first]

#p_ce_005
transf_p_ce_005 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_005):
    X = X[:take_first, :]
    transf_p_ce_005[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_ce_005 = loss_p_ce_005[:take_first]

#p_ce_01
transf_p_ce_01 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_01):
    X = X[:take_first, :]
    transf_p_ce_01[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_ce_01 = loss_p_ce_01[:take_first]

#p_ce_03
transf_p_ce_03 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_03):
    X = X[:take_first, :]
    transf_p_ce_03[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_ce_01 = loss_p_ce_01[:take_first]

#p_mse_001
transf_p_mse_001 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_001):
    X = X[:take_first, :]
    transf_p_mse_001[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_mse_001 = loss_p_mse_001[:take_first]

#p_mse_005
transf_p_mse_005 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_005):
    X = X[:take_first, :]
    transf_p_mse_005[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_01
transf_p_mse_01 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_01):
    X = X[:take_first, :]
    transf_p_mse_01[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_03
transf_p_mse_03 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_03):
    X = X[:take_first, :]
    transf_p_mse_03[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
    

transformed = np.concatenate((transf_p_ce_001, transf_p_ce_005, transf_p_ce_01, transf_p_ce_03, transf_p_mse_001, transf_p_mse_005, transf_p_mse_01, transf_p_mse_03)) #.reshape((8*STEPS,take_first*NUM_ANCH))
y_loss = torch.cat((loss_p_ce_001, loss_p_ce_005, loss_p_ce_01, loss_p_ce_03, loss_p_mse_001, loss_p_mse_005, loss_p_mse_01, loss_p_mse_03))

pca = PCA(n_components=2, random_state=0)
visio = pca.fit_transform(transformed)

visio_back = visio.reshape(8,-1, 2)
sets = []
for i in range(8):
    visio_df = pd.DataFrame(data=visio_back[i], columns = ['principal component 1', 'principal component 2'])
    visio_df['y_loss'] = y_loss[i*STEPS:(i+1)*STEPS]
    sets.append(visio_df)

concatenated = pd.concat([sets[i].assign(dataset=names[i]) for i in range(8)])
concatenated.rename(columns={'dataset': 'method'}, inplace=True)

fig = plt.figure(figsize=(16,10))
sns.scatterplot(x="principal component 1", y="principal component 2", data=concatenated,
                hue='y_loss', style='method', markers=my_markers, palette=sns.color_palette("hot", as_cmap=True), alpha=0.3)

"""
principal_cifar_Df = pd.DataFrame(data = visios
            , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y_loss'] = y_loss.numpy()

fig = plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[:STEPS], legend="brief", markers=my_markers[0], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[STEPS:2*STEPS], legend="brief", markers=my_markers[1], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[2*STEPS:3*STEPS], legend="brief", markers=my_markers[2], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[3*STEPS:4*STEPS], legend="brief", markers=my_markers[3], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[4*STEPS:5*STEPS], legend="brief", markers=my_markers[4], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[5*STEPS:6*STEPS], legend="brief", markers=my_markers[5], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[6*STEPS:7*STEPS], legend="brief", markers=my_markers[5], alpha=0.3
)
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y_loss", palette=sns.color_palette("crest", as_cmap=True),
    data=principal_cifar_Df[7*STEPS:8*STEPS], legend="brief", markers=my_markers[5], alpha=0.3
)
"""

#kws = dict(s=50, linewidth=.5, edgecolor="w")
#pal = ['red', 'green', 'blue', 'red', 'green', 'blue',]
#g = sns.FacetGrid(principal_cifar_Df, col="sex", hue="size", palette=pal, hue_kws=dict(marker=["^", "^", "^", "v", "v", "v"]))
#g = (g.map(plt.scatter, "total_bill", "tip", **kws).add_legend())

plt.savefig("combined.png")

#for file in os.listdir("/mydir"):
#    if file.endswith(".txt"):
#        print(os.path.join("/mydir", file))
#wandb.log({"trajectory_loss": wandb.Image(plt)})
    