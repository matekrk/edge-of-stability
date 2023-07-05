
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

import torch

from relative_space import transform_space

NUM_METHODS = 6 # 3 # 10
STEPS = 50

take_first = 100
vgg_dim = 512
DIMS = (take_first, vgg_dim)

NUM_ANCH = take_first//10
ind_anchors = np.random.choice(np.arange(take_first), NUM_ANCH, replace=False)

p_ce_0001 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.001/traj"
p_ce_001_0 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.01/traj"
p_ce_001_1 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_1/ce/gd/lr_0.01/traj"
p_ce_001_2 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_2/ce/gd/lr_0.01/traj"
p_ce_005_0 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.05/traj"
p_ce_005_1 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_3/ce/gd/lr_0.05/traj"
p_ce_005_2 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_4/ce/gd/lr_0.05/traj"
p_ce_01 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.1/traj"
p_ce_03 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/ce/gd/lr_0.3/traj"
p_mse_0001 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.001/traj"
p_mse_001_0 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.01/traj"
p_mse_001_1 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_1/mse/gd/lr_0.01/traj"
p_mse_001_2 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_2/mse/gd/lr_0.01/traj"
p_mse_005_0 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.05/traj"
p_mse_005_1 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_3/mse/gd/lr_0.05/traj"
p_mse_005_2 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_4/mse/gd/lr_0.05/traj"
p_mse_01 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.1/traj"
p_mse_03 = "/home/mateusz.pyla/stan/rez/cifar10-5k-1k/vgg11/seed_0/mse/gd/lr_0.3/traj"

my_markers = ["D", "<", "v", "^", ">", "o", "s", "*", "X", "P"]
#assert len(my_markers) == NUM_METHODS
my_shorter_markers = ["D", "o", "X"]
#assert len(my_shorter_markers) == NUM_METHODS
my_medium_markers = ["v", "^", ">", "o", "*", "P"]
assert len(my_medium_markers) == NUM_METHODS

feat_p_ce_0001 = torch.load(os.path.join(p_ce_0001, "all.pt"))
feat_p_ce_001_0 = torch.load(os.path.join(p_ce_001_0, "all.pt"))
feat_p_ce_001_1 = torch.load(os.path.join(p_ce_001_1, "all.pt"))
feat_p_ce_001_2 = torch.load(os.path.join(p_ce_001_2, "all.pt"))
feat_p_ce_005_0 = torch.load(os.path.join(p_ce_005_0, "all.pt"))
feat_p_ce_005_1 = torch.load(os.path.join(p_ce_005_1, "all.pt"))
feat_p_ce_005_2 = torch.load(os.path.join(p_ce_005_2, "all.pt"))
feat_p_ce_01 = torch.load(os.path.join(p_ce_01, "all.pt"))
feat_p_ce_03 = torch.load(os.path.join(p_ce_03, "all.pt"))
feat_p_mse_0001 = torch.load(os.path.join(p_mse_0001, "all.pt"))
feat_p_mse_001_0 = torch.load(os.path.join(p_mse_001_0, "all.pt"))
feat_p_mse_001_1 = torch.load(os.path.join(p_mse_001_1, "all.pt"))
feat_p_mse_001_2 = torch.load(os.path.join(p_mse_001_2, "all.pt"))
feat_p_mse_005_0 = torch.load(os.path.join(p_mse_005_0, "all.pt"))
feat_p_mse_005_1 = torch.load(os.path.join(p_mse_005_1, "all.pt"))
feat_p_mse_005_2 = torch.load(os.path.join(p_mse_005_2, "all.pt"))
feat_p_mse_01 = torch.load(os.path.join(p_mse_01, "all.pt"))
feat_p_mse_03 = torch.load(os.path.join(p_mse_03, "all.pt"))

loss_p_ce_0001 = torch.load(os.path.join(p_ce_0001, "losses.pt"))
loss_p_ce_001_0 = torch.load(os.path.join(p_ce_001_0, "losses.pt"))
loss_p_ce_001_1 = torch.load(os.path.join(p_ce_001_1, "losses.pt"))
loss_p_ce_001_2 = torch.load(os.path.join(p_ce_001_2, "losses.pt"))
loss_p_ce_005_0 = torch.load(os.path.join(p_ce_005_0, "losses.pt"))
loss_p_ce_005_1 = torch.load(os.path.join(p_ce_005_1, "losses.pt"))
loss_p_ce_005_2 = torch.load(os.path.join(p_ce_005_2, "losses.pt"))
loss_p_ce_01 = torch.load(os.path.join(p_ce_01, "losses.pt"))
loss_p_ce_03 = torch.load(os.path.join(p_ce_03, "losses.pt"))
loss_p_mse_0001 = torch.load(os.path.join(p_mse_0001, "losses.pt"))
loss_p_mse_001_0 = torch.load(os.path.join(p_mse_001_0, "losses.pt"))
loss_p_mse_001_1 = torch.load(os.path.join(p_mse_001_1, "losses.pt"))
loss_p_mse_001_2 = torch.load(os.path.join(p_mse_001_2, "losses.pt"))
loss_p_mse_005_0 = torch.load(os.path.join(p_mse_005_0, "losses.pt"))
loss_p_mse_005_1 = torch.load(os.path.join(p_mse_005_1, "losses.pt"))
loss_p_mse_005_2 = torch.load(os.path.join(p_mse_005_2, "losses.pt"))
loss_p_mse_01 = torch.load(os.path.join(p_mse_01, "losses.pt"))
loss_p_mse_03 = torch.load(os.path.join(p_mse_03, "losses.pt"))

#"""
loss_p_ce_0001 = torch.nn.functional.normalize(loss_p_ce_0001, dim=0)
loss_p_ce_001_0 = torch.nn.functional.normalize(loss_p_ce_001_0, dim=0)
loss_p_ce_001_1 = torch.nn.functional.normalize(loss_p_ce_001_1, dim=0)
loss_p_ce_001_2 = torch.nn.functional.normalize(loss_p_ce_001_2, dim=0)
loss_p_ce_005_0 = torch.nn.functional.normalize(loss_p_ce_005_0, dim=0)
loss_p_ce_005_1 = torch.nn.functional.normalize(loss_p_ce_005_1, dim=0)
loss_p_ce_005_2 = torch.nn.functional.normalize(loss_p_ce_005_2, dim=0)
loss_p_ce_01 = torch.nn.functional.normalize(loss_p_ce_01, dim=0)
loss_p_ce_03 = torch.nn.functional.normalize(loss_p_ce_03, dim=0)
loss_p_mse_0001 = torch.nn.functional.normalize(loss_p_mse_0001, dim=0)
loss_p_mse_001_0 = torch.nn.functional.normalize(loss_p_mse_001_0, dim=0)
loss_p_mse_001_1 = torch.nn.functional.normalize(loss_p_mse_001_1, dim=0)
loss_p_mse_001_2 = torch.nn.functional.normalize(loss_p_mse_001_2, dim=0)
loss_p_mse_005_0 = torch.nn.functional.normalize(loss_p_mse_005_0, dim=0)
loss_p_mse_005_1 = torch.nn.functional.normalize(loss_p_mse_005_1, dim=0)
loss_p_mse_005_2 = torch.nn.functional.normalize(loss_p_mse_005_2, dim=0)
loss_p_mse_01 = torch.nn.functional.normalize(loss_p_mse_01, dim=0)
loss_p_mse_03 = torch.nn.functional.normalize(loss_p_mse_03, dim=0)
"""
loss_p_ce_0001 -= torch.median(loss_p_ce_0001)
loss_p_ce_001_0 -= torch.median(loss_p_ce_001_0)
loss_p_ce_001_1 -= torch.median(loss_p_ce_001_1)
loss_p_ce_001_2 -= torch.median(loss_p_ce_001_2)
loss_p_ce_005_0 -= torch.median(loss_p_ce_005_0)
loss_p_ce_005_1 -= torch.median(loss_p_ce_005_1)
loss_p_ce_005_2 -= torch.median(loss_p_ce_005_2)
loss_p_ce_01 -= torch.median(loss_p_ce_01)
loss_p_ce_03 -= torch.median(loss_p_ce_03)
loss_p_mse_0001 -= torch.median(loss_p_mse_0001)
loss_p_mse_001_0 -= torch.median(loss_p_mse_001_0)
loss_p_mse_001_1 -= torch.median(loss_p_mse_001_1)
loss_p_mse_001_2 -= torch.median(loss_p_mse_001_2)
loss_p_mse_005_0 -= torch.median(loss_p_mse_005_0)
loss_p_mse_005_1 -= torch.median(loss_p_mse_005_1)
loss_p_mse_005_2 -= torch.median(loss_p_mse_005_2)
loss_p_mse_01 -= torch.median(loss_p_mse_01)
loss_p_mse_03 -= torch.median(loss_p_mse_03)
#"""

#p_ce_0001
transf_p_ce_0001 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_0001):
    X = X[:take_first, :]
    transf_p_ce_0001[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_001_0
transf_p_ce_001_0 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_001_0):
    X = X[:take_first, :]
    transf_p_ce_001_0[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_001_1
transf_p_ce_001_1 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_001_1):
    X = X[:take_first, :]
    transf_p_ce_001_1[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_001_2
transf_p_ce_001_2 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_001_2):
    X = X[:take_first, :]
    transf_p_ce_001_2[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
#loss_p_ce_002_2 = loss_p_ce_001_2[:take_first]

#p_ce_005_0
transf_p_ce_005_0 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_005_0):
    X = X[:take_first, :]
    transf_p_ce_005_0[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_005_1
transf_p_ce_005_1 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_005_1):
    X = X[:take_first, :]
    transf_p_ce_005_1[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_005_2
transf_p_ce_005_2 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_005_2):
    X = X[:take_first, :]
    transf_p_ce_005_2[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_01
transf_p_ce_01 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_01):
    X = X[:take_first, :]
    transf_p_ce_01[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_ce_03
transf_p_ce_03 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_ce_03):
    X = X[:take_first, :]
    transf_p_ce_03[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_0001
transf_p_mse_0001 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_0001):
    X = X[:take_first, :]
    transf_p_mse_0001[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_001
transf_p_mse_001_0 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_001_0):
    X = X[:take_first, :]
    transf_p_mse_001_0[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_001
transf_p_mse_001_1 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_001_1):
    X = X[:take_first, :]
    transf_p_mse_001_1[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_001
transf_p_mse_001_2 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_001_2):
    X = X[:take_first, :]
    transf_p_mse_001_2[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_005_0
transf_p_mse_005_0 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_005_0):
    X = X[:take_first, :]
    transf_p_mse_005_0[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_005_1
transf_p_mse_005_1 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_005_1):
    X = X[:take_first, :]
    transf_p_mse_005_1[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_005_2
transf_p_mse_005_2 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_005_2):
    X = X[:take_first, :]
    transf_p_mse_005_2[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_01
transf_p_mse_01 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_01):
    X = X[:take_first, :]
    transf_p_mse_01[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])

#p_mse_03
transf_p_mse_03 = np.zeros((STEPS, take_first * NUM_ANCH))
for i, X in enumerate(feat_p_mse_03):
    X = X[:take_first, :]
    transf_p_mse_03[i] = torch.flatten(transform_space(X, ind_anchors=ind_anchors)[0])
    

#transformed = np.concatenate((transf_p_ce_0001, transf_p_ce_001_0, transf_p_ce_005_0, transf_p_ce_01, transf_p_ce_03, transf_p_mse_0001, transf_p_mse_001_0, transf_p_mse_005_0, transf_p_mse_01, transf_p_mse_03))
#y_loss = torch.cat((loss_p_ce_0001, loss_p_ce_001_0, loss_p_ce_001_1, loss_p_ce_001_2, loss_p_ce_005_0, loss_p_ce_005_1, loss_p_ce_005_2, loss_p_ce_01, loss_p_ce_03, loss_p_mse_0001, loss_p_mse_001_0, loss_p_mse_001_1, loss_p_mse_001_2, loss_p_mse_005_0, loss_p_mse_005_1, loss_p_mse_005_2,  loss_p_mse_01, loss_p_mse_03))

#transformed = np.concatenate((transf_p_ce_005_0, transf_p_ce_005_1, transf_p_ce_005_2))
#y_loss = torch.cat((loss_p_ce_005_0, loss_p_ce_005_1, loss_p_ce_005_2))
#transformed = np.concatenate((transf_p_mse_005_0, transf_p_mse_005_1, transf_p_mse_005_2))
#y_loss = torch.cat((loss_p_mse_005_0, loss_p_mse_005_1, loss_p_mse_005_2))
transformed = np.concatenate((transf_p_ce_001_0, transf_p_ce_001_1, transf_p_ce_001_2, transf_p_ce_005_0, transf_p_ce_005_1, transf_p_ce_005_2))
y_loss = torch.cat((loss_p_ce_001_0, loss_p_ce_001_1, loss_p_ce_001_2, loss_p_ce_005_0, loss_p_ce_005_1, loss_p_ce_005_2))


pca = PCA(n_components=2, random_state=0)
visio = pca.fit_transform(transformed)

visio_back = visio.reshape(NUM_METHODS,-1, 2)
sets = []
for i in range(NUM_METHODS):
    visio_df = pd.DataFrame(data=visio_back[i], columns = ['principal component 1', 'principal component 2'])
    visio_df['y_loss'] = y_loss[i*STEPS:(i+1)*STEPS]
    sets.append(visio_df)

#names = ["CE-0.001-0", "CE-0.01-0", "CE-0.05-0", "CE-0.1-0", "CE-0.3-0", "MSE-0.001-0", "MSE-0.01-0", "MSE-0.05-0", "MSE-0.1-0", "MSE-0.3-0"]
names = ["CE-0.01-0", "CE-0.01-1", "CE-0.01-2", "CE-0.05-0", "CE-0.05-1", "CE-0.05-2"]
#names = ["MSE-0.05-0", "MSE-0.05-1", "MSE-0.05-2"]
assert len(names) == NUM_METHODS

concatenated = pd.concat([sets[i].assign(dataset=names[i]) for i in range(NUM_METHODS)])
concatenated.rename(columns={'dataset': 'method'}, inplace=True)

fig = plt.figure(figsize=(16,10))
sns.scatterplot(x="principal component 1", y="principal component 2", data=concatenated,
                hue='y_loss', style='method', markers=my_medium_markers, palette=sns.color_palette("hot", as_cmap=True), alpha=0.3)

plt.savefig("figures/combined_ce_0.01_0.05_norm.png")

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

#for file in os.listdir("/mydir"):
#    if file.endswith(".txt"):
#        print(os.path.join("/mydir", file))
#wandb.log({"trajectory_loss": wandb.Image(plt)})
    