from math import sqrt
import os
from typing import List
import wandb
import matplotlib.pyplot as plt

def init(args, file_key, file_entity, file_project):
    f = open(file_key, "r")
    wandb_key = f.read()
    wandb.login(key=wandb_key)

    f = open(file_entity, "r")
    wandb_entity = f.read()

    f = open(file_project, "r")
    wandb_project = f.read()

    run = wandb.init(project=wandb_project, entity=wandb_entity, config=args, dir=os.environ["RESULTS"], 
                     tags=args.wandb_tag, group=args.wandb_group, mode="disabled" if args.no_wandb else None)
    
    wandb.define_metric("train/step")
    wandb.define_metric("train/grads")
    wandb.define_metric("test/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("test/*", step_metric="test/step")

def log(results, step, summary = False, metric = False):
    if summary:
        for k, v in results.items():
            wandb.run.summary[k] = v
    elif metric:
        for k, v in results.items():
            wandb.define_metric(k, step_metric=v)
    else:
        for k, v in results.items():
            if isinstance(v, float) or isinstance(v, int):
                results[k] = v
            elif k.endswith("evecs"):
                results[k] = wandb.Image(plt.imshow(v), cmap="viridis")
            elif k.endswith("evals"):
                results[k] = wandb.Image(plt.bar(range(len(v)), v))
            elif k.endswith("hist"):
                n_bins = int(sqrt(len(v))) if len(v) > 4000 else 64
                results[k] = wandb.Histogram(v, num_bins=n_bins)
            elif isinstance(v, plt.Figure):
                results[k] = wandb.Image(v)
            elif isinstance(v, list):
                results[k] = [wandb.Image(img) for img in v]
            elif isinstance(v, tuple):
                tab_content, tab_columns, tab_label, tab_value, tab_title = v
                assert tab_content is not None and tab_columns is not None and tab_label is not None and tab_value is not None and tab_title is not None
                tab = wandb.Table(data=tab_content, columns=tab_columns)
                results[k] = wandb.plot.bar(tab, tab_label, tab_value, tab_title)
            
        wandb.log(results, step=step)
