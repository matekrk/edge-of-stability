from matplotlib import pyplot as plt

def plot_average(dict_models):
    pass

def parse_vslr(aggregate_models, by_metric):
    pass

def do_something_vslr():
    pass

def plot_vslr(model, dict_models):
    lrs, vals = [], []

    for k, v in dict_models:
        lrs.append(k)
        load_directory = get_gd_directory(dataset, arch_id, loss, opt, lr, eig_freq, seed, beta, delta, 86)
        network = model
        network.load_state_dict(torch.load(f"{load_directory}/model_snapshot_final_{load_step}"))
        x = do_something_vslr()
        vals.append(x)

        
    plt.scatter(lrs, vals)