import wandb
import torch

from sam import SAM
from traker import trak_onebatch
from utilities import compute_empirical_sharpness, compute_grad_norm, compute_loss_for_single_instance, get_dataloader, iterate_dataset, split_batch, split_batch_trak

def omega_penalty():
    pass

def train_epoch(network, train_dataset, physical_batch_size, loss_fn, acc_fn, 
                optimizer, optimizer_outliners_data, optimizer_outliners_features,
                swa, swa_model, swa_scheduler, ema, ema_model, ema_decay,
                omega_wd_0, omega_wd_1, omega_wd_2,
                eliminate_outliners_data, eliminate_outliners_data_strategy, eliminate_outliners_data_gamma,
                eliminate_outliners_features, eliminate_outliners_features_strategy, eliminate_outliners_features_gamma,
                keep_random_layers, keep_random_neurons, log_epoch, last_layer = None):

    if last_layer is None:
        if network._get_name() == "LeNet2":
            last_layer = network.fc3
        elif network._get_name() == "ResNet9":
            last_layer = network.classifier[3]
        elif network._get_name() == "ResNet":
            last_layer = network.fc

    network.train()

    gradients = 0
    optimizer.zero_grad()
    X_loss, X_acc, X_e1 = [], [], [] 
    out_ratio, in_loss, in_acc, out_loss, out_acc, in_e1, out_e1 = [], [], [], [], [], [], []
    weighted_norms, L0_norms, L1_norms, L2_norms = [], [], [], []

    for batch in iterate_dataset(train_dataset, physical_batch_size):
        (X, y) = batch
        X, y = X.cuda(), y.cuda()

        total_weighted_norm, total_L0_norm, total_L1_norm, total_L2_norm = omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2, return_absolute=True)
        weighted_norms.append(total_weighted_norm)
        L0_norms.append(total_L0_norm)
        L1_norms.append(total_L1_norm)
        L2_norms.append(total_L2_norm)

        if eliminate_outliners_data:
            if eliminate_outliners_data_strategy == "gradient_vmap":
                network.eval()
                optimizer.zero_grad()
                vmap_loss = torch.vmap(compute_loss_for_single_instance, in_dims=(None, None, 0, 0))
                losses = vmap_loss(network, loss_fn, X, y)
                norm_gradients = [compute_grad_norm(torch.autograd.grad(loss, network.parameters(), retain_graph=True)).cpu().numpy() for loss in losses]
                gradients += 1
                X_inliners, y_inliners, X_outliners, y_outliners = split_batch(X, y, norm_gradients, eliminate_outliners_data_gamma)

                if log_epoch is not None:
                    out_ratio.append(len(X_outliners)/len(X))
                    in_loss.append(loss_fn(network(X_inliners), y_inliners)/len(X_inliners))
                    in_acc.append(acc_fn(network(X_inliners), y_inliners)/len(X_inliners))
                    out_loss.append(loss_fn(network(X_outliners), y_outliners)/len(X_outliners))
                    out_acc.append(acc_fn(network(X_outliners), y_outliners)/len(X_outliners))
                    in_e1.append(compute_empirical_sharpness(network, loss_fn, X_inliners, y_inliners))
                    out_e1.append(compute_empirical_sharpness(network, loss_fn, X_outliners, y_outliners))
            
            elif eliminate_outliners_data_strategy == "trak": # FIXME
                train_dataloader = get_dataloader(train_dataset, physical_batch_size)
                scores = trak_onebatch(network, "/home/mateuszpyla/stan/sharpness/results/mnist/lenet2/mse/sgd/lr_0.1/seed_12/freq_1/start_0", train_dataloader, batch, len(train_dataloader.dataset), len(X), max_models=15, save_dir_end="duringtraining")
                X_inliners, y_inliners, X_outliners, y_outliners = split_batch_trak(X, y, scores)
                gradients += 1

                if log_epoch is not None:
                    out_ratio.append(len(X_outliners)/len(X))
                    in_loss.append(loss_fn(network(X_inliners), y_inliners)/len(X_inliners))
                    in_acc.append(acc_fn(network(X_inliners), y_inliners)/len(X_inliners))
                    out_loss.append(loss_fn(network(X_outliners), y_outliners)/len(X_outliners))
                    out_acc.append(acc_fn(network(X_outliners), y_outliners)/len(X_outliners))
                    in_e1.append(compute_empirical_sharpness(network, loss_fn, X_inliners, y_inliners))
                    out_e1.append(compute_empirical_sharpness(network, loss_fn, X_outliners, y_outliners))
            else:
                raise NotImplementedError()
                
        else:
            X_inliners, y_inliners = X, y
            X_outliners, y_outliners = None, None

        mask_outweights = torch.zeros(last_layer.weight.shape[1], dtype=torch.bool)
        if eliminate_outliners_features:
            if eliminate_outliners_features_strategy in ["gradient_vmap", "grad"]:
                optimizer.zero_grad()
                loss = loss_fn(network(X_inliners), y_inliners) / len(X_inliners)
                loss.backward()
                gradients += 1
                grads = last_layer.weight.grad
                _, outweight = torch.topk(grads.abs(), k=int(eliminate_outliners_features_gamma), dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights = torch.zeros(grads.shape[1], dtype=torch.bool)
                mask_outweights[outweight] = 1
                #if grad_ind_grad_lr is not None:
                #    optimizer.param_groups[-2]['lr'] = grad_ind_grad_lr
                #TODO: mask

                
                # def f(network, x):
            #     _, f = network(x, return_features=True)

                #vmap_loss = torch.vmap(compute_loss_for_single_instance, in_dims=(None, None, 0))
                #losses = vmap_loss(network, f, X)

                # prev_grad = optimizer.param_groups[0]['params'][-2].grad.copy()
                # optimizer.param_groups[0]['params'][-2].grad = 0.0
                # inweight = torch.inverse(outweight)
                # if grad_ind_grad_lr is not None:
                #     optimizer.param_groups["fc3.weight"]["lr"] = grad_ind_grad_lr
                # optimizer.add_param_group({'params': optimizer.param_groups[0]['params'][-2]})

            elif eliminate_outliners_features_strategy == "norm":
                optimizer.zero_grad()
                outputs, features = network(X_inliners, return_features=True)
                loss = loss_fn(outputs, y_inliners) / len(X_inliners)
                loss.backward()
                gradients += 1
                mean_feat_norm = features.norm(dim=0)
                grads = last_layer.weight.grad
                _, outweight = torch.topk(mean_feat_norm.abs(), k=int(eliminate_outliners_features_gamma), dim=1)
                outweight = outweight.flatten().unique()
                mask_outweights[outweight] = 1

                torch.autograd.grad(loss, network.parameters(), retain_graph=True)
                gradients += 1

        # TODO: keep_random_neurons
        frozen_layers = []
        if keep_random_layers:
            for name, param in network.named_parameters():
                if torch.rand(1).item() < keep_random_layers:
                    param.requires_grad = False
                    frozen_layers.append(param)

        optimizer.zero_grad()
        f_X_inliners = network(X_inliners)
        loss = loss_fn(f_X_inliners, y_inliners) / len(X_inliners)
        loss += omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2)

        if not optimizer_outliners_data:
            X_loss.append(loss.item())
            X_acc.append((acc_fn(f_X_inliners, y_inliners).item())/len(X_inliners))
            X_e1.append(compute_empirical_sharpness(network, loss_fn, X_inliners, y_inliners))

        loss.backward()
        last_layer.weight.grad[:, mask_outweights] = 0.0
        if isinstance(optimizer, SAM):
            def closure():
                loss = loss_fn(network(X_inliners), y_inliners) / len(X_inliners)
                loss += omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2)
                loss.backward()
                last_layer.weight.grad[:, mask_outweights] = 0.0
                return loss
            optimizer.step(closure)
            gradients += 2
        else:
            optimizer.step()
            gradients += 1

        if optimizer_outliners_data:
            optimizer_outliners_data.zero_grad()
            loss = loss_fn(network(X_outliners), y_outliners) / len(X_outliners)
            loss += omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2)
            loss.backward()
            last_layer.weight.grad[:, mask_outweights] = 0.0
            if isinstance(optimizer_outliners_data, SAM):
                def closure():
                    loss = loss_fn(network(X_inliners), y_inliners) / len(train_dataset)
                    loss.backward()
                    last_layer.weight.grad[:, mask_outweights] = 0.0
                    return loss
                optimizer_outliners_data.step(closure)
                gradients += 2
            else:
                optimizer_outliners_data.step()
                gradients += 1

        if optimizer_outliners_features: #FIXME
            optimizer_outliners_features.zero_grad()
            loss = loss_fn(network(X_outliners), y_outliners) / len(train_dataset)
            loss += omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2)
            loss.backward()
            last_layer.weight.grad[:, ~mask_outweights] = 0.0
            if isinstance(optimizer_outliners_features, SAM):
                def closure():
                    loss = loss_fn(network(X_outliners), y_outliners) / len(train_dataset)
                    loss += omega_penalty(network, omega_wd_0, omega_wd_1, omega_wd_2)
                    loss.backward()
                    last_layer.weight.grad[:, ~mask_outweights] = 0.0
                    return loss
                optimizer_outliners_features.step(closure)
                gradients += 2
            else:
                optimizer_outliners_features.step()
                gradients += 1

        for layer in frozen_layers:
            layer.requires_grad = True

        if swa:
          if swa_model is None:
              swa_model = torch.optim.swa_utils.AveragedModel(network)
          else:
              swa_model.update_parameters(network)
              if swa_scheduler is not None:
                swa_scheduler.step()

        if ema:
            if ema_model is None:
              ema_model = torch.optim.swa_utils.AveragedModel(network, 
                                                              torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay), 
                                                              use_buffers=True)
            else:
              ema_model.update_parameters(network)
    
    if log_epoch:
        norm_dict = {"train/running/weighted_norms": sum(weighted_norms)/len(weighted_norms),
                     "train/running/L0_norms": sum(L0_norms)/len(L0_norms),
                     "train/running/L1_norms": sum(L1_norms)/len(L1_norms),
                     "train/running/L2_norms": sum(L2_norms)/len(L2_norms)}
        if eliminate_outliners_data:
            wandb.log(dict({
                    "train/running/data_outliners_ratio_avg": sum(out_ratio)/len(out_ratio),
                    "train/running/data_outliners_ratio_his": wandb.Histogram(out_ratio),
                    "train/running/data_inliners_loss_avg": sum(in_loss)/len(in_loss),
                    "train/running/data_inliners_loss_his": wandb.Histogram(in_loss),
                    "train/running/data_inliners_acc_avg": sum(in_acc)/len(in_acc),
                    "train/running/data_inliners_acc_his": wandb.Histogram(in_acc),
                    "train/running/data_outliners_loss_avg": sum(out_loss)/len(out_loss),
                    "train/running/data_outliners_loss_his": wandb.Histogram(out_loss),
                    "train/running/data_outliners_acc_avg": sum(out_loss)/len(out_loss),
                    "train/running/data_outliners_acc_his": wandb.Histogram(out_loss),
                    "train/running/data_inliners_e1_avg": sum(in_e1)/len(in_e1),
                    "train/running/data_inliners_e1_his": wandb.Histogram(in_e1),
                    "train/running/data_outliners_e1_avg": sum(out_e1)/len(out_e1),
                    "train/running/data_outliners_e1_his": wandb.Histogram(out_e1)}, **norm_dict),
                    step=log_epoch)
        else:
            wandb.log(dict({
                    "train/running/data_X_loss_avg": sum(X_loss)/len(X_loss),
                    "train/running/data_X_loss_his": wandb.Histogram(X_loss),
                    "train/running/data_X_acc_avg": sum(X_acc)/len(X_acc),
                    "train/running/data_X_acc_his": wandb.Histogram(X_acc),
                    "train/running/data_X_e1_avg": sum(X_e1)/len(X_e1),
                    "train/running/data_X_e1_his": wandb.Histogram(X_e1)}, **norm_dict),
                    step=log_epoch)

    return gradients # , other
