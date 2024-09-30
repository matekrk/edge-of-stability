

from new_objective import compute_losses_dataloader

def generate_subnetworks(arch_id, network):
    pass

def verify_network(network, loss_fn, acc_fn, train_dataset, train_dataloader, test_datasets, test_dataloaders,
                    threshold_in = 0.7, threshold_out = 0.3):
    # performance in-domain
    # performance out-domain
    condition = False
    train_loss, train_acc = compute_losses_dataloader(network, [loss_fn, acc_fn], train_dataloader, no_grad=True)
    test_loss_acc_s = [compute_losses_dataloader(network, [loss_fn, acc_fn], test_dataloader, no_grad=True) for test_dataloader in test_dataloaders]
    if train_acc > threshold_in and all([x[1] < threshold_out for x in test_loss_acc_s]):
        condition = True
    return condition


