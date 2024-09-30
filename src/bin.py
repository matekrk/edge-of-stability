def f(X, y):
                return loss_fn(network(X.unsqueeze(0)), y.unsqueeze(0))
            losses = torch.vmap(f)(X, y)
            grads = torch.vmap(torch.autograd.grad)(losses, list(network.parameters()), create_graph=True)
            grad_norms = [torch.vmap(torch.norm)(g) for g in grads] # torch.vmap(torch.norm)(grads)

def compute_grad_norm(loss, params):
    grads = parameters_to_vector(torch.autograd.grad(loss, inputs=params))
    # grads = torch.autograd.grad(loss, params, retain_graph=True)
    grad_norm = torch.norm(torch.stack([g.detach().norm() for g in grads]))
    return grad_norm

network.train()
        network.zero_grad()
        #for param in network.parameters():
        #    param.retains_grad = True

        def compute_grad(sample, target, model, loss_fn):
            sample = sample.unsqueeze(0)
            target = target.unsqueeze(0)
            prediction = model(sample)
            loss = loss_fn(prediction, target)
            return torch.func.grad(loss)
            # return torch.autograd.grad(loss, list(model.parameters()))

        def compute_sample_grads(data, targets, model, loss_fn):
            #vmapped_compute_grad = torch.vmap(compute_grad, in_dims=(0, 0, None, None))
            #sample_grads = vmapped_compute_grad(data, targets, model, loss_fn)
            vmapped_compute_grad = torch.vmap(torch.func.grad(loss_fn), in_dims=None)
            sample_grads = vmapped_compute_grad(model.parameters(), data, targets)
            return sample_grads
        
        def compute_grad_norm(grads):
            grads = [param_grad.detach().flatten() for param_grad in grads if param_grad is not None]
            norm = torch.cat(grads).norm()
            return norm
        
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            sample_grads = compute_sample_grads(X.cuda(), y.cuda(), network, loss_fn)
            grad_norms = [compute_grad_norm(grads) for grads in sample_grads]
            break

        #for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
        #    losses = loss_fn_ind(network(X.cuda()), y.cuda())
        #    # batch_gradient = parameters_to_vector(torch.autograd.grad(losses, inputs=network.parameters()))
        #    z = torch.vmap(compute_grad_norm, in_dims=(0, None))(losses, network.parameters())
        #    break
        #print(z)