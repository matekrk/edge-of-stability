import inspect
import torch
import torch.optim as optim

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GeneralizedSAM(torch.optim.Optimizer):
    def __init__(self, params, opti_type = "sgd", **kwargs):
        if opti_type == 'sam':
            self.optimizer = SAM(params, optim.SGD, **kwargs)
        else:
            filtered_kwargs = {k:v for k,v in kwargs.items() if k in inspect.signature(optim.SGD.__init__).parameters}
            self.optimizer = optim.SGD(params, **filtered_kwargs)

        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults

    def step(self, closure=None):
        self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, main_type, main_lr, 
                 aux_type = None, aux_lr = None, data_type = None, data_lr = None, 
                 main_params = None, aux_params = None, data_params = None, 
                 **kwargs):
        super(CombinedOptimizer, self).__init__(params, defaults={})
        if main_params is None:
            main_params = params
        self.main_optimizer = GeneralizedSAM(main_params, opti_type=main_type, lr=main_lr, **kwargs)
        self.main_lr = main_lr
        if aux_lr is not None:
            if aux_params is None:
                aux_params = params
            if aux_type is None:
                aux_type = main_type
            self.aux_optimizer = GeneralizedSAM(aux_params, opti_type=aux_type, lr=aux_lr, **kwargs)
            self.aux_lr = aux_lr
        else:
            self.aux_optimizer = None
        if data_lr is not None:
            if data_params is None:
                data_params = params
            if data_type is None:
                data_type = main_type
            self.data_optimizer = GeneralizedSAM(data_params, opti_type=data_type, lr=data_lr, **kwargs)
            self.data_lr = data_lr
        else:
            self.data_optimizer = None

    def step(self, main=True, aux=True, data=True):
        if main:
            main_opti_params = list(self.main_optimizer.param_groups[0]['params'])
            main_grad = [x.grad for x in main_opti_params]
        if aux and self.aux_optimizer is not None:
            aux_opti_params = list(self.aux_optimizer.param_groups[0]['params'])
            aux_grad = [x.grad for x in aux_opti_params]
        if data and self.data_optimizer is not None:
            data_opti_params = list(self.data_optimizer.param_groups[0]['params'])
            data_grad = [x.grad for x in data_opti_params]
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if main:
                    p.data -= self.main_lr * main_grad[i]
                if aux and self.aux_optimizer is not None:
                    p.data -= self.aux_lr * aux_grad[i]
                if data and self.data_optimizer is not None:
                    p.data -= self.data_lr * data_grad[i]

    def get_lr(self, main=True, aux=False, data=False):
        if main:
            return self.main_lr
        elif aux:
            if self.aux_optimizer is None:
                return None
            return self.aux_lr
        elif data:
            if self.data_optimizer is None:
                return None
            return self.data_lr


def get_optimizer(params, opti_type, lr, aux_type, aux_lr, data_type, data_lr, momentum, dampening, wd, rho, adaptive):
    return CombinedOptimizer(params, opti_type, lr, aux_type, aux_lr, data_type, data_lr, None, None, None, momentum=momentum, dampening=dampening, weight_decay=wd, rho=rho, adaptive=adaptive)

def get_name_optimizer(opti_type, lr, aux_type, aux_lr, data_type, data_lr, momentum, dampening, wd, rho, adaptive):
    aux_str = "no_aux" if aux_lr is None else f"{opti_type if aux_type is None else aux_type}_{aux_lr}"
    data_str = "no_data" if data_lr is None else f"{opti_type if data_type is None else data_type}_{data_lr}"
    return opti_type + f"_{lr}" + "-" +  aux_str + "-" + data_str + "-" + f"m_{momentum}-d_{dampening}-wd_{wd}-rho_{rho}-a{'T' if adaptive else 'F'}"
