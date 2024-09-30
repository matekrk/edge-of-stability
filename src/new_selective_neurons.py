
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from new_archs_utils import get_init_dict, get_last_layer
from utilities import iterate_dataset

def forward_grad(features, targets, class_activations, return_nodes):
    vals = list(return_nodes.values()) 
    prev = vals[0] 
    counter = 0 
    nums = []
    for val in vals: 
        if val != prev: 
            counter = 0 
        
        nums.append(counter)
        counter += 1 
        prev = val 

    nums = dict(zip(return_nodes.keys(), nums))

    for k, v in features.items(): 
        # Key is the node name, value is the actual feature, i.e, activation output tensor  
        index = return_nodes[k] 
        activations = torch.mean(v.view(v.size(0), v.size(1), -1), dim=2)

        for i, activation in enumerate(activations): 
            activation = torch.unsqueeze(activation, dim=0)
            num = nums[k]

            if targets[i] not in class_activations[index]:
                class_activations[index].update({targets[i]: {}})  # ex: {layer_3: {class_0: {} } }
            
            if num in class_activations[index][targets[i]]:
                class_activations[index][targets[i]][num] += activation.cpu()
            else:
                class_activations[index][targets[i]].update({num: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
   
    return class_activations 

def forward(input_batch, target, class_activations, class_grads, return_nodes, feature_extractor, loss_fn, last_layer):
    feature_extractor.train()
    optim = torch.optim.SGD(feature_extractor.parameters(), lr=1.0)
    optim.zero_grad()
    out = feature_extractor(input_batch)
    out_last = out[last_layer]
    loss = loss_fn(out_last, target)
    loss.backward()

    vals = list(return_nodes.values()) 
    prev = vals[0] 
    counter = 0 
    nums = []
    for val in vals: 
        if val != prev: 
            counter = 0 
        
        nums.append(counter)
        counter += 1 
        prev = val 
    
    nums = dict(zip(return_nodes.keys(), nums))
    targets = torch.argmax(target, dim=1) if len(target.size()) == 2 else target
    targets = targets.cpu().numpy().tolist()

    for k, v in out.items(): 
        # Key is the node name, value is the actual feature, i.e, activation output tensor
        index = return_nodes[k]
        activations = torch.mean(v.detach().view(v.size(0), v.size(1), -1), dim=2).cpu().numpy()
        #g = dict(feature_extractor.named_modules())[k].weight.grad.detach()
        #gradient = torch.mean(g.view(g.size(0), g.size(1), -1), dim=2).cpu().numpy() 

        
        # if targets[i] #TODO: so far gradient independent of classes
        # class_grads[index][targets[i]].update({num: gradient})

        for i, activation in enumerate(activations): 
            activation = np.expand_dims(activation, axis=0)
            num = nums[k]
            if targets[i] not in class_activations[index]:
                class_activations[index].update({targets[i]: {}})  # ex: {layer_3: {class_0: {} } }
                #class_grads[index].update({targets[i]: {}})
            
            if num in class_activations[index][targets[i]]:
                class_activations[index][targets[i]][num] += activation
                #class_grads[index][targets[i]][num] += gradient
            else:
                class_activations[index][targets[i]].update({num: activation})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
                #class_grads[index][targets[i]].update({num: gradient})
   
    return input_batch, class_activations, class_grads


def get_class_activations(arch_id, network, dataset, batch_size, loss_fn):
    # format {module_part_of_net: {class: layer_within_this_module: {id_layer: activations}}}

    network.eval()

    counter = 0

    class_activations, return_nodes = get_init_dict(arch_id, network)
    class_grads, _ = get_init_dict(arch_id, network)
    feature_extractor = create_feature_extractor(network, list(return_nodes.keys()))
    last_layer = get_last_layer(arch_id, network)

    for (X, y) in iterate_dataset(dataset, batch_size):
        input_batch, class_activations, class_grads = forward(X, y, class_activations, class_grads, return_nodes, feature_extractor, loss_fn, last_layer)
        counter += 1

    return class_activations, class_grads

    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):

            input_batch, class_activations, class_grads = forward(X, y, class_activations, class_grads, return_nodes, feature_extractor, loss_fn)
            # if counter > 110:
            #     break
            counter += 1

            # X, y = X.to()
        
        return class_activations, class_grads
    
# TODO: potential batch_size 1 slow but for each data
# def get_grad_selectivity(model, dataset, loss_fn, batch_size = 1000, epsilon = 1e-6):
#     grad_

def get_class_selectivity(arch_id, model, dataset, loss_fn, batch_size=1000, epsilon = 1e-6, last_layer = None):
    
    class_activations, class_grads = get_class_activations(arch_id, model, dataset, batch_size, loss_fn)

    class_selectivity = {
        1: {},
        2: {},
        3: {},
        4: {}, 
        5: {}
    }

    neurons_predictions = {}

    for layer_k, layer_v in class_activations.items():
        if class_activations[layer_k]:
            neurons_predictions[layer_k] = {}
            for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
                for class_k in sorted(class_activations[layer_k].keys()):
                    if class_k > 0:
                        all_activations_for_this_bottleneck = np.concatenate((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), axis=0)
                    else:
                        all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
                
                all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.T

                u_max = np.max(all_activations_for_this_bottleneck, axis=1)
                neurons_predictions[layer_k][bottleneck_k] = np.argmax(all_activations_for_this_bottleneck, axis=1)
                u_sum = np.sum(all_activations_for_this_bottleneck, axis=1)
                u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

                selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)

                selectivity = np.clip(selectivity, a_min=0, a_max=1)
                
                class_selectivity[layer_k].update({bottleneck_k: selectivity})

    return class_selectivity, neurons_predictions, class_activations

def unpack_class_selectivity(class_selectivity):
    # format
    # {layer_k: {
    #               class_i: { bottleneck_k: selectivity}
    #           }
    # }
    return 
