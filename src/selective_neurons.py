
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from utilities import iterate_dataset

def forward(input_batch, target, class_activations, return_nodes, feature_extractor): 
    out = feature_extractor(input_batch)
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
        activations = torch.mean(v.view(v.size(0), v.size(1), -1), dim=2).cpu().numpy()

        for i, activation in enumerate(activations): 
            activation = np.expand_dims(activation, axis=0)
            num = nums[k]
            if targets[i] not in class_activations[index]:
                class_activations[index].update({targets[i]: {}})  # ex: {layer_3: {class_0: {} } }
            
            if num in class_activations[index][targets[i]]:
                class_activations[index][targets[i]][num] += activation
            else:
                class_activations[index][targets[i]].update({num: activation})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
   
    return input_batch, class_activations 


def get_class_activations(network, dataset, batch_size):
    # format {module_part_of_net: {class: layer_within_this_module: {id_layer: activations}}}

    network.eval()

    counter = 0

    if network._get_name().startswith("VGG"):
        class_activations = {
            1: {},
            2: {},
            3: {},
            4: {}, 
            5: {}
        }
        return_nodes = {
                        'features.1': 1, 'features.3': 1, 
                        'features.5': 2, 'features.6': 2,
                        'features.8': 3, 'features.9': 3, 
                        'features.11': 4, 'features.12': 4,
                        'classifier.1': 5, 'classifier.4': 5
                    }
    elif network._get_name().startswith("ResNet"):
        class_activations = {
            1: {},
            2: {},
            3: {},
            4: {}, 
            5: {}
        }
        return_nodes = {
                        'conv1': 1,
                        'layer1.0.conv1': 2, 'layer1.1.conv2': 2, 'layer1.2.conv1': 2,
                        'layer2.0.conv1': 3, 'layer2.0.conv2': 3, 'layer2.1.conv1': 3,
                        'layer3.0.conv1': 4, 'layer3.0.conv2': 4, 'layer3.1.conv1': 4,
                        'fc': 5,
                    }
        
    elif network._get_name().startswith("Fully_connected"):
        class_activations = {
            1: {},
            2: {},
            3: {},
        }
        return_nodes = {
                        'features.0': 1,
                        'features.1': 2,
                        'classifier.0': 3,
                    }
        
    elif network._get_name().startswith("Convnet"):
        class_activations = {
            1: {},
            2: {},
            3: {},
        }
        return_nodes = {
                        'features.0': 1,
                        'features.3': 2,
                        'classifier.0': 3,
                    }

    feature_extractor = create_feature_extractor(network, list(return_nodes.keys()))

    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):

            input_batch, class_activations = forward(X, y, class_activations, return_nodes, feature_extractor)
            # if counter > 110:
            #     break
            counter += 1
        
        return class_activations

def get_class_selectivity(model, dataset, batch_size=1000, epsilon = 1e-6):
    
    class_activations = get_class_activations(model, dataset, batch_size)

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
