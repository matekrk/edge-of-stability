import os
import torch

def load_checkpoint(network, model_path):
    assert os.path.isdir(model_path)
    network.load_state_dict(torch.load(model_path))
    print("Loaded model successfully")

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "swish":
        return torch.nn.SiLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))

def get_target_layers(arch_id, network):
    if arch_id == "lenet":
        return [network.feature_extractor[1][1]]
    elif arch_id == "resnet9":
        return [network.feature_extractor[-1][-1][-1]]

def get_last_layer(arch_id, network):
    softmax = int(network.softmax)
    if arch_id == "lenet":
        return f"classifier.{0+softmax}" # network.classifier[-1-softmax]
    elif arch_id == "resnet9":
        return f"classifier.{3+softmax}" # network.classifier[-1-softmax]

def get_init_dict(arch_id, network):
    softmax = int(network.softmax)
    if arch_id == "lenet":
        class_activations = {
            1: {},
            2: {},
            3: {},
            4: {}
        }
        return_nodes = {
                        'feature_extractor.0.0': 1,
                        'feature_extractor.1.0': 2,
                        'feature_extractor.3': 3,
                        'feature_extractor.4': 3,
                        f'classifier.{0+softmax}': 4,
                    }
    elif arch_id == "resnet9":

        class_activations = {
            1: {},
            2: {},
            3: {},
            4: {}, 
            5: {}
        }
        return_nodes = {
                        'feature_extractor.0.0': 1, 'feature_extractor.1.0': 1,
                        'feature_extractor2.0.0': 2, 'feature_extractor.2.1.0': 2,
                        'feature_extractor.3.0': 3, 'feature_extractor.4.0': 3,
                        'feature_extractor.5.0.0': 4, 'feature_extractor.5.1.0': 4,
                        f'classifier.{3+softmax}': 5
                    }

    return class_activations, return_nodes
