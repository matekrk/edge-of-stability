from typing import List

import torch
import torch.nn as nn

from resnet_cifar import resnet9, resnet32
from vgg import vgg11_nodropout, vgg11_nodropout_bn
from lenet import lenet_mnist, lenet_mnist2
from data_generic import num_classes, num_input_channels, image_size, num_pixels

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

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


def fully_connected_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)

class Fully_connected_net(nn.Module):
    def __init__(self, dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> None:
        super(Fully_connected_net, self).__init__()
        self.flatten = nn.Flatten()
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
            features.extend([
                nn.Linear(prev_width, widths[l], bias=bias),
                get_activation(activation),
            ])
        classifier = [nn.Linear(widths[-1], num_classes(dataset_name), bias=bias)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-2]

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

def fully_connected_net_bn(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
            nn.BatchNorm1d(widths[l])
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)

class Fully_connected_net_bn(nn.Module):
    def __init__(self, dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> None:
        super(Fully_connected_net_bn, self).__init__()
        self.flatten = nn.Flatten()
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
            features.extend([
                nn.Linear(prev_width, widths[l], bias=bias),
                get_activation(activation),
                nn.BatchNorm1d(widths[l])
            ])
        classifier = [nn.Linear(widths[-1], num_classes(dataset_name), bias=bias)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-3]

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

def convnet(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)

class Convnet(nn.Module):
    def __init__(self, dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool = True) -> None:
        super(Convnet, self).__init__()
        size = image_size(dataset_name)
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
            features.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                get_pooling(pooling),
            ])
            size //= 2
        self.flatten = nn.Flatten()
        classifier = [nn.Linear(widths[-1]*size*size, num_classes(dataset_name))]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-3]

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def convnet_bn(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            nn.BatchNorm2d(widths[l]),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)

class Convnet_bn(nn.Module):
    def __init__(self, dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool = True) -> None:
        super(Convnet_bn, self).__init__()
        size = image_size(dataset_name)
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
            features.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                nn.BatchNorm2d(widths[l]),
                get_pooling(pooling),
            ])
            size //= 2
        self.flatten = nn.Flatten()
        classifier = [nn.Linear(widths[-1]*size*size, num_classes(dataset_name))]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-4]

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def make_deeplinear(L: int, d: int, seed=8):
    torch.manual_seed(seed)
    layers = []
    for l in range(L):
        layer = nn.Linear(d, d, bias=False)
        nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
    network = nn.Sequential(*layers)
    return network.cuda()

def make_one_layer_network(h=10, seed=0, activation='tanh', sigma_w=1.9):
    torch.manual_seed(seed)
    network = nn.Sequential(
        nn.Linear(1, h, bias=True),
        get_activation(activation),
        nn.Linear(h, 1, bias=False),
    )
    nn.init.xavier_normal_(network[0].weight, gain=sigma_w)
    nn.init.zeros_(network[0].bias)
    nn.init.xavier_normal_(network[2].weight)
    return network


def load_architecture(arch_id: str, dataset_name: str, dynamic=False) -> nn.Module:
    #  ======   fully-connected networks =======
    if arch_id.startswith("fc"):
        if arch_id[2:].startswith("-depth"): # ======= vary depth =======
            depth = int(arch_id[8])
            widths = [200] * depth
            activation = arch_id[10:]
        else:
            widths = [200, 200]
            activation = arch_id[3:]

        return Fully_connected_net(dataset_name, widths, activation, bias=True)

        """
        if arch_id == 'fc-relu':
            return Fully_connected_net(dataset_name, [200, 200], 'relu', bias=True)
        elif arch_id == 'fc-elu':
            return Fully_connected_net(dataset_name, [200, 200], 'elu', bias=True)
        elif arch_id == 'fc-tanh':
            return Fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
        elif arch_id == 'fc-hardtanh':
            return Fully_connected_net(dataset_name, [200, 200], 'hardtanh', bias=True)
        elif arch_id == 'fc-softplus':
            return Fully_connected_net(dataset_name, [200, 200], 'softplus', bias=True)
        elif arch_id == "fc-swish":
            return Fully_connected_net(dataset_name, [200, 200], 'swish', bias=True)
        elif arch_id == "fc-gelu":
            return Fully_connected_net(dataset_name, [200, 200], 'gelu', bias=True)
        
        
        elif arch_id == 'fc-depth1-tanh':
            return Fully_connected_net(dataset_name, [200], 'tanh', bias=True)
        elif arch_id == 'fc-depth2-tanh':
            return Fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
        elif arch_id == 'fc-depth3-tanh':
            return fully_connected_net(dataset_name, [200, 200, 200], 'tanh', bias=True)
        elif arch_id == 'fc-depth4-tanh':
            return Fully_connected_net(dataset_name, [200, 200, 200, 200], 'tanh', bias=True)
        """

    #  ======   convolutional networks =======
    elif arch_id.startswith("cnn-bn"): #  ======   convolutional networks with BN =======
        # 'cnn-bn-avgpool-relu':
        if arch_id[6:].startswith("-avgpool"): # ======= average pooling =======
            activation = arch_id[15:]
            pooling = "average"
        else:
            activation = arch_id[7:]
            pooling = "max"

        return Convnet_bn(dataset_name, [32, 32], activation, pooling, bias=True)

    elif arch_id.startswith("cnn"):
        if arch_id[3:].startswith("-avgpool"): # ======= average pooling =======
            activation = arch_id[12:]
            pooling = "average"
        else:
            activation = arch_id[4:]
            pooling = "max"

        return Convnet(dataset_name, [32, 32], activation, pooling, bias=True)
        """
        elif arch_id == 'cnn-relu':
            return Convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
        elif arch_id == 'cnn-elu':
            return Convnet(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
        elif arch_id == 'cnn-tanh':
            return Convnet(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)
        elif arch_id == 'cnn-swish':
            return Convnet(dataset_name, [32, 32], activation='swish', pooling='max', bias=True)
        elif arch_id == 'cnn-gelu':
            return Convnet(dataset_name, [32, 32], activation='gelu', pooling='max', bias=True)
        elif arch_id == 'cnn-avgpool-relu':
            return Convnet(dataset_name, [32, 32], activation='relu', pooling='average', bias=True)
        elif arch_id == 'cnn-avgpool-elu':
            return Convnet(dataset_name, [32, 32], activation='elu', pooling='average', bias=True)
        elif arch_id == 'cnn-avgpool-tanh':
            return Convnet(dataset_name, [32, 32], activation='tanh', pooling='average', bias=True)
        elif arch_id == 'cnn-avgpool-swish':
            return Convnet(dataset_name, [32, 32], activation='swish', pooling='average', bias=True)
        elif arch_id == 'cnn-avgpool-gelu':
            return Convnet(dataset_name, [32, 32], activation='gelu', pooling='average', bias=True)

        
        elif arch_id == 'cnn-bn-relu':
            return Convnet_bn(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
        elif arch_id == 'cnn-bn-elu':
            return Convnet_bn(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
        elif arch_id == 'cnn-bn-tanh':
            return Convnet_bn(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)
        elif arch_id == 'cnn-bn-swish':
            return Convnet_bn(dataset_name, [32, 32], activation='swish', pooling='max', bias=True)
        elif arch_id == 'cnn-bn-gelu':
            return Convnet_bn(dataset_name, [32, 32], activation='gelu', pooling='max', bias=True)

        """

    #  ======   real networks on CIFAR-10  =======
    elif arch_id == 'resnet9':
        return resnet9()
    elif arch_id == 'resnet32':
        return resnet32()
    elif arch_id == 'vgg11':
        return vgg11_nodropout()
    elif arch_id == 'vgg11-bn':
        return vgg11_nodropout_bn()
    
    #  ======   real networks on MNIST  =======
    elif arch_id == 'lenet':
        return lenet_mnist(dynamic)
    elif arch_id == 'lenet2':
        return lenet_mnist2()

    # ====== additional networks ========
    # elif arch_id == 'transformer':
        # return TransformerModelFixed()
    elif arch_id == 'deeplinear':
        return make_deeplinear(20, 50)
    elif arch_id == 'regression':
        return make_one_layer_network(h=100, activation='tanh')
