'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.representation_dim = 512
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(True),
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(True),
        )
        self.classi = nn.Sequential(
            nn.Linear(self.representation_dim, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        f = self.encoder(x)
        x = self.classi(f)
        # x = self.classifier(x)
        if return_features:
            return x, f
        return x

    def get_representation_dim(self):
        return self.representation_dim

class VGGNoDropout(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGGNoDropout, self).__init__()
        self.features = features
        self.representation_dim = 512
        self.encoder = nn.Sequential(
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(True),
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(True),
        )
        self.classi = nn.Sequential(
            nn.Linear(self.representation_dim, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, return_features = False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        f = self.encoder(x)
        x = self.classi(f)
        if return_features:
            return x, f
        return x

    def get_representation_dim(self):
        return self.representation_dim


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))

def vgg11_nodropout():
    """VGG 11-layer model (configuration "A")"""
    return VGGNoDropout(make_layers(cfg['A']))

def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg11_nodropout_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGGNoDropout(make_layers(cfg['A'], batch_norm=True))



def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))

def vgg11_no_dropout():
    """VGG 19-layer model (configuration "E")"""
    return VGGNoDropout(make_layers(cfg['A']))

def vgg19_no_dropout():
    """VGG 19-layer model (configuration "E")"""
    return VGGNoDropout(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))