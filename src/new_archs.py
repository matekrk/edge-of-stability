import abc
import math
from typing import List
import torch
from torch import nn
from new_archs_utils import *
    
class ClassificationNet(torch.nn.Module, abc.ABC):
    def __init__(self, input_shape, output_shape, softmax) -> None:
        super().__init__()
        w, h, c = input_shape
        assert w == h
        self.c = c
        self.w = w

        self.feature_extractor = nn.Sequential(nn.Linear(w*h*c, 1)) # dumb
        classifier = [nn.Linear(1, output_shape)] # dumb
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)
        self.softmax = softmax

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def forward_if(self, x: torch.Tensor, return_features: bool = False):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y
    
class FullyConnected(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax, widths: List[int], bn: bool, bias: bool, activation: str) -> None:
        super(FullyConnected, self).__init__(input_shape, output_shape, softmax)
        input_prod = eval('*'.join(str(item) for item in input_shape))
        modules = [nn.Flatten()]
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else input_prod
            block = [
                nn.Linear(prev_width, widths[l], bias=bias),
                get_activation(activation),
            ]
            if bn:
                block.append(nn.BatchNorm1d(widths[l]))
            modules.extend(block)
        self.feature_extractor = nn.Sequential(*modules)
        classifier = [nn.Linear(widths[-1], output_shape, bias=bias)]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)

        #self.last = self.classifier[-1]
        #self.gradcam = self.features[-3]

    def forward(self, x, return_features=False):
        return super().forward(x, return_features)
    
class Convnet(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax, widths: List[int], pooling: str, bn: bool, bias: bool, activation: str) -> None:
        super(Convnet, self).__init__(input_shape, output_shape, softmax)
        _CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}
        feature_extractor = []
        size = self.w
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else self.c
            module = [
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
            ]
            if bn:
                module.append(nn.BatchNorm2d(widths[l]))
            module.append(get_pooling(pooling))
            feature_extractor.extend(module)
            size //= 2
        # self.flatten = nn.Flatten()
        classifier = [nn.Flatten(), nn.Linear(widths[-1]*size*size, output_shape)]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier = nn.Sequential(*classifier)
        
        #self.last = self.classifier[-1]
        #self.gradcam = self.feature_extractor[-3]

    def forward_if(self, x, return_features=False):
        f = self.feature_extractor(x)
        # y = self.flatten(y)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y
    
    #def forward(self, x, return_features=False):
        #return super().forward(x, return_features)

class LeNet(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax, pooling):
        super(LeNet, self).__init__(input_shape, output_shape, softmax)

        def mini_conv_block(n_in, n_out, k_size, pad):
            return nn.Sequential(
                nn.Conv2d(n_in, n_out, k_size, padding=pad),
                nn.ReLU(),
                get_pooling(pooling)
            )

        self.feature_extractor = nn.Sequential(mini_conv_block(1, 6, 5, pad=2), 
                                               mini_conv_block(6, 16, 5, pad=0), 
                                               nn.Flatten(start_dim=1), 
                                               nn.Linear(16*5*5, 120), 
                                               nn.Linear(120, 84))
        classifier = [nn.Linear(84, output_shape)]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)

class SkipSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual
    
def r9_conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax):
        super(ResNet9, self).__init__(input_shape, output_shape, softmax)
        
        self.feature_extractor = nn.Sequential(
            r9_conv_block(self.c, 64), # conv1
            r9_conv_block(64, 128, pool=True), # conv2
            SkipSequential(r9_conv_block(128, 128), r9_conv_block(128, 128)), # res 1
            r9_conv_block(128, 256, pool=True), # conv3
            r9_conv_block(256, 512, pool=True), # conv4
            SkipSequential(r9_conv_block(512, 512), r9_conv_block(512, 512)) # res 2
        )
        
        classifier = [
            nn.AdaptiveMaxPool2d((1,1)), 
            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(512, output_shape)
        ]
    
        if softmax:
            classifier.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*classifier)
        
        #self.last = self.classifier
        #self.gradcam = self.res2[-1]
        
    def forward_if(self, x, return_features=False):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y
    
    #def forward(self, x, return_features=False):
        #return super().forward(x, return_features)

def conv3x3(in_planes, out_planes, stride=1, bn=False, relu=False):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)

def conv1x1(in_planes, out_planes, stride=1, bn=False, relu=False):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax, block, layers):
        super(ResNet, self).__init__(input_shape, output_shape, softmax)

        self.num_layers = sum(layers)
        self.inplanes = 16
        self.latent_size = 64
        self.layer0 = nn.Sequential(conv3x3(self.c, self.inplanes), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.latent_size, layers[2], stride=2)

        self.feature_extractor = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, nn.AdaptiveAvgPool2d((1, 1)))

        classifier = [nn.Flatten(), nn.Linear(self.latent_size, output_shape)]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)
        # self.last = self.fc
        # self.gradcam = self.layer3[-1]

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_features=False):
        f = self.feature_extractor(x)
        # y = f.view(f.size(0), -1)
        y = self.classifier(y)
        if return_features:
            return y, f
        return y
    
class MobileNetV1(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax):
        super(MobileNetV1, self).__init__(input_shape, output_shape, softmax)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.feature_extractor = nn.Sequential(
            conv_bn(self.c, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        classifier = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, output_shape)]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x, return_feature=False):
        f = self.feature_extractor(x)
        # x = x.view(-1, 1024)
        y = self.classifier(f)
        if return_feature:
            return y, f
        return y

def dwise_conv(ch_in, stride=1):
    return nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(ClassificationNet):
    def __init__(self, input_shape, output_shape, softmax):
        super(MobileNetV2, self).__init__(input_shape, output_shape, softmax)
        w, h, c = input_shape # or c,w,h
        size = w # assert w == h

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.latent_size = 1280

        self.stem_conv = conv3x3(c, 32, stride=2)
        layers = [self.stem_conv]
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c
        self.last_conv = conv1x1(input_channel, self.latent_size)
        layers.append(self.last_conv)
        # self.layers = nn.Sequential(*layers)
        self.feature_extractor = nn.Sequential(*layers, )

        classifier = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout2d(0.2),
            nn.Linear(self.latent_size, output_shape)
        ]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
            
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x, return_features=False):
        # x = self.stem_conv(x)
        # x = self.layers(x)
        # x = self.last_conv(x)
        # x = self.avg_pool(x).view(-1, 1280)
        f = self.feature_extractor(x)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y
    
    # def forward(self, x, return_features=False):
        # return super().forward(x, return_features)
    
class SELayer(nn.Module):

	def __init__(self, inplanes, isTensor=True):
		super(SELayer, self).__init__()
		if isTensor:
			# if the input is (N, C, H, W)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
			)
		else:
			# if the input is (N, C)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Linear(inplanes, inplanes // 4, bias=False),
				nn.BatchNorm1d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Linear(inplanes // 4, inplanes, bias=False),
			)

	def forward(self, x):
		atten = self.SE_opr(x)
		atten = torch.clamp(atten + 3, 0, 6) / 6
		return x * atten

class HS(nn.Module):

	def __init__(self):
		super(HS, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class Shufflenet(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, ksize, stride, activation, useSE):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw-linear
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, stride, activation, useSE):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class ShuffleNet(ClassificationNet): # so called ShuffleNetV2_Plus
    def __init__(self, input_shape, output_shape, softmax, architecture=None, model_size='Large'):
        super(ShuffleNet, self).__init__(input_shape, output_shape, softmax)

        size = self.w

        assert size % 32 == 0
        assert architecture is not None

        self.latent_size = 1280
        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, self.latent_size]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, self.latent_size]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, self.latent_size]
        else:
            raise NotImplementedError


        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = [
            nn.Conv2d(self.c, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        ]

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    print('Shuffle3x3')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 1:
                    print('Shuffle5x5')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 2:
                    print('Shuffle7x7')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 3:
                    print('Xception')
                    self.features.append(Shuffle_Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)

        self.conv_last = [
            nn.Conv2d(input_channel, self.latent_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_size),
            HS(),
            nn.AvgPool2d(7),
            SELayer(1280)
        ]

        self.feature_extractor = nn.Sequential(*(self.first_conv + self.features + self.conv_last))

        classifier = [
            nn.Flatten(), 
            nn.Linear(1280, 1280, bias=False),
            HS(),
            nn.Dropout(0.2), 
            nn.Linear(1280, output_shape, bias=False)
        ]
        
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)

        self._initialize_weights()

    def forward(self, x, return_features=False):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y
    
    # def forward(self, x, return_features=False):
        # return super().forward(x, return_features)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VGG(ClassificationNet):
    '''
    VGG model
    '''
    def __init__(self, input_shape, output_shape, softmax, cfg):
        super(VGG, self).__init__(input_shape, output_shape, softmax)

        self.feature_extractor = self.make_layers(cfg)
        classifier = [
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, output_shape),
        ]
        if softmax:
            classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*classifier)
        
        #self.last = self.classifier[-1]
        #self.gradcam = self.features[-3]
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.c
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

    def forward(self, x, return_features = False):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        if return_features:
            return y, f
        return y

def load_network(arch_id, input_shape, output_shape, softmax, **kwargs):

    dict_name_class = {
        "fc": FullyConnected,
        "conv": Convnet,
        "lenet": LeNet,
        "resnet9": ResNet9,
        # "resnet18": ResNet18,
        "resnet32": ResNet,
        "mobilenet": MobileNetV1,
        "mobilenet2": MobileNetV2,
        "shufflenet": ShuffleNet,
        "vgg11": VGG,
        "vgg16": VGG,
        "vgg19": VGG,
    }

    net_kwargs = kwargs.copy()

    if arch_id.startswith("lenet"):
        net_kwargs["pooling"] = "max"

    dict_resnet = {
        "20": [3, 3, 3],
        "32": [5, 5, 5],
        "44": [7, 7, 7],
        "56": [9, 9, 9],
        "110": [18, 18, 18],
        "1202": [200, 200, 200]
    }
    if arch_id.startswith("resnet") and arch_id != "resnet9":
        net_kwargs["layers"] = dict_resnet[arch_id[6:]]
        net_kwargs["block"] = BasicBlock

    dict_vgg = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    }
    if arch_id.startswith("vgg"):
        net_kwargs["cfg"] = dict_vgg[arch_id[3:]]

    arch_default_shufflenet = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    if arch_id.startswith("shufflenet"):
        net_kwargs["architecture"] = arch_default_shufflenet
        net_kwargs["model_size"] = "Large"    

    return dict_name_class[arch_id](input_shape, output_shape, softmax, **net_kwargs)
