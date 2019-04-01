""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

__all__ = ['resnet152']

from .auxillary_classifier import auxillary_classifier2


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n

        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, split_points=2, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.avg_size = int(56 / 2)
        self.in_size = 56

        self.blocks = nn.ModuleList([])
        self.base_blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])
        self.auxillary_size_tracker = []

        ## Initial layer
        layer = [nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3,bias=False),
                 nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        self.base_blocks.append(nn.Sequential(*layer))
        self.auxillary_size_tracker.append((self.in_size,self.inplanes))

        self._make_layer(block, 64, layers[0], **kwargs)
        self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        len_layers = len(self.base_blocks)
        split_depth = math.ceil(len(self.base_blocks) / split_points)

        for splits_id in range(split_points):
            left_idx = splits_id * split_depth
            right_idx = (splits_id + 1) * split_depth
            if right_idx > len_layers:
                right_idx = len_layers
            self.blocks.append(nn.Sequential(*self.base_blocks[left_idx:right_idx]))
            in_size, planes = self.auxillary_size_tracker[right_idx-1]
            self.auxillary_nets.append(
                auxillary_classifier2(in_size=in_size,
                                      n_lin=kwargs['nlin'], feature_size=planes,
                                      input_features=planes, mlp_layers=kwargs['mlp'],
                                      batchn=True, num_classes=1000)
            )


        self.auxillary_nets[len(self.auxillary_nets)-1] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            View(512 * block.expansion),
            nn.Linear(512 * block.expansion, num_classes)
        )

        self.main_cnn = rep(self.blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            self.avg_size = int(self.avg_size/2)
            self.in_size =  int(self.in_size/2)


        self.base_blocks.append(block(self.inplanes, planes, stride, downsample))
        self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            self.base_blocks.append(block(self.inplanes, planes))
            self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation
    
def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
