""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .auxillary_classifier import auxillary_classifier2

__all__ = [ 'vgg11', 'vgg13', 'vgg16','vgg19']

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class vgg_rep(nn.Module):
    def __init__(self, blocks):
        super(vgg_rep, self).__init__()
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

class vgg_greedy(nn.Module):
    def __init__(self, config_vgg, in_size=224,**kwargs):
        super(vgg_greedy, self).__init__()
        if kwargs['block_size']>1:
            self.make_layers_block(config_vgg,in_size,**kwargs)
        else:
            self.make_layers(config_vgg, in_size=in_size, **kwargs)
        self.blocks = nn.ModuleList(self.blocks)
        self.main_cnn = vgg_rep(self.blocks)
        self.auxillary_nets[len(self.auxillary_nets)-1] = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            View( 512*7*7),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096,1000),
        )
        self._initialize_weights()

    def make_layers(self, cfg, in_size, **kwargs):
        self.blocks = []
        self.auxillary_nets = nn.ModuleList([])
        in_channels = 3
        avg_size = 112
        last_M = False
        for v in cfg:
            if v == 'M':
                layer = [nn.MaxPool2d(kernel_size=2, stride=2)]
                avg_size = int(avg_size / 2)
                in_size = int(in_size / 2)
                last_M = True
                continue
            else:
                if last_M:
                    last_M=False
                else:
                    layer = []
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            in_channels = v
            self.blocks.append(nn.Sequential(*layer))


            model_c = auxillary_classifier2(in_size=in_size,
                                           n_lin=kwargs['nlin'], feature_size=in_channels,
                                           input_features=in_channels, mlp_layers=kwargs['mlp'],
                                           batchn=True, num_classes=1000).cuda()
            self.auxillary_nets.append(model_c)

    def make_layers_block(self, cfg, in_size, **kwargs):
        self.blocks = []
        self.auxillary_nets = nn.ModuleList([])
        block_size = kwargs['block_size']
        in_channels = 3
        avg_size = 112

        last_M = False
        layer = []
        block_current = 0
        for v in cfg:
            if v == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
                avg_size = int(avg_size / 2)
                in_size = int(in_size / 2)
                last_M = True
                continue
            else:
                if last_M:
                    last_M=False

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            block_current += 1
            in_channels = v
            if block_current == block_size:
                self.blocks.append(nn.Sequential(*layer))


                model_c = auxillary_classifier2(in_size=in_size,
                                           n_lin=kwargs['nlin'], feature_size=in_channels,
                                           input_features=in_channels, mlp_layers=kwargs['mlp'],
                                           batchn=True, num_classes=1000).cuda()
                self.auxillary_nets.append(model_c)
                block_current = 0
                layer = []

        if block_current ==1:
            self.blocks.append(nn.Sequential(*layer))
            model_c = auxillary_classifier2(in_size=in_size,
                                           n_lin=kwargs['nlin'], feature_size=in_channels,
                                           input_features=in_channels, mlp_layers=kwargs['mlp'],
                                           batchn=True, num_classes=1000).cuda()
            self.auxillary_nets.append(model_c)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0)

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def vgg19(**kwargs):
    model = vgg_greedy(cfg['E'], **kwargs)
    return model

def vgg16(**kwargs):
    model = vgg_greedy(cfg['D'], **kwargs)
    return model

def vgg11(**kwargs):
    model = vgg_greedy(cfg['A'], **kwargs)
    return model

def vgg13(**kwargs):
    model = vgg_greedy(cfg['B'], **kwargs)
    return model


