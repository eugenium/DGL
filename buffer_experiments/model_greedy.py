import torch
import torch.nn as nn
import torch.nn.functional as F

class block_conv(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,downsample=False,batchn=True):
        super(block_conv, self).__init__()
        self.downsample = downsample
        if downsample:
            self.down = psi(2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if batchn:
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = identity()  # Identity

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class ds_conv(nn.Module):
    """ds_conv defaults to block_conv but can implement other downsamplings. They all have the same shape behavior"""
    
    def __init__(self, in_planes, planes, downsample=False, ds_type='psi', batchn=True):
        super(ds_conv, self).__init__()
        self.downsample = downsample
        self.ds_type = ds_type
        self.in_planes = in_planes
        self.planes = planes
        self.batchn = batchn

        self.build()

    def build(self):
        """Builds the forward model depending on the downsampler"""
        planes = self.planes
        in_planes = self.in_planes
        if self.batchn:
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = identity()

        if self.downsample is False:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=not self.batchn)
            self.conv_op = self.conv1
        elif self.ds_type == 'psi':
            self.down = psi(2)
            self.conv1 = nn.Conv2d(4 * in_planes, planes, kernel_size=3, stride=1, padding=1, bias=not self.batchn)
            self.conv_op = nn.Sequential(self.down, self.conv1)
        elif self.ds_type == 'stride':
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=not self.batchn)
            self.conv_op = self.conv1
        elif self.ds_type == 'maxpool':
            self.down = nn.MaxPool2d(2)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=not self.batchn)
            self.conv_op = nn.Sequential(self.down, self.conv1)
        elif self.ds_type == 'avgpool':
            self.down = nn.AvgPool2d(2)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=not self.batchn)
            self.conv_op = nn.Sequential(self.down, self.conv1)
        else:
            raise ValueError("I don't get {self.ds_type}. Only know False, True, 'psi', 'stride', 'maxpool', 'avgpool'")

    def forward(self, x):
        conv = self.conv_op(x)
        out = F.relu(self.bn1(conv))
        return out


class block_skip(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(block_skip, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.bn3(out)
        return out


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        """Expects x.shape == (batch, channel, height, width).
           Converts to (batch, channel, height / block_size, block_size, 
                                        width / block_size, block_size),
           transposes to put the two 'block_size' dims before channel,
           then reshapes back into (batch, block_size ** 2 * channel, ...)"""

        bs = self.block_size # lol, bs
        batch, channel, height, width = x.shape
        if ((height % bs) != 0) or (width % bs != 0):
            raise ValueError("height and width must be divisible by block_size")

        # reshape (creates a view)
        x1 = x.reshape(batch, channel, height // bs, bs, width // bs, bs)
        # transpose (also creates a view)
        x2 = x1.permute(0, 3, 5, 1, 2, 4)
        # reshape into new order (must copy and thus makes contiguous)
        x3 = x2.reshape(batch, bs ** 2 * channel, height // bs, width // bs)
        return x3





class auxillary_classifier(nn.Module):
    def __init__(self,avg_size=4,feature_size=256,
                 input_features=256, in_size=32,
                 num_classes=10,n_lin=0,mlp_layers=0,batchn=True):
        super(auxillary_classifier, self).__init__()
        self.n_lin=n_lin

        if n_lin==0:
            feature_size = input_features

        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            if batchn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = identity()

            self.blocks.append(nn.Sequential(nn.Conv2d(input_features, feature_size,
                                                       kernel_size=3, stride=1, padding=1, bias=False),bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        if batchn:
            self.bn = nn.BatchNorm2d(feature_size)
        else:
            self.bn = identity()  # Identity

        self.avg_size=avg_size
        if mlp_layers > 0:

            mlp_feat = feature_size * (in_size // avg_size) * (in_size // avg_size)
            layers = []
            for l in range(mlp_layers):
                if l==0:
                    in_feat = mlp_feat
                    mlp_feat = mlp_feat
                else:
                    in_feat = mlp_feat
                if batchn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = identity()
                layers +=[nn.Linear(in_feat,mlp_feat),bn_temp,nn.ReLU(True)]
            layers += [nn.Linear(mlp_feat,num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True

        else:
            self.mlp = False
            self.classifier = nn.Linear(feature_size*(in_size//avg_size)*(in_size//avg_size), num_classes)


    def forward(self, x):
        out = x
        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)
        if(self.avg_size>1):
            out = F.avg_pool2d(out, self.avg_size)
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input



class greedyNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=256, downsampling=1, downsample=[], batchnorm=True):
        super(greedyNet, self).__init__()
        self.in_planes = feature_size
        self.down_sampling = psi(downsampling)
        self.downsample_init = downsampling
        self.conv1 = nn.Conv2d(3 * downsampling * downsampling, self.in_planes, kernel_size=3, stride=1, padding=1,
                               bias=not batchnorm)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            self.bn1 = identity()  # Identity
        self.RELU = nn.ReLU()
        self.blocks = []
        self.block = block
        self.blocks.append(nn.Sequential(self.conv1, self.bn1, self.RELU))  # n=0
        self.batchn = batchnorm
        for n in range(num_blocks - 1):
            if n in downsample:
                if block in (block_conv, block_skip):
                    pre_factor = 4
                else:
                    pre_factor = 1
                self.blocks.append(block(self.in_planes * pre_factor, self.in_planes * 2,downsample=True, batchn=batchnorm))
                self.in_planes = self.in_planes * 2
            else:
                self.blocks.append(block(self.in_planes, self.in_planes,batchn=batchnorm))

        self.blocks = nn.ModuleList(self.blocks)
        for n in range(num_blocks):
            for p in self.blocks[n].parameters():
                p.requires_grad = False

    def unfreezeGradient(self, n):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = False

        for p in self.blocks[n].parameters():
            p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = True

    def add_block(self, downsample=False):
        if downsample:
            if self.block in (block_conv, block_skip):
                pre_factor = 4 # the old block needs this factor 4
            else:
                pre_factor = 1
            self.blocks.append(
                self.block(self.in_planes * pre_factor, self.in_planes * 2, downsample=True, batchn=self.batchn))
            self.in_planes = self.in_planes * 2
        else:
            self.blocks.append(self.block(self.in_planes, self.in_planes,batchn=self.batchn))

    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n

        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x

        if n==0 and self.downsample_init > 1:
            x = self.down_sampling(x)
        out = self.blocks[n](x)
        return out


