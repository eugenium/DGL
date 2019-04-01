Here we compare to the results of http://www.pitt.edu/~zhh39/others/icmlhuoa.pdf

models.py contains the ResNet-110 model and split locations as implemented in the authors code repository https://github.com/slowbull/DDG
and modified from https://github.com/slowbull/DDG/blob/master/models/resnet_ddg.py with wrappers to work in our framework


simply run cifar_dgl.py with default arguments. A log script with learning curves will be generated.

Default optimization settings are taken from authors paper and source repo.