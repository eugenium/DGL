# Decoupled Greedy Learning

This contains the source code for key experiments associated with the paper https://arxiv.org/abs/1901.08164. If you find this code helpful please cite our paper

@article{belilovsky2019decoupled,
  title={Decoupled Greedy Learning of CNNs},
  author={Belilovsky, Eugene and Eickenberg, Michael and Oyallon, Edouard},
  journal={arXiv preprint arXiv:1901.08164},
  year={2019}
}

In all experiments there is a log file which is generated in the local directory containing the per epoch training/val accuracy and other useful data. 


For questions or comments please contact at: eugene.belilovsky@umontreal.ca

## dni_comparisons
contains experiments used in Section 5
This relies on the package https://github.com/koz4k/dni-pytorch . Which is also copied in the directory for completness

end to end baseline

``
python cifar_cnn_dni.py 
``

DNI

``
python cifar_cnn_dni.py --dni
``

DNI with context

``
python cifar_cnn_dni.py --dni --context
``

DGL (our method)

``
python cifar_dgl.py 
``

## imagenet_dgl 
this contains our source code for the ImageNet experiments. Additionally this implementation provides a preliminary interface for specifying and learning greedy models, which will be released along with the paper.

VGG-13 K=10 layer by layer 

``
python imagenet_dgl.py IMAGENET_DIR  --arch vgg13 --block_size 1 --half --dynamic-loss-scale -j THREADS
``

VGG-13 K=4 

``
python imagenet_dgl.py IMAGENET_DIR  --arch vgg13 --block_size 3 --half --dynamic-loss-scale -j THREADS
``

VGG-19 K=4 

``
python imagenet_dgl.py IMAGENET_DIR  --arch vgg19 --block_size 4 --half --dynamic-loss-scale -j THREADS
``

VGG-19 K=2

``
python imagenet_dgl.py IMAGENET_DIR  --arch vgg19 --block_size 8 --half --dynamic-loss-scale -j THREADS
``

ResNet152 K=2

``
python imagenet_dgl.py IMAGENET_DIR  --arch resnet152 --half --dynamic-loss-scale -j THREADS
``

### run baseline (end-to-end) code

Resnet152

``
python imagenet.py IMAGENET_DIR  --arch resnet152 --half --dynamic-loss-scale -j THREADS
``

vgg13,vgg19. note: we use batchnorm versions of the pytorch baseline model repo for vgg, the models trained in our code above uses the batchnorm versions as well 

``
python imagenet.py IMAGENET_DIR  --arch vgg13_bn --half --dynamic-loss-scale -j THREADS
``

``
python imagenet.py IMAGENET_DIR  --arch vgg19_bn --half --dynamic-loss-scale -j THREADS
``


## ddg_comparisons
Comparison to DDG, separate README provided in directory

## auxiliary_nets_study
From Sec 5 Auxiliary nets

compares auxiliary networks, to evaluate MLP-SR-aux

```
python cifar_dgl.py --type_aux mlp-sr
```

CNN auxiliary is given by 'cnn' and MLP auxiliary by 'mlp'


## Learning curves
From Sec 5 Sequential vs Parallel. 
Contains notebook comparing the learnign curves for sequential greedy and dgl. To generate the logs run cifar_sequential_greedy.py for sequential and cifar_dgl.py for dgl with default arguments. 

## buffer_experiments
This repostory contains experiments associated with Section 5.3 that evaluate the asynchronous versions of DGL proposed in our work

An example can be run as follows with M=30, slowing down layer 3 (of 6 layers) by a factor 1.11
note: the slowdown factor reported in the paper is 1/(1-noise)

``
 python cifar_buffer.py --buffer 30 --noise 0.1 --layer_noise 3 --seed 0
``
