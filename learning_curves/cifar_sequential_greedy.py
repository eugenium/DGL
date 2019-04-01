from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torchvision.datasets.cifar import CIFAR10
from random import randint
import datetime
import itertools
import time
from models import auxillary_classifier2, Net

#### Some helper functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
            
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
                        
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



#####

def lr_scheduler(lr_0,epoch):
    lr = lr_0*0.2**(epoch // 15)
    return lr
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--name', default='',type=str,help='name')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


##################### Logs
time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
name_log_txt=name_log_txt +'.log'

with open(name_log_txt, "a") as text_file:
    print(args, file=text_file)

def main():
    global args, best_prec1
    args = parser.parse_args()


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset_class = CIFAR10(root='.', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = CIFAR10(root='.', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)



    model = Net()
    model = model.cuda()
    
    ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    ############### Initialize all
    layer_optim = [None] * ncnn
    layer_lr = [0.1] * ncnn
    for n in range(ncnn):
        to_train = itertools.chain(model.main_cnn.blocks[n].parameters(),
                                           model.auxillary_nets[n].parameters())
        layer_optim[n] = optim.SGD(to_train, lr=layer_lr[n], momentum=0.9,  
                                  weight_decay=5e-4)

######################### Lets do the training
    criterion = nn.CrossEntropyLoss().cuda()
    for n in range(ncnn):
        for epoch in range(1, args.epochs+1):
            # Make sure we set the bn right
            model.train()
            for k in range(n):
                #For sequential greedy we need to make
                # sure the bottom layers are frozen including batch norm statistics
                model.blocks[k].eval()

            #For each epoch let's store each layer individually
            batch_time = AverageMeter()
            batch_time_total = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()


            layer_lr[n] = lr_scheduler( 0.1, epoch-1)
            for param_group in layer_optim[n].param_groups:
                param_group['lr'] = layer_lr[n]
            end = time.time()

            for i, (inputs, targets) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                targets = targets.cuda(non_blocking = True)
                inputs = inputs.cuda(non_blocking = True)
                inputs = torch.autograd.Variable(inputs)
                targets = torch.autograd.Variable(targets)

                #Main loop
                representation = inputs
                end_all = time.time()
                for k in range(n):
                    # Forward only
                    outputs, representation = model(representation, n=k)

                representation = representation.detach()

                outputs, representation = model(representation, n=n)
                layer_optim[n].zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                layer_optim[n].step()

                # measure accuracy and record loss
                # measure elapsed time
                batch_time.update(time.time() - end)

                prec1 = accuracy(outputs.data, targets)
                losses.update(float(loss.data[0]), float(inputs.size(0)))
                top1.update(float(prec1[0]), float(inputs.size(0)))
            ##### evaluate on validation set
            top1test = validate(val_loader, model, criterion, epoch, n)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                      .format(n+1, epoch, losses.avg, top1.avg, top1test), file=text_file)

def validate(val_loader, model, criterion, epoch, n):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    all_targs = []
    model.eval()

    end = time.time()
    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input = torch.autograd.Variable(input)
            target = torch.autograd.Variable(target)

            representation = input
            output, _ = model(representation, n=n, upto=True)


            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(float(loss.data[0]), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            total += input.size(0)
        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))


    return top1.avg


if __name__ == '__main__':
    main()
