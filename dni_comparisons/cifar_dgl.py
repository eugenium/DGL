from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import dni
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torchvision.datasets.cifar import CIFAR10
from random import randint
import datetime
import itertools
import time
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
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--context', action='store_true', default=False,
                    help='enable context (label conditioning) in DNI')
parser.add_argument('--name', default='',type=str,help='name')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#Description from DNI paper
#CIFAR-10 CNN The hidden layers are all convolutional
#layers with 128 5 × 5 filters with resolution preserving
#padding, followed by batch-normalisation, ReLU and 3×3
#spatial max-pooling in the first layer and avg-pooling in
#the remaining ones. The synthetic gradient model has two
#hidden layers with 128 5×5 filters with resolution preserving padding, batch-normalisation and ReLU, followed by
#a final 128 5 × 5 filter convolutional layer with resolution
#preserving padding.
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



class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])

        self.blocks.append(nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((3,3))
        ))
        self.blocks.append(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d((3, 3))
        ))
        self.blocks.append(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d((3, 3))
        ))
        
        self.auxillary_nets.append(ConvSynthesizer())
        self.auxillary_nets.append(ConvSynthesizer())
        self.auxillary_nets.append(nn.Sequential(View(128),nn.Linear(128,10)))
        self.main_cnn = rep(self.blocks)

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation


class ConvSynthesizer(nn.Module):
    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.hidden = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.output = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.output.weight, 0)
        self.fc = nn.Linear(128,10)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
    def forward(self, trigger):
        x = self.input_trigger(trigger)
        x = self.hidden(F.relu(self.bn1(x)))
        x = self.output(F.relu(self.bn2(x)))
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = x.view(-1,128)
        return self.fc(x)






##################### Logs
time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
name_log_txt=name_log_txt +'.log'

with open(name_log_txt, "a") as text_file:
    print(args, file=text_file)

def main():
    global args, best_prec1
    args = parser.parse_args()


#### #### To simplify data parallelism we make an nn module with multiple outs
    model = Net()
    model = model.cuda()
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
    val_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    ############### Initialize all
    num_ep = args.epochs
    #ncnn = 3
    layer_optim = [None] * ncnn
    


######################### Lets do the training
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(num_ep):
        # Make sure we set the bn right
        model.train()
        #[auxillary_nets[n].train() for n in range(n_cnn)]

        #For each epoch let's store each layer individually
        batch_time = [AverageMeter() for _ in range(n_cnn)]
        batch_time_total = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(n_cnn)]
        top1 = [AverageMeter() for _ in range(n_cnn)]


        if epoch  == 0:
            for n in range(ncnn):
                to_train = itertools.chain(model.main_cnn.blocks[n].parameters(),
                                           model.auxillary_nets[n].parameters())
                layer_optim[n] = optim.Adam(to_train,lr=3e-5)#optim.SGD(to_train, lr=layer_lr[n],
                                  #         momentum=args.momentum,
                                   #        weight_decay=args.weight_decay)

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
            for n in range(ncnn):
                end = time.time()
                # Forward
                layer_optim[n].zero_grad()
                outputs, representation = model(representation, n=n)

                loss = criterion(outputs, targets)
                loss.backward()
                layer_optim[n].step()

                representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
                batch_time[n].update(time.time() - end)

                prec1 = accuracy(outputs.data, targets)
                losses[n].update(float(loss.data[0]), float(inputs.size(0)))
                top1[n].update(float(prec1[0]), float(inputs.size(0)))





        ##### evaluate on validation set
        top1test = validate(val_loader, model, criterion, epoch, ncnn-1)
        with open(name_log_txt, "a") as text_file:
            print("n: {}, epoch {}, loss: {:.5f}, train top1:{} test top1:{} "
                      .format(ncnn, epoch, losses[ncnn-1].avg, top1[ncnn-1].avg,top1test), file=text_file)


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
