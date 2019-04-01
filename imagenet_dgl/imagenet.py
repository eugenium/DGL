import argparse
import os
import shutil
import time
from collections import OrderedDict
import torch
import torch.optim as optimizer
import torch._utils
from functools import partial
import itertools


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from utils import AverageMeter, accuracy, convnet_half_precision,DataParallelSpecial
import json
import numpy as np
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



from random import randint
import datetime
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ds-type', default='maxpool', help="type of downsampling"
                                                         "Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=10,type=int, help='number of epochs')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--debug', default=0,type=int,help='debugging')
parser.add_argument('--start_epoch', default=1,type=int,help='which n we resume')
parser.add_argument('--save_folder', default='.',type=str,help='folder to save')
#related to mixed precision
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                         '--static-loss-scale.')
args = parser.parse_args()
best_prec1 = 0


################# Setup arguments
args.debug = args.debug > 0


if args.half:
    from fp16 import FP16_Optimizer
    from fp16.fp16util import  BN_convert_float
    if args.half:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
##################### Logs
time_stamp = str(datetime.datetime.now().isoformat())
name_log_txt = time_stamp + str(randint(0, 1000)) + args.name
name_log_txt=name_log_txt +'.log'

with open(name_log_txt, "a") as text_file:
    print(args, file=text_file)

def main():
    global args, best_prec1
    args = parser.parse_args()

    N_img = 224
    N_img_scale= 256

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(N_img),

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(N_img_scale),
            transforms.CenterCrop(N_img),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

#### #### To simplify data parallelism we make an nn module with multiple outs
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    model = nn.DataParallel(model)
    model = model.cuda()
    if args.half:
        model = model.half()
        model = BN_convert_float(model)
    ############### Initialize all
    num_ep = args.nepochs


############## Resume if we need to resume
    if (args.resume):
        name = args.resume
        model_dict = torch.load(name)
        model.load_state_dict(model_dict)
        print('model loaded')
######################### Lets do the training
    criterion = nn.CrossEntropyLoss().cuda()

    lr = args.lr
    to_train = itertools.chain(model.parameters())
    optim = optimizer.SGD(to_train, lr=lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    if args.half:
        optim = FP16_Optimizer(optim,
                               static_loss_scale=args.static_loss_scale,
                               dynamic_loss_scale=args.dynamic_loss_scale,
                               dynamic_loss_args={'scale_window': 1000})

    for epoch in range(args.start_epoch,num_ep+1):
        # Make sure we set the bn right
        model.train()

        #For each epoch let's store each layer individually
        batch_time_total = AverageMeter()
        data_time = AverageMeter()
        lossm = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()


        if epoch % args.epochdecay == 0:
                lr = lr / 10.0
                to_train = itertools.chain(model.parameters())
                optim = optimizer.SGD(to_train, lr=lr,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)
                if args.half:
                    optim = FP16_Optimizer(optim,
                                                    static_loss_scale=args.static_loss_scale,
                                                    dynamic_loss_scale=args.dynamic_loss_scale,
                                                    dynamic_loss_args={'scale_window': 1000})
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            targets = targets.cuda(non_blocking = True)
            inputs = inputs.cuda(non_blocking = True)
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)
            if args.half:
                inputs = inputs.half()

            end = time.time()

            # Forward
            optim.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            # update
            if args.half:
                optim.backward(loss)
            else:
                loss.backward()

            optim.step()

            # measure accuracy and record loss
            # measure elapsed time
            batch_time_total.update(time.time() - end)
            end = time.time()
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            lossm.update(float(loss.data[0]), float(inputs.size(0)))
            top1.update(float(prec1[0]), float(inputs.size(0)))
            top5.update(float(prec5[0]), float(inputs.size(0)))


            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time_total,
                    data_time=data_time, loss=lossm, top1=top1, top5=top5))


            if args.debug and i > 50:
                break


        top1test, top5test  = validate(val_loader, model, criterion, epoch)
        with open(name_log_txt, "a") as text_file:
            print("lr: {}, epoch {}, train top1:{}(top5:{}), "
                  "test top1:{} (top5:{})"
                  .format(lr, epoch, top1.avg, top5.avg,
                          top1test, top5test), file=text_file)

    #####Checkpoint
        if not args.debug:
            torch.save(model.state_dict(), args.save_folder + '/' + \
                   name_log_txt + '_current_model.t7')


    ############Save the final model
    torch.save(model.state_dict(), args.save_folder + '/' + name_log_txt + '_model.t7')


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    with torch.no_grad():
        total = 0
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input = torch.autograd.Variable(input)
            target = torch.autograd.Variable(target)
            if args.half:
                input = input.half()

            # compute output
            output = model(input)


            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(float(loss.data[0]), float(input.size(0)))
            top1.update(float(prec1[0]), float(input.size(0)))
            top5.update(float(prec5[0]), float(input.size(0)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

            total += input.size(0)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
