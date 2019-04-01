import argparse
import os
import shutil
import time
from collections import OrderedDict
import torch
import torch.optim as optim
import torch._utils
from functools import partial
import itertools
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
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
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg_19',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
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
parser.add_argument('--ncnn',  default=10,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=10,type=int, help='number of epochs')
parser.add_argument('--feature_size',  default=64,type=int, help='size of the averaging')
parser.add_argument('--nlin',  default=3,type=int, help='SR')
parser.add_argument('--ensemble', default=1,type=int,help='ensemble') # not implemented yet
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--debug', default=0,type=int,help='debugging')
parser.add_argument('--large_size_images', default=1,type=int,help='use small image for dev')
parser.add_argument('--start_epoch', default=0,type=int,help='which n we resume')
parser.add_argument('--resume_epoch', default=0,type=int,help='which epoch we resume')
parser.add_argument('--resume_feat', default=0,type=int,help='dilate')
parser.add_argument('--mlp', default=3,type=int,help='mlp depth')
parser.add_argument('--block_size', default=1,type=int,help='how many layers in a block')
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
args.ensemble = args.ensemble>0
args.debug = args.debug > 0

device_ids = [i for i in range(torch.cuda.device_count())]
ids_aux = [i%len(device_ids) for i in range(args.ncnn)]


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


#### setup sizes and dataloaders
    if args.large_size_images==0:
        N_img = 112
        N_img_scale = 128
        print('using 112')
    elif args.large_size_images ==1:
        N_img = 224
        N_img_scale= 256

    in_size = N_img# // args.ds
    n_cnn = args.ncnn

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
    model = models.__dict__[args.arch](nlin=args.nlin,
                                       mlp=args.mlp, block_size=args.block_size).cuda()

    args.ncnn = len(model.main_cnn.blocks)
    n_cnn = len(model.main_cnn.blocks)
    with open(name_log_txt, "a") as text_file:
        print(model, file=text_file)
    if len(device_ids) == 1:
        model = nn.DataParallel(model) #single gpu mode, we do the DataParallle so we can still do .module later
    else:
        model = DataParallelSpecial(model)

    if args.half:
        model = model.half()
        model = BN_convert_float(model)
    ############### Initialize all
    num_ep = args.nepochs
    layer_epoch = [0] * n_cnn
    layer_lr = [args.lr * 10.0] * n_cnn
    layer_optim = [None] * n_cnn


############## Resume if we need to resume
    if (args.resume):
        name = args.resume
        model_dict = torch.load(name)
        model.load_state_dict(model_dict)
        print('model loaded')
######################### Lets do the training
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(args.start_epoch,num_ep):
        # Make sure we set the bn right
        model.train()

        #For each epoch let's store each layer individually
        batch_time = [AverageMeter() for _ in range(n_cnn)]
        batch_time_total = AverageMeter()
        data_time = AverageMeter()
        losses = [AverageMeter() for _ in range(n_cnn)]
        top1 = [AverageMeter() for _ in range(n_cnn)]
        top5 = [AverageMeter() for _ in range(n_cnn)]


        if epoch % args.epochdecay == 0 or (args.resume and epoch==args.start_epoch):
            for n in range(args.ncnn):
                layer_lr[n] = layer_lr[n] / 10.0
                to_train = itertools.chain(model.module.main_cnn.blocks[n].parameters(),
                                           model.module.auxillary_nets[n].parameters())
                layer_optim[n] = optim.SGD(to_train, lr=layer_lr[n],
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)
                if args.half:
                    layer_optim[n] = FP16_Optimizer(layer_optim[n],
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

            #Main loop
            if torch.cuda.device_count() >1:
                _,representation = model(inputs,init=True) #This only initializes the multi-gpu
            else:
                representation = inputs
            end_all = time.time()
            for n in range(args.ncnn):
                end = time.time()
                # Forward
                layer_optim[n].zero_grad()
                outputs, representation = model(representation, n=n)

                loss = criterion(outputs, targets)
                # update
                if args.half:
                    layer_optim[n].backward(loss)
                else:
                    loss.backward()

                layer_optim[n].step()

                # pass to next layer lets detach
                if torch.cuda.device_count() >1:
                    representation = [rep.detach() for rep in representation]
                else:
                    representation = representation.detach()
                # measure accuracy and record loss
                # measure elapsed time
                batch_time[n].update(time.time() - end)

                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
                losses[n].update(float(loss.data[0]), float(inputs.size(0)))
                top1[n].update(float(prec1[0]), float(inputs.size(0)))
                top5[n].update(float(prec5[0]), float(inputs.size(0)))


                if i % args.print_freq == 0:
                    print('n:{0} Epoch: [{1}][{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        n, epoch, i, len(train_loader), batch_time=batch_time[n],
                        data_time=data_time, loss=losses[n], top1=top1[n], top5=top5[n]))
                    if n == args.ncnn-1:
                        batch_time_total.update(time.time() - end_all)
                        str = '-'
                        for _ in range(10):
                            str+='-'
                        str+=' Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(epoch, i,
                                len(train_loader), batch_time=batch_time_total, data_time=data_time)
                        print(str)

            if args.debug and i > 50:
                break



        ##### evaluate on validation set
        for n in range(args.ncnn):
           # if args.debug:
            #    break
            top1test, top5test, top1ens, top5ens = validate(val_loader, model, criterion, epoch, n)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train top1:{}(top5:{}), "
                      "test top1:{} (top5:{}), top1ens:{} top5ens:{}"
                      .format(n, epoch, top1[n].avg, top5[n].avg,
                              top1test, top5test, top1ens, top5ens), file=text_file)

    #####Checkpoint
        if not args.debug:
            torch.save(model.state_dict(), args.save_folder + '/' + \
                   name_log_txt + '_current_model.t7')

    ############Save the final model
    torch.save(model.state_dict(), args.save_folder + '/' + name_log_txt + '_model.t7')



all_outs = [[] for n in range(args.ncnn)]
def validate(val_loader, model, criterion, epoch, n):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targs = []
    model.eval()

    end = time.time()
    all_outs[n] = []

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
            if len(device_ids)>1:
            	_, representation = model(input,init=True)
            else:
                representation = input
            output, _ = model(representation, n=n, upto=True)


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
            if args.ensemble:
                all_outs[n].append(F.softmax(output).data.float().cpu())
                all_targs.append(target.data.cpu())
            total += input.size(0)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if args.ensemble:
        all_outs[n] = torch.cat(all_outs[n])
        all_targs = torch.cat(all_targs)
        #This is all on cpu so we dont care

        weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
        total_out = torch.zeros([total, 1000])

        # very lazy
        for i in range(n + 1):
            total_out += float(weight[i]) * all_outs[i]

        prec1, prec5 = accuracy(total_out, all_targs, topk=(1, 5))

        print(' * Ensemble Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
              .format(top1=prec1[0], top5=prec5[0]))
        return top1.avg,top5.avg,prec1[0],prec5[0]
    return top1.avg, top5.avg,-1,-1


if __name__ == '__main__':
    main()
