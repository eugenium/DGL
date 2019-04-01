import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import itertools
import torchvision
import torchvision.transforms as transforms
import argparse

from model_greedy import *
from torch.autograd import Variable

from random import randint
import datetime
import json

from utils import onehot,naive_cross_entropy_loss,CIFAR10_Index

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ncnn',  default=6,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=15,type=int, help='number of epochs')
parser.add_argument('--avg_size',  default=16,type=int, help='size of the averaging')
parser.add_argument('--feature_size',  default=128,type=int, help='size of the averaging')
parser.add_argument('--ds-type', default='maxpool', help="type of downsampling. Defaults to old block_conv with psi. Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--nlin',  default=0,type=int, help='nlin')
parser.add_argument('--ensemble', default=1,type=int,help='ensemble')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--batch_size', default=128,type=int,help='boost')
parser.add_argument('--bn', default=1,type=int,help='debug')
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='debug')
parser.add_argument('--prog', default=1,type=int,help='debug')
parser.add_argument('--mlp', default=3,type=int,help='debug')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--width_aux', default=256,type=int,help='dilate')
parser.add_argument('--buffer', default=0,type=int,help='buff?')
parser.add_argument('--down', default='[1,3]', type=str,
                        help='layer at which to invertible downsample')
parser.add_argument('--decay', default='[]', type=str,
                    help='list of decay')
parser.add_argument('--noise', default=0.0, type=float,
                    help='proba to drop a sample at layer n to simulate issues in communication')
parser.add_argument('--layer_noise', default=0, type=int,
                    help='proba to drop a sample at layer n to simulate issues in communication')
parser.add_argument('--buffer-sampling', default='priority_lifo')
parser.add_argument('--layer-sequence', default='random')

parser.add_argument("--max-buffer-reuse", type=int, default=np.inf)

parser.add_argument("--seed", type=int, default=None)

args = parser.parse_args()
opts = vars(args)
args.ensemble = args.ensemble>0
args.bn = args.bn > 0
args.prog = args.prog > 0
if args.debug:
    args.nepochs = 1 # we run just one epoch per greedy layer training

downsample =  list(np.array(json.loads(args.down)))
use_cuda = torch.cuda.is_available()
n_cnn = args.ncnn

cudnn.benchmark = True
if args.ds_type is None:
    block_conv_ = block_conv
else:
    from functools import partial

    block_conv_ = partial(ds_conv, ds_type=args.ds_type)


if args.seed is not None:
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
else:
    raise "seed is None"

############################################### Logging

buffer_sampling = args.buffer_sampling


save_name = 'layersize_'+str(args.feature_size)+'width_' \
            + str(args.width_aux) + 'depth_' + str(args.nlin) + 'ds_type_' + str(args.ds_type) +'down_'\
            +  args.down+'buff_'+str(args.buffer)+'buffer_sampling_'+buffer_sampling\
            +   'noise_'+str(args.noise)+'layernoise_'+str(args.layer_noise)\
            + f'layer_sequence_{args.layer_sequence}max_buffer_reuse{args.max_buffer_reuse}_'
#logging
time_stamp = str(datetime.datetime.now().isoformat())
time_stamp = time_stamp[:-10]

name_log_dir = ''.join('{}{}-'.format(key, val) for key, val in sorted(opts.items()))+time_stamp
name_log_dir = 'runs/'+name_log_dir

name_log_txt = time_stamp + save_name + str(randint(0, 1000)) + args.name
debug_log_txt = name_log_txt + '_debug.log'
name_save_model = name_log_txt + '.t7'
name_log_txt=name_log_txt   +'.log'

print(f"log file {name_log_txt}")
with open(name_log_txt, "a") as text_file:
    print(opts, file=text_file)
########################################################################################


########################################################33 Data setup
print('==> Preparing data..')
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

trainset_class = CIFAR10_Index(root='.', train=True, download=True,transform=transform_train)
trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


class Buffer(object):
    def __init__(self, buffer_size,n_buffer,bs, sample_method='priority_lifo'):
        if buffer_size == 0:
           buffer_size = 1

        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_labels = []
        self.buffer_counter = []
        self.b_counter = []
        self.bs=bs
        self.filled_in = []
        self.hist = []
        self.sample_method = sample_method
        for i in range(n_buffer):

            self.buffer.append(None)
            self.buffer_labels.append([None] * self.buffer_size)
            self.buffer_counter.append(0)
            self.filled_in.append(False)
            self.hist.append(torch.zeros(self.buffer_size))





    def add_sample(self,x,labels,n):
        if(self.buffer[n] is None):
            storage = torch.zeros((self.buffer_size, x.size(0), x.size(1), x.size(2), x.size(3)))
            self.buffer[n]=storage
        if(x.size(0)==self.bs):
            self.buffer[n][self.buffer_counter[n],...]=x
            self.buffer_labels[n][self.buffer_counter[n ]]=labels
            self.hist[n][self.buffer_counter[n]]=1
            self.buffer_counter[n]=(self.buffer_counter[n]+1) % self.buffer_size
            if(self.buffer_counter[n]==0):
                self.filled_in[n] = True
    def get_sample(self, n, max_reuse=np.inf):

        nmax = self.buffer_size if self.filled_in[n] else self.buffer_counter[n]
        use_counts = self.hist[n][:nmax]
        if (use_counts >= max_reuse).all():
            return None
        if self.buffer_size == 1:
            return self.buffer[n][0],self.buffer_labels[n][0]

        if self.sample_method == 'priority_lifo':
            # select all batches that have been used the least number of times
            # use the freshest one among them
            buf_data = self.buffer[n][:nmax]
            buf_labels = self.buffer_labels[n][:nmax]
            min_use_count = use_counts.min()
            eligible_batches = np.where(use_counts == min_use_count)[0]
            # set everything after current batch pointer to corresponding negative number
            # which is still the correct index, just from the right,
            # then use the maximal index found, which
            # is the most recently added
            eligible_batches[eligible_batches > self.buffer_counter[n]] -= nmax  # not sure whether to subtract this or self.buffer_size :/ it shouldn't matter, because nmax == self.buffer_size when filled in, but before that nothing is filled in after nmax
            selected_batch_index = eligible_batches.max()
            use_counts[selected_batch_index] += 1
            return buf_data[selected_batch_index], buf_labels[selected_batch_index]
	
# Model
def train_model():
    sys.stdout.flush()

    net = greedyNet(block_conv_,1,args.feature_size,downsample=downsample,batchnorm=args.bn)
    net_c = auxillary_classifier(avg_size=args.avg_size, in_size=32,
                                 n_lin=args.nlin, feature_size=args.width_aux,
                                 input_features=args.feature_size, mlp_layers=args.mlp, batchn=args.bn)
    sys.stdout.flush()
    auxillary_nets = [nn.DataParallel(net_c).cuda()]
    with open(name_log_txt, "a") as text_file:
        print(net, file=text_file)
    print("just after first opening of log file")
    sys.stdout.flush()
############### Initialize all
    layer_epoch = [0] * n_cnn
    layer_lr = [args.lr * 5.0] * n_cnn
    layer_optim = [None] * n_cnn
    in_size = 32
    for n in range(n_cnn):
        if args.down and n in downsample:
            args.avg_size = int(args.avg_size / 2)
            in_size = int(in_size / 2)
            args.feature_size = int(args.feature_size * 2)
            if args.prog:
                args.width_aux = args.width_aux * 2

        net_c = None
        if n < n_cnn - 1:
            net_c = auxillary_classifier(avg_size=args.avg_size, in_size=in_size,
                                      n_lin=args.nlin, feature_size=args.width_aux,
                                      input_features=args.feature_size, mlp_layers=args.mlp, batchn=args.bn).cuda()

            auxillary_nets.append(nn.DataParallel(net_c).cuda())
            net.add_block(n in downsample)
            with open(name_log_txt, "a") as text_file:
                print(net, file=text_file)
                print(net_c, file=text_file)

    if use_cuda:
        net = net.cuda()
    net.unfreezeAll() # we keep this for now for fine grained control if we need

    buffer = None


    to_train = True
    first_iter = True
    counter_first_iter = 0
    iteration_tracker = [0]*n_cnn
    epoch_tracker = [-1]*n_cnn
    epoch = -1
    epoch_finished = [False for _ in range(n_cnn)]
    train_loss = [0] * n_cnn
    correct = [0] * n_cnn
    total = [0] * n_cnn

    trainloader_classifier_iterator = iter(trainloader_classifier)
    n_layer = 0
    num_batch = len(trainloader_classifier)
    proba = torch.ones(n_cnn).float()
    if(args.noise>0):
        proba[args.layer_noise]=proba[args.layer_noise]-args.noise
    proba = proba*1.0/proba.sum()
    random_gen = torch.distributions.categorical.Categorical(probs=proba)
    continue_to_train = [True] * n_cnn
    epochdecay = False


    for n_layer in range(args.ncnn):
        layer_lr[n_layer] = layer_lr[n_layer] / 5.0 
        to_train = itertools.chain(net.blocks[n_layer].parameters(), auxillary_nets[n_layer].parameters())  
        layer_optim[n_layer] = optim.SGD(to_train, lr=layer_lr[n_layer], momentum=0.9, weight_decay=5e-4)

    buffer = Buffer(args.buffer,args.ncnn,args.batch_size,sample_method=buffer_sampling)
    
    while to_train:
        # First, we select a worker
        if first_iter:

            n_layer = counter_first_iter//2
            if(counter_first_iter>2*(n_cnn-1)):
                first_iter = False
            counter_first_iter = counter_first_iter + 1
        else:
            if args.buffer > 0 and args.layer_sequence != 'sequential':
                n_layer = random_gen.sample()
            else:
                n_layer = (n_layer + 1) % args.ncnn

        # Let's see if we should update the epoch
        if epoch_finished[n_layer]:
	    # Let's also see if we should update the current epoch
            if all(e >= epoch for e in epoch_tracker):
                # summarize previouss stats at training time:
                for n in range(args.ncnn):
                    if float(total[n]) == 0:
                        total[n] = 1
                acc = [100. * float(correct[n]) / float(total[n]) for n in range(args.ncnn)]

                #Evaluate
                if(epoch>0):
                    for n in range(n_cnn):
                        if args.ensemble:
                            acc_test, acc_test_ensemble = test(net, auxillary_nets[n], epoch, n, args.ensemble)
                            with open(name_log_txt, "a") as text_file:
                                print("n: {}, epoch {},epochtrue {}, train {}, test {},ense {} "
                                      .format(n, epoch, epoch_tracker[n], acc[n], acc_test, acc_test_ensemble), file=text_file)
                        else:
                            acc_test = test(net, auxillary_nets[n], epoch, n)
                            with open(name_log_txt, "a") as text_file:
                                print("n: {}, epoch {}, train {}, test {}, ".format(n, layer_epoch[n], acc[n],acc_test),file=text_file)
                # if the max amount of epoch is reached
                if epoch>args.nepochs:
                    to_train = False

                train_loss = [0] * n_cnn
                correct = [0] * n_cnn
                total = [0] * n_cnn

                # Let's go back in train mode
                net.train()
                [auxillary_nets[n].train() for n in range(n_cnn)]

                print('epoch ' + str(epoch))
                epoch = epoch + 1

#                # Let's see if we should update the LR
            if epoch_tracker[n_layer]>0 and epoch_tracker[n_layer] % args.epochdecay == 0:
                if epoch>0:
                    epochdecay = True
                    # Let's reset the buffers and the algorithm!
                first_iter = True
                counter_first_iter = 0
                       # for n in range(args.ncnn):
                layer_lr[n_layer] = layer_lr[n_layer] / 5.0
                to_train = itertools.chain(net.blocks[n_layer].parameters(), auxillary_nets[n_layer].parameters())  # list(filter(lambda p: p.requires_grad, net.parameters()))
                layer_optim[n_layer] = optim.SGD(to_train, lr=layer_lr[n_layer], momentum=0.9, weight_decay=5e-4)
                       # n_layer = 0
    
#

            epoch_tracker[n_layer] = epoch_tracker[n_layer] + 1
            if epoch_tracker[n_layer]>=args.nepochs:
                continue_to_train[n_layer] = False
            epoch_finished[n_layer]=False    

        if epochdecay:
            epochdecay=False
            continue

        representation = None
        # if this is the first worker, read from real data
        if n_layer==0:
            # test if we already empited the loop
            try:
                inputs, targets, batch_idx = next(trainloader_classifier_iterator)
            except StopIteration:
                trainloader_classifier_iterator = iter(trainloader_classifier)
                inputs, targets, batch_idx = next(trainloader_classifier_iterator)

            revised_targets = onehot(targets, 10)
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
            revised_targets = Variable(revised_targets.cuda()).detach()
            representation = inputs
            labels = revised_targets
        else:
            sample = buffer.get_sample(n_layer - 1, args.max_buffer_reuse)
            if sample is not None:
                representation, labels = sample
                representation = representation.cuda()
                labels = labels.cuda()
            else:
                print(f"Experiencing buffer overuse in layer {n_layer}, not processing")
                representation = None
                continue
        iteration_tracker[n_layer] = (iteration_tracker[n_layer] + 1) % num_batch
        if iteration_tracker[n_layer]==0:
            epoch_finished[n_layer]=True
        if representation is not None:
            if not continue_to_train[n_layer]:
                net.blocks[n_layer].eval()
                auxillary_nets[n_layer].eval()
    
    
                with torch.no_grad():
                    representation = net.forward(representation, n_layer)
                    outputs = auxillary_nets[n_layer](representation)
                    loss = naive_cross_entropy_loss(outputs, labels)
            else:         
                representation = net.forward(representation, n_layer)
                outputs = auxillary_nets[n_layer](representation)
                loss = naive_cross_entropy_loss(outputs, labels)
                # update
    
                layer_optim[n_layer].zero_grad()
                loss.backward()
                layer_optim[n_layer].step()
    
            # pass to next layer lets detach
            representation = representation.detach()
            if (n_layer<n_cnn-1):
                buffer.add_sample(representation.cpu(), labels.detach().cpu(),n_layer)
    
            # track statistics for each layer
            train_loss[n_layer] += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total[n_layer] += targets.size(0)
    
            _, labels_pred = torch.max(labels.data, 1)
    
            correct[n_layer] += predicted.eq(labels_pred).cpu().sum()
    


    return net,auxillary_nets[-1],acc[-1],acc_test

all_outs = [[] for i in range(args.ncnn)]
def test(net, auxillary_net, epoch, n, ensemble=False):
    global best_acc
    global acc_test_ensemble
    all_targs = []
    net.eval()
    auxillary_net.eval()
    test_loss = 0
    correct = 0
    total = 0

    criterion_classifier = nn.CrossEntropyLoss()

    all_outs[n] = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = auxillary_net(net(inputs, n, upto=True))

        loss = criterion_classifier(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if args.ensemble:
            all_outs[n].append(outputs.data.cpu())
            all_targs.append(targets.data.cpu())
    acc = 100. * float(correct) / float(total)

    if ensemble:
        all_outs[n] = torch.cat(all_outs[n])
        all_targs = torch.cat(all_targs)
        #This is all on cpu so we dont care
        weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
        total_out = torch.zeros((total,10))

        for i in range(n+1):
            total_out += float(weight[i])*all_outs[i]


        _, predicted = torch.max(total_out, 1)
        correct = predicted.eq(all_targs).sum()
        acc_ensemble = 100*float(correct)/float(total)


    if ensemble:
        return acc,acc_ensemble
    else:
        return acc


net, net_c, acc_test, acc_train = train_model()
state_final = {
            'net': net,
            'net_c': net_c,
            'acc_test': acc_test,
            'acc_train': acc_train,
        }
torch.save(state_final,save_name)
