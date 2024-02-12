#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.folder # imagenet
from sogclr.classification_model import ClassificationModel

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = torchvision_model_names

parser = argparse.ArgumentParser(description='SogCLR ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/data/imagenet100/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-m', '--model_path', 
                    default='./saved_models/20221005_cifar10_resnet50_sogclr-128-2048_bz_256_E50_WR10_lr_0.005_sqrt_wd_1e-06_t_0.1_g_0.9_lars/model_best.pth.tar', 
                    help='path to saved model')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--mlp-dim', default=2048, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--t', default=0.1, type=float,
                    help='softmax temperature (default: 1.0)')
parser.add_argument('--num_proj_layers', default=2, type=int,
                    help='number of non-linear projection heads')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# dataset 
parser.add_argument('--data_name', default='imagenet1000', type=str) 
parser.add_argument('--save_dir', default='./saved_models/', type=str) 


# sogclr
parser.add_argument('--learning-rate-scaling', default='sqrt', type=str,
                    choices=['sqrt', 'linear'],
                    help='learing rate scaling (default: sqrt)')


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # sizes for each dataset
    if args.data_name == 'cifar10': 
        data_size = 50020
    elif args.data_name == 'imagenet100' or args.data_name == 'imagenet50': 
        data_size = 129395+1
    elif args.data_name == 'imagenet1000': 
        data_size = 1281167+1 
    else:
        data_size = 1000000 

    if args.data_name == 'cifar10': 
        class_size = 10
    elif  args.data_name == 'imagenet50': 
        class_size = 50
    elif args.data_name == 'imagenet100':
        class_size = 100
    elif args.data_name == 'imagenet1000': 
        class_size = 1000
    else:
        raise ValueError

    print ('pretraining on %s'%args.data_name)

    # create model
    set_all_seeds(2022)
    print("=> creating model '{}'".format(args.arch))
    model = sogclr.builder.SimCLR_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.dim, args.mlp_dim, args.t, N=data_size, num_proj_layers=args.num_proj_layers)
    model.cuda()
    scaler = torch.cuda.amp.GradScaler()

    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        if args.gpu is None:
            checkpoint = torch.load(args.model_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.model_path, map_location=loc)
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        scaler.load_state_dict(checkpoint['scaler'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.model_path, checkpoint['epoch']))
        
        model_stage2 = ClassificationModel(model.base_encoder, num_classes=class_size)
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))


    # infer learning rate before changing batch size
    if args.learning_rate_scaling == 'linear':
        # infer learning rate before changing batch size
        args.lr = args.lr * args.batch_size / 256
    else:
        # sqrt scaling  
        args.lr = args.lr / math.sqrt(args.batch_size)
        
    print ('initial learning rate:', args.lr)      
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_stage2 = model_stage2.cuda(args.gpu)
        # comment out the following line for debugging

    if args.optimizer == 'lars':
        optimizer = sogclr.optimizer.LARS(model_stage2.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model_stage2.parameters(), args.lr,
                                weight_decay=args.weight_decay)
   
    # log_dir 
    save_root_path = args.save_dir
    global_batch_size = args.batch_size
    logdir = 'stage2_%s_cel-%s-%s_bz_%s_E%s_lr_%.3f_%s_wd_%s_t_%s_%s'%(args.data_name, args.dim, args.mlp_dim, global_batch_size, args.epochs, args.lr, args.learning_rate_scaling, args.weight_decay, args.t, args.optimizer )
    summary_writer = SummaryWriter(log_dir=os.path.join(save_root_path, logdir))
    print (logdir)
    
    cudnn.benchmark = True

    # Data loading code
    mean = {'cifar10':  [0.4914, 0.4822, 0.4465],
            'imagenet50':  [0.485, 0.456, 0.406],
            'imagenet100':  [0.485, 0.456, 0.406],
            'imagenet1000': [0.485, 0.456, 0.406],
            }[args.data_name]
    std = {'cifar10':   [0.2023, 0.1994, 0.2010],
           'imagenet50':   [0.229, 0.224, 0.225],
           'imagenet100':   [0.229, 0.224, 0.225],
            'imagenet1000': [0.229, 0.224, 0.225],
            }[args.data_name]

    image_size = {'cifar10':32, 'imagenet50':224, 'imagenet100':224, 'imagenet1000':224}[args.data_name]
    normalize = transforms.Normalize(mean=mean, std=std)

    # simclr
    augmentation1 = [
        transforms.RandomResizedCrop(image_size, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([sogclr.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if args.data_name == 'imagenet1000' or args.data_name == 'imagenet100' or args.data_name == 'imagenet50' or args.data_name == 'cifar10':
        traindir = os.path.join(args.data, 'train')
        train_dataset = sogclr.folder.ImageFolder(
            traindir,
            transforms.Compose(augmentation1))
    else:
        raise ValueError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    best_loss = 1000000000000

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        start_time = time.time()
        loss = train(train_loader, model_stage2, optimizer, scaler, summary_writer, epoch, args)
        print('elapsed time (s): %.1f'%(time.time() - start_time))

        if epoch % 10 == 0 or args.epochs - epoch < 3:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_stage2.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=os.path.join(save_root_path, logdir, 'checkpoint_%04d.pth.tar' % epoch) )
        
        if loss < best_loss:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_stage2.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=os.path.join(save_root_path, logdir, 'model_best.pth.tar') )
            print("Saved model_best.pth.tar on epoch", epoch+1)

    summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    criterion = nn.CrossEntropyLoss()

    end = time.time()
    iters_per_epoch = len(train_loader)
    total_loss, total_batch = 0, 0

    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(images)
        loss = criterion(outputs, labels)

        losses.update(loss.item(), images.size(0))
        total_loss += loss
        total_batch += index.shape[0]
        summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return total_loss/total_batch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch) / (args.epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
