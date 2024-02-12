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
import torch.optim
import torch.utils.data
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
                    help='mini-batch size (default: 4096)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--mlp-dim', default=2048, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--t', default=0.1, type=float,
                    help='softmax temperature (default: 1.0)')
parser.add_argument('--num_proj_layers', default=2, type=int,
                    help='number of non-linear projection heads')

# dataset 
parser.add_argument('--data_name', default='imagenet1000', type=str) 
parser.add_argument('--save_dir', default='./saved_models/', type=str) 



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
    model = ClassificationModel(model.base_encoder, num_classes=class_size)
    model = model.cuda(args.gpu)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        if args.gpu is None:
            checkpoint = torch.load(args.model_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.model_path, map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))

 
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging

    global_batch_size = args.batch_size
    
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

    if args.data_name == 'imagenet1000' or args.data_name == 'imagenet100' or args.data_name == 'imagenet50' or args.data_name == 'cifar10':
        testdir = os.path.join(args.data, 'test')
        test_dataset = sogclr.folder.ImageFolder(
            testdir,
            transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
    else:
        raise ValueError

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10011, shuffle=False, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()

    validate(test_loader, model, criterion, args)
        

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            print(output.shape)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
