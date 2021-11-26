#!/usr/bin/env python3
# coding: utf-8
import os
import os.path as osp
from pathlib import Path
import numpy as np
import argparse 
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark=True

from utils.ddfa import DDFADataset, ToTensor, Normalize, SGD_NanHandler, CenterCrop, Compose_GT, ColorJitter
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from model_building import SynergyNet as SynergyNet
from benchmark_validate import benchmark_pipeline


# global args (configuration)
args = None # define the static training setting, which wouldn't and shouldn't be changed over the whole experiements.

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('-j', '--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--resume_pose', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0,1', type=str)
    parser.add_argument('--filelists-train',default='', type=str)
    parser.add_argument('--root', default='')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--arch', default='mobilenet_v2', type=str, help="Please choose [mobilenet_v2, mobilenet_1, resnet50, resnet101, or ghostnet]")
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--param-fp-train',default='',type=str)
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('--save_val_freq', default=10, type=int)

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    #global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, model, optimizer, epoch, lr):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_name = list(model.module.get_losses())
    losses_name.append('loss_total')
    losses_meter = [AverageMeter() for i in range(len(losses_name))]

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda(non_blocking=True)

        target = target[:,:62]
        target.requires_grad = False  
        target = target.float().cuda(non_blocking=True)
        
        losses = model(input, target)

        data_time.update(time.time() - end)

        loss_total = 0
        for j, name in enumerate(losses):
            mean_loss = losses[name].mean()
            losses_meter[j].update(mean_loss, input.size(0))
            loss_total += mean_loss

        losses_meter[j+1].update(loss_total, input.size(0))
        
        ### compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        flag, _ = optimizer.step_handleNan()

        if flag:
            print("Nan encounter! Backward gradient error. Not updating the associated gradients.")

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            msg = 'Epoch: [{}][{}/{}]\t'.format(epoch, i, len(train_loader)) + \
                  'LR: {:.8f}\t'.format(lr) + \
                  'Time: {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)
            for k in range(len(losses_meter)):
                msg += '{}: {:.4f} ({:.4f})\t'.format(losses_name[k], losses_meter[k].val, losses_meter[k].avg)
            logging.info(msg)


def main():
    """ Main funtion for the training process"""
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    model = SynergyNet(args)
    torch.cuda.set_device(args.devices_id[0])

    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    # step2: optimization: loss and optimization method

    optimizer = SGD_NanHandler(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            
        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    normalize = Normalize(mean=127.5, std=128)

    train_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        gt_transform=True,
        transform=Compose_GT([ColorJitter(0.4,0.4,0.4), ToTensor(), CenterCrop(5), normalize]) #
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)

    
    # step4: run
    cudnn.benchmark = True
    if args.test_initial: # if testing the performance from the initial
        logging.info('Testing from initial')
        benchmark_pipeline(model)
        

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, lr)

        filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
        # save checkpoints and current model validation
        if (epoch % args.save_val_freq == 0) or (epoch==args.epochs):
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                },
                filename
            )
            logging.info('\nVal[{}]'.format(epoch))
            benchmark_pipeline(model)

if __name__ == '__main__':
    main()
