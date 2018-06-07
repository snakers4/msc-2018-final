# Custom tensorboard logging
from utils.TbLogger import Logger

import os
import time
import tqdm
import shutil
import pickle 
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import  metrics

# torch imports
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from   torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

# custom classes
from utils.StudentDataset import StudentDataset
from models.StudentModels import load_model, FineTuneModel

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Student training')

# ============ basic params ============#
parser.add_argument('--workers',             default=4,             type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',              default=50,            type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch',         default=0,             type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size',          default=64,            type=int, help='mini-batch size (default: 64)')
parser.add_argument('--fold_num',            default=0,             type=int, help='fold number 0 - 3 (default: 0)')

# ============ data loader and model params ============#
parser.add_argument('--val_size',            default=0.1,           type=float, help='share of the val set')
parser.add_argument('--do_augs',             default=False,         type=str2bool, help='Whether to use aygs')
parser.add_argument('--arch',                default='resnet18',    type=str, help='text id for saving logs')
parser.add_argument('--ths',                 default=1e-2,          type=float, help='share of the val set')

# ============ optimization params ============#
parser.add_argument('--lr',                  default=1e-3,          type=float, help='initial learning rate')
parser.add_argument('--m1',                  default=10,            type=int, help='lr decay milestone 1')
parser.add_argument('--m2',                  default=30,            type=int, help='lr decay milestone 2')
parser.add_argument('--optimizer',           default='adam',        type=str, help='model optimizer')

# ============ logging params and utilities ============#
parser.add_argument('--print-freq',          default=10,            type=int, help='print frequency (default: 10)')
parser.add_argument('--lognumber',           default='test_model',  type=str, help='text id for saving logs')
parser.add_argument('--tensorboard',         default=False,         type=str2bool, help='Use tensorboard to for loss visualization')
parser.add_argument('--tensorboard_images',  default=False,         type=str2bool, help='Use tensorboard to see images')
parser.add_argument('--resume',              default='',            type=str, help='path to latest checkpoint (default: none)')

# ============ other params ============#
parser.add_argument('--predict',             dest='predict',       action='store_true', help='generate prediction masks')
parser.add_argument('--predict_train',       dest='predict_train', action='store_true', help='generate prediction masks')
parser.add_argument('--evaluate',            dest='evaluate',      action='store_true', help='just evaluate')


best_mse = 1
train_minib_counter = 0
valid_minib_counter = 0

args = parser.parse_args()
print(args)

# add fold number to the lognumber 
if not (args.predict or args.predict_train):
    args.lognumber = args.lognumber + '_fold' + str(args.fold_num)

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    if not (args.predict or args.predict_train):
        logger = Logger('./tb_logs/{}'.format(args.lognumber))
    else:
        logger = Logger('./tb_logs/{}'.format(args.lognumber + '_predictions'))


def main():
    global args, best_mse
    global logger
  
    # do transfer learning
    model = load_model(args.arch,pretrained=True)
    model = FineTuneModel(model,args.arch,512)

    # model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     # Only finetunable params
                                     lr=args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        # Only finetunable params
                                        lr=args.lr)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    # Only finetunable params
                                    lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')    
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mse = checkpoint['best_mse']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))    
    
    if args.predict:
        pass
        
    else:
        train_dataset = StudentDataset(mode = 'train',
                               test_size=args.val_size,
                               mean=model.module.mean,
                               std=model.module.std,
                               do_augs=args.do_augs)

        val_dataset = StudentDataset(mode = 'val',
                               test_size=args.val_size,
                               mean=model.module.mean,
                               std=model.module.std,
                               do_augs=args.do_augs)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        criterion = torch.nn.MSELoss().cuda()

        scheduler = MultiStepLR(optimizer, milestones=[args.m1,args.m2], gamma=0.1)  

        for epoch in range(args.start_epoch, args.epochs):
            # adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_loss,train_hard_mse,train_hard_mse_crude,train_hard_mse_fine = train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            val_loss,val_hard_mse,val_hard_mse_crude,val_hard_mse_fine = validate(val_loader, model, criterion)

            scheduler.step()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'train_epoch_loss': train_loss,
                    'valid_epoch_loss': val_loss,
                    'train_epoch_hmse': train_hard_mse,
                    'valid_epoch_hmse': val_hard_mse,
                    'train_epoch_hmse_crude': train_hard_mse_crude,
                    'valid_epoch_hmse_crude': val_hard_mse_crude,                    
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)                     

            # remember best prec@1 and save checkpoint
            is_best = val_loss < best_mse
            best_mse = min(val_loss, best_mse)
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),            
                'best_mse': best_mse,
                },
                is_best,
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
                'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
   
def train(train_loader, model, criterion, optimizer, epoch):
    global train_minib_counter
    global logger
        
    # for cyclic LR
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    hard_mse = AverageMeter()
    hard_mse_crude = AverageMeter()
    hard_mse_fine = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input,target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)        
        
        input = input.float()
        target = target.float()
       
        input_var = torch.autograd.Variable(input.cuda(async=True))
        target_var = torch.autograd.Variable(target.cuda(async=True))

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        hard_mse_score = measure_hard_mse(output.data, target_var.data,args.ths)
        hard_mse_score_crude = measure_hard_mse(output.data, target_var.data,args.ths*10)
        hard_mse_score_fine = measure_hard_mse(output.data, target_var.data,args.ths/10)
        
        losses.update(loss.data[0], input.size(0))
        hard_mse.update(hard_mse_score, input.size(0))
        hard_mse_crude.update(hard_mse_score_crude, input.size(0))
        hard_mse_fine.update(hard_mse_score_fine, input.size(0))
                
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
                'train_lr': current_lr,
                'train_hard_mse': hard_mse.val,
                'train_hard_mse_crude': hard_mse_crude.val,
                'train_hard_mse_fine': hard_mse_fine.val,                
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter) 
        
        
        train_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data   {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LOSS   {loss.val:.5f} ({loss.avg:.5f})\t'
                  'HMSE0  {hard_mse_fine.val:.5f} ({hard_mse_fine.avg:.5f})\t'                  
                  'HMSE1  {hard_mse.val:.5f} ({hard_mse.avg:.5f})\t'
                  'HMSE2  {hard_mse_crude.val:.5f} ({hard_mse_crude.avg:.5f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,data_time=data_time,
                      loss=losses, hard_mse=hard_mse,hard_mse_crude=hard_mse_crude,hard_mse_fine=hard_mse_fine))
             
    return losses.avg,hard_mse.avg,hard_mse_crude.avg,hard_mse_fine.avg

def validate(val_loader, model, criterion):
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    

    sm = torch.nn.Softmax(dim=1)
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    hard_mse = AverageMeter()
    hard_mse_crude = AverageMeter()    
    hard_mse_fine = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
           
    for i, (input,target) in enumerate(val_loader):
        
        input = input.float()
        target = target.float()
       
        input_var = torch.autograd.Variable(input.cuda(async=True),volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True),volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)


        # measure accuracy and record loss
        hard_mse_score = measure_hard_mse(output.data, target_var.data,args.ths)
        hard_mse_score_crude = measure_hard_mse(output.data, target_var.data,args.ths*10)        
        hard_mse_score_fine = measure_hard_mse(output.data, target_var.data,args.ths/10)
        
        losses.update(loss.data[0], input.size(0))
        hard_mse.update(hard_mse_score, input.size(0))
        hard_mse_crude.update(hard_mse_score_crude, input.size(0))        
        hard_mse_fine.update(hard_mse_score_fine, input.size(0))
                        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'val_loss': losses.val,
                'val_hard_mse': hard_mse.val,
                'val_hard_mse_crude': hard_mse_crude.val, 
                'val_hard_mse_fine': hard_mse_fine.val, 
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'LOSS   {loss.val:.5f} ({loss.avg:.5f})\t'
                  'HMSE0  {hard_mse_fine.val:.5f} ({hard_mse_fine.avg:.5f})\t'                      
                  'HMSE1  {hard_mse.val:.5f} ({hard_mse.avg:.5f})\t'
                  'HMSE2  {hard_mse_crude.val:.5f} ({hard_mse_crude.avg:.5f})\t'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses,hard_mse=hard_mse,hard_mse_crude=hard_mse_crude,hard_mse_fine=hard_mse_fine))

    return losses.avg,hard_mse.avg,hard_mse_crude.avg,hard_mse_fine.avg

def predict(val_loader, model):
    pass

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def measure_hard_mse(output,target,ths):
    _ = torch.abs(output - target)
    _ = (_ < ths) * 1
    items = _.shape[0] * _.shape[1]
    
    return float(_.sum() / items)
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()