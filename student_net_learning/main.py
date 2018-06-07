'''
Simple transfer learning.
Teacher model: Image descriptors from black-box model
Student model: VGG|ResNet|DenseNet
'''

from __future__ import print_function
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import PIL

from models import *
from dataset import ImageListDataset
from utils import progress_bar

#import MCS2018

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

#torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description='PyTorch student network training')

parser.add_argument('--lr',
                    default=0.001, 
                    type=float, 
                    help='learning rate')
parser.add_argument('--resume',
                    action='store_true', 
                    help='resume from checkpoint')
parser.add_argument('--optimizer',
                    type=str, 
                    help='optimizer type', 
                    default='adam')
parser.add_argument('--criterion',
                    type=str, 
                    help='criterion', 
                    default='MSE')
parser.add_argument('--root',
                    default='../data/',
                    type=str, 
                    help='data root path')
parser.add_argument('--datalist', 
                    default='../data/datalist/',
                    type=str, 
                    help='datalist path')
parser.add_argument('--batch_size', 
                    type=int, 
                    help='mini-batch size',
                    default=16)
parser.add_argument('--name',
                    required=True,
                    type=str, 
                    help='session name')
parser.add_argument('--log_dir_path',
                    default='./logs',
                    type=str, 
                    help='log directory path')
parser.add_argument('--epochs',
                    required=True,
                    type=int,
                    help='number of epochs')
parser.add_argument('--cuda',
                    action='store_true', 
                    help='use CUDA')
parser.add_argument('--model_name', 
                    type=str, 
                    help='model name', 
                    default='ResNet18')
parser.add_argument('--down_epoch', 
                    type=int, 
                    help='epoch number for lr * 1e-1', 
                    default=30)
parser.add_argument('--max_train_imgs', 
                    type=int, 
                    help='max images per epoch', 
                    default=0)
parser.add_argument('--finetune',
                    action='store_true', 
                    help='train only last layer')
parser.add_argument('--ignore_prev_run',
                    action='store_true', 
                    help='ignore previous loss and epoch (in case of resume)')
parser.add_argument('--use_norm',
                    action='store_true', 
                    help='use normalization')
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args, just_print=False):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    
    lr = args.lr * (0.1 ** (epoch//args.down_epoch))
    
    #if just_print:
    #    print()
    #    print('LR', lr)
    #else:
    
    print()
    print('LR', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def l2_norm(input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = normp#torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
#black_box = MCS2018.Predictor(0)
def train(epoch):
    '''
    Train function for each epoch
    '''

    global net
    global trainloader
    global args
    global log_file
    global optimizer
    global criterion

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    adjust_learning_rate(optimizer, epoch, args)

    last_batch = len(trainloader) if args.max_train_imgs == 0 else args.max_train_imgs // args.batch_size
    for batch_idx, (inputs, targets) in enumerate(trainloader):
    
        if batch_idx >= last_batch:
            break

        inputs, targets = inputs, targets.squeeze()
        #for i, inpt in enumerate(inputs):
        #    targets[i] = black_box.submit(np.array(inpt.unsqueeze(0), dtype=np.float32)).squeeze()
        
        #targets = black_box.submit(np.array(inputs, dtype=np.float32)).squeeze()
        
        #adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)

        optimizer.zero_grad()
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)
        outputs = net(inputs)
        
        if args.use_norm:
            outputs = l2_norm(outputs)

        #print(outputs.data.cpu().numpy().min())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        curr_batch_loss = loss.item()
        train_loss += curr_batch_loss
        total += targets.size(0)

        log_file.write('train,{epoch},'\
                       '{batch},{loss:.6f}\n'.format(epoch=epoch, 
                                                     batch=batch_idx,
                                                     loss=curr_batch_loss))
        progress_bar(batch_idx, 
                     last_batch,
                     'Loss: {l:.6f}'.format(l = train_loss/(batch_idx+1)))

def validation(epoch):
    
    global net
    global valloader
    global best_loss
    global args
    global log_file

    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        inputs, targets = inputs, targets.squeeze()
        if args.cuda:
            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)
        inputs, targets = Variable(inputs), Variable(targets)
        
        with torch.no_grad():
            outputs = net(inputs)
            if args.use_norm:
                outputs = l2_norm(outputs)
            loss = criterion(outputs, targets)

        curr_batch_loss = loss.item()
        val_loss += curr_batch_loss

        log_file.write('val,{epoch},'\
                       '{batch},{loss:.6f}\n'.format(epoch=epoch, 
                                                     batch=batch_idx,
                                                     loss=curr_batch_loss))
        progress_bar(batch_idx, 
                     len(valloader), 
                     'Loss: {l:.6f}'.format(l = val_loss/(batch_idx+1)))
    val_loss = val_loss/(batch_idx+1)
    if val_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict() if torch.cuda.device_count() <= 1 \
                                    else net.module.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
            'arguments': args
        }
        session_checkpoint = 'checkpoint/{name}/'.format(name=args.name)
        if not os.path.isdir(session_checkpoint):
            os.makedirs(session_checkpoint)
        torch.save(state, session_checkpoint + 'best_model_chkpt.t7')
        torch.save(state, session_checkpoint + 'epoch_' + str(epoch) + '_loss_' + args.criterion + '_{:0.6f}_chkpt.t7'.format(val_loss))
        best_loss = val_loss

def main():
    global net
    global trainloader
    global valloader
    global best_loss
    global log_file
    global optimizer
    global criterion
    #initialize
    start_epoch = 0
    best_loss = np.finfo(np.float32).max

    #augmentation
    random_rotate_func = lambda x: x.rotate(random.randint(-15,15),
                                            resample=Image.BICUBIC)
    random_scale_func = lambda x: transforms.Scale(int(random.uniform(1.0,1.4)\
                                                   * max(x.size)))(x)
    gaus_blur_func = lambda x: x.filter(PIL.ImageFilter.GaussianBlur(radius=1))
    median_blur_func = lambda x: x.filter(PIL.ImageFilter.MedianFilter(size=3))

    #train preprocessing
    transform_train = transforms.Compose([
        #transforms.Lambda(lambd=random_rotate_func),
        #transforms.RandomAffine(15, translate=0.1, scale=(0.9, 1.1), shear=10, resample=PIL.Image.BICUBIC, fillcolor=0),
        #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.RandomRotation(15, resample=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Resize((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    #validation preprocessing
    transform_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize((112,112)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    print('==> Preparing data..')
    trainset = ImageListDataset(root=args.root, 
                                list_path=args.datalist, 
                                split='train', 
                                transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=2, 
                                              pin_memory=False)

    valset = ImageListDataset(root=args.root, 
                               list_path=args.datalist, 
                               split='val', 
                               transform=transform_val)

    valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=1, 
                                             pin_memory=False)

    # Create model
    net = None
    if args.model_name == 'ResNet18':
        net = ResNet18()
    elif args.model_name == 'ResNet34':
        net = ResNet34()
    elif args.model_name == 'ResNet50':
        net = ResNet50()
    elif args.model_name == 'ResNet50-norm':
        net = ResNet50(False, finetune=args.finetune)
    elif args.model_name == 'ResNet101':
        net = ResNet101()
    elif args.model_name == 'ResNet101-norm':
        net = ResNet101(False)
    elif args.model_name == 'DenseNet121':
        net = DenseNet121()
    elif args.model_name == 'DenseNet201':
        net = DenseNet201()
    elif args.model_name == 'Xception':
        net = xception(pretrained=None, num_classes=512, finetune=args.finetune)
    elif args.model_name == 'VGG11':
        net = VGG('VGG11')
    elif args.model_name == 'VGG13':
        net = VGG('VGG13')
    elif args.model_name == 'VGG16':
        net = VGG('VGG16')
    elif args.model_name == 'VGG19':
        net = VGG('VGG19')
    elif args.model_name == 'InceptionV4':
        net = inceptionv4(pretrained=None, num_classes=512)
    print('==> Building model..')

    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('checkpoint\\{0}\\best_model_chkpt.t7'.format(args.name))
        
        # partially load dict.
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        
        net.load_state_dict(model_dict)
        prev_loss = checkpoint['loss']
        print('loss', prev_loss)
        
        if not args.ignore_prev_run:
            start_epoch = checkpoint['epoch'] + 1
            best_loss = prev_loss

    print('Trainable layers', len(list(filter(lambda p: p.requires_grad, net.parameters()))))
    # Choosing of criterion
    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'HUBER':
        criterion = nn.SmoothL1Loss()
    elif args.criterion == 'CLASS':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None # Add your criterion

    # Choosing of optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Load on GPU
    if args.cuda:
        print ('==> Using CUDA')
        print (torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = net.cuda()
        cudnn.benchmark = True
        print ('==> model on GPU')
        criterion = criterion.cuda()
    else:
        print ('==> model on CPU')
    
    if not os.path.isdir(args.log_dir_path):
       os.makedirs(args.log_dir_path)
    log_file_path = os.path.join(args.log_dir_path, args.name + '.log')
    # logger file openning
    log_file = open(log_file_path, 'w')
    log_file.write('type,epoch,batch,loss,acc\n')

    #print ('==> Model')
    #print(net)

    try:
        for epoch in range(start_epoch, args.epochs):
            train(epoch)
            validation(epoch)
        print ('==> Best loss: {0:.5f}'.format(best_loss))
    except Exception as e:
        print (e.message)
        log_file.write(e.message)
    finally:
        log_file.close()

if __name__ == '__main__':
    net = None
    trainloader = None
    valloader = None
    best_loss = None
    log_file = None
    optimizer = None
    criterion = None

    main()
