import MCS2018

# Custom tensorboard logging
from utils.TbLogger import Logger

import os
import sys
import time
import tqdm
import shutil
import pickle 
import zipfile
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import  metrics
from skimage.io import imread
from PIL import Image
from skimage.measure import compare_ssim as sk_ssim

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
from utils.AttackDataset import AttackDataset
from utils.Viz import compare_img
from models.AttackerLoss import AttackerLoss
from models.AttackModels import Attacker34E2E as Attacker34
from models.StudentModels import load_model, FineTuneModel


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Student training')

SSIM_THR = 0.95

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

REVERSE_MEAN = [-0.485, -0.456, -0.406]
REVERSE_STD = [1/0.229, 1/0.224, 1/0.225]

# ============ basic params ============#
parser.add_argument('--workers',             default=4,             type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',              default=50,            type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch',         default=0,             type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size',          default=64,            type=int, help='mini-batch size (default: 64)')
parser.add_argument('--fold_num',            default=0,             type=int, help='fold number 0 - 3 (default: 0)')

# ============ data loader and model params ============#
parser.add_argument('--val_size',            default=0.1,           type=float, help='share of the val set')
parser.add_argument('--do_augs',             default=False,         type=str2bool, help='Whether to use augs')
parser.add_argument('--do_shuffle',          default=False,         type=str2bool, help='Whether to shuffle imgs in img lists')
parser.add_argument('--noise_multiplier',    default=1.0,           type=float, help='noise multiplier')

parser.add_argument('--is_deconv',           default=True ,         type=str2bool, help='Whether to use use_running_mean')

parser.add_argument('--use_running_mean',    default=False,         type=str2bool, help='Whether to use use_running_mean')
parser.add_argument('--ssim_weight',         default=10.0,          type=float, help='noise multiplier')
parser.add_argument('--l2_weight',           default=1.0,           type=float, help='noise multiplier')

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


best_val_eval_l2 = 2
train_minib_counter = 0
train_eval_minib_counter = 0
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
    global args, best_val_eval_l2
    global logger
  
    # do transfer learning
    model = Attacker34(is_deconv = args.is_deconv)
    # model.freeze_encoder()
    
    # model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    bb_model = MCS2018.Predictor(0)
    
    loss_model = get_model('resnet18',
                           'weights/resnet18_scale_fold0_best.pth.tar')
    
    loss_model = torch.nn.DataParallel(loss_model).cuda()
    
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
            best_val_eval_l2 = checkpoint['best_val_eval_l2']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))    
    
    if args.predict:
        
        model.eval()
        print('Running predictions ...')
        
        # import MCS2018
        # net = MCS2018.Predictor(1)  
        
        predict_dset = AttackDataset(mode = 'test',
                                     shuffle_img_list=False,
                                     do_repeat=False,
                                     do_combinations=False
                                    )
        
        loader = torch.utils.data.DataLoader(predict_dset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last=False)
        
        if not os.path.isdir('../submissions/'):
            os.makedirs('../submissions')
            
        submit_file = '../submissions/{}.zip'.format(args.lognumber)
        descriptors_path = '../submissions/{}.npy'.format(args.lognumber)     
        
        with tqdm.tqdm(total=len(loader)) as pbar:

            target_png_paths = []
            # descriptors = np.ones((5000, 512), dtype=np.float32)
            ssim_scores = []
            idx=0
            
            for i, (input,target,jpg_paths,png_paths) in enumerate(loader):
                
                input = input.float()
                target = target.float()

                input = input.float()
                target = target.float()

                input_var = torch.autograd.Variable(input.cuda(async=True),volatile=True)
                target_var = torch.autograd.Variable(target.cuda(async=True),volatile=True)
                
                tensor_attacked_imgs = model('test',input_var,target_var,None)
                tensor_attacked_imgs = tensor_attacked_imgs.data.cpu()
                
                pil_attacked_imgs = []
                
                for _ in tensor_attacked_imgs:
                    pil_attacked_imgs.append(transforms.ToPILImage()(_))
                    
                target_png_paths.extend(list(png_paths))
                
                for pil_attacked_img,jpg_path,png_path in zip(pil_attacked_imgs,jpg_paths,png_paths):
                    
                    pil_attacked_img.save(png_path)
                    
                    # test that imgs are indeed similar
                    img_crop_jpg=img_to_crop(Image.open(jpg_path))
                    img_crop_png=Image.open(png_path)

                    ssim_score = sk_ssim(np.array(img_crop_jpg), np.array(img_crop_png), multichannel=True)                
                    # assert ssim_score>0.95

                    ssim_scores.append(ssim_score)
                    avg_ssim = sum(ssim_scores)/len(ssim_scores)

                    # create descriptions                    
                    img_arr = crop_to_tensor(img_crop_png)
                    
                    # img_des = net.submit(img_arr).squeeze()
                    # descriptors[idx] = img_des                    
                    
                    # pbar.set_description('Epoch [{0}/{1}]'.format(str(epoch).zfill(3),str(epochs).zfill(3)), refresh=False)
                    pbar.set_postfix(avg_ssim=avg_ssim, refresh=False)

                    idx+=1 
                
                # if i>10:
                #    sys.exit()
                    
                pbar.update(1)        

        # assert that the resulting img_id list is equal to that of the submission df
        sub_df = pd.read_csv('../data/submit_list.csv')
        # just check the id sequence
        submission_list = list(sub_df.path.apply(lambda x: x.split('.')[0]).values)
        submitted_list = [_.split('/')[-1].split('.')[0] for _ in target_png_paths]
        
        pd.DataFrame(target_png_paths, columns=['png_path']).to_csv('../submissions/created_pngs.csv')
        
        assert submission_list == submitted_list

        # np.save(descriptors_path, descriptors)
        print('Zipping the submission')
        with zipfile.ZipFile(submit_file,'w') as myzip:
            for img_path in tqdm.tqdm(target_png_paths,
                                 desc='archive'):
                myzip.write(img_path, os.path.basename(img_path))
            # myzip.write(descriptors_path, arcname='descriptors.npy')
    
        
        
    else:
        
        train_dataset_attack = AttackDataset(mode = 'train',
                                             test_size=args.val_size,
                                             shuffle_img_list=args.do_shuffle,
                                             do_repeat=True, 
                                             do_combinations=False
                                            )

        val_dataset_eval = AttackDataset(mode = 'val',
                                         test_size=args.val_size,
                                         shuffle_img_list=False,
                                         do_repeat=False,
                                         do_combinations=True
                                        )   

        train_loader_attack = torch.utils.data.DataLoader(
            train_dataset_attack,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        val_loader_eval = torch.utils.data.DataLoader(
            val_dataset_eval,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)        

        criterion = AttackerLoss(use_running_mean=args.use_running_mean,
                                 ssim_weight=args.ssim_weight,
                                 l2_weight=args.l2_weight).cuda()

        scheduler = MultiStepLR(optimizer, milestones=[args.m1,args.m2], gamma=0.1)  

        for epoch in range(args.start_epoch, args.epochs):
            # adjust_learning_rate(optimizer, epoch)
            
            # train for one epoch

            train_loss,train_similarity_loss,train_distance_loss = train(train_loader_attack,
                                                                         model,loss_model,
                                                                         criterion,
                                                                         optimizer,
                                                                         epoch)

            # ,train_eval_ssim,train_eval_l2,train_eval_hard_ssim
            
            # evaluate on validation set
            val_eval_ssim,val_eval_l2,val_eval_hard_ssim = validate(val_loader_eval,
                                                                    model,loss_model,bb_model,
                                                                    criterion)

            scheduler.step()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'epocha_train_loss': train_loss,
                    'epocha_train_similarity_loss': train_similarity_loss,
                    'epocha_train_distance_loss': train_distance_loss,
                    # 'epocha_train_eval_ssim': train_eval_ssim,
                    # 'epocha_train_eval_l2': train_eval_l2,
                    # 'epocha_train_eval_hard_ssim': train_eval_hard_ssim,
                    'epocha_val_eval_ssim': val_eval_ssim,
                    'epocha_val_eval_l2': val_eval_l2,
                    'epocha_val_eval_hard_ssim': val_eval_hard_ssim,
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)                     

            # remember best prec@1 and save checkpoint
            is_best = val_eval_l2 < best_val_eval_l2
            best_val_eval_l2 = min(val_eval_l2, best_val_eval_l2)
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),            
                'best_val_eval_l2': best_val_eval_l2,
                },
                is_best,
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
                'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
   
def train(train_loader_attack,
          model,loss_model,
          criterion,
          optimizer,
          epoch):
    
    global train_minib_counter,train_eval_minib_counter
    global logger
        
    # for cyclic LR
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    sim_losses = AverageMeter()
    l2_losses = AverageMeter()
    
    # switch to train mode
    model.train()
    
    # just use eval model to calculate loss 
    loss_model.eval()
    
    print('Training...')
    end = time.time()
    
    s = torch.nn.Tanh()

    for i, (input,target) in enumerate(train_loader_attack):
        
        # measure data loading time
        data_time.update(time.time() - end)        
        
        input = input.float()
        target = target.float()
       
        input_var = torch.autograd.Variable(input.cuda(async=True))
        target_var = torch.autograd.Variable(target.cuda(async=True))

        # compute output
        source_mod = model('attack_model_loss',input_var,target_var,None)
        # source_mod = s(source_mod)
        # source_mod = source_mod * input_var.view(-1, 3, 112, 112)
        
        
        modified_descriptions = loss_model(source_mod)
     
        avg_target_descriptions = []
        
        target_vectors_pass = target.view(-1,target.size(2),target.size(3),target.size(4),target.size(5))
        
        for j in range(0,5):
            avg_target_descriptions.append(loss_model(torch.autograd.Variable(target_vectors_pass[:,j,:,:,:].contiguous().cuda(async=True))) ) 

        avg_target_descriptions = torch.stack(avg_target_descriptions,dim=1).mean(dim=1)
        
        loss, similarity_loss, distance_loss = criterion(input_var.view(-1,3,112,112), # conform to the format of model output
                                                         source_mod,
                                                         avg_target_descriptions,
                                                         modified_descriptions)

        losses.update(loss.data[0], input.size(0))
        sim_losses.update(similarity_loss.data[0], input.size(0))
        l2_losses.update(distance_loss.data[0], input.size(0))
                
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
                'train_sim_loss': sim_losses.val,
                'train_l2_loss': l2_losses.val,
                'train_lr': current_lr,                
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter) 
        
        
        train_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data   {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LOSS   {loss.val:.5f} ({loss.avg:.5f})\t'
                  'SLOSS  {sim_losses.val:.5f} ({sim_losses.avg:.5f})\t'                  
                  'L2LOSS {l2_losses.val:.5f} ({l2_losses.avg:.5f})\t'.format(
                      epoch, i, len(train_loader_attack),
                      batch_time=batch_time,data_time=data_time,
                      loss=losses,sim_losses=sim_losses,l2_losses=l2_losses))

    # eval_ssims.avg,eval_l2s.avg,eval_sk_hard_ssims.avg
    return losses.avg,sim_losses.avg,l2_losses.avg

def validate(val_loader_eval,
             model,loss_model,bb_model,
             criterion):
    
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    eval_ssims = AverageMeter()    
    eval_l2s = AverageMeter()
    eval_sk_hard_ssims = AverageMeter()
    eval_bb_l2s = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    loss_model.eval()
    
    # s = torch.nn.Tanh()
    
    end = time.time()
           
    for i, (input,target,target5) in enumerate(val_loader_eval):
        
        input = input.float()
        target = target.float()
       
        input = input.float()
        target = target.view(-1,target.size(2),target.size(3),target.size(4)).float()
        target5 = target5.float()
        
        input_var = torch.autograd.Variable(input.cuda(async=True),volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True),volatile=True)
        target5_var = torch.autograd.Variable(target5.cuda(async=True),volatile=True)

        source_mod = model('eval_model_loss',
                          input_var,
                          None,
                          target5_var)
        
        # source_mod = s(source_mod)
        # source_mod = source_mod * input_var.view(-1, 3, 112, 112)   
        
        modified_descriptions = loss_model(source_mod)
        target_descriptions = loss_model(target_var)          
        
        loss, similarity_loss, distance_loss = criterion(input_var.view(-1,3,112,112),
                                                         source_mod,
                                                         target_descriptions,
                                                         modified_descriptions)
        
        input_imgs_pil = []
        mod_imgs_pil = []
        
        for t in input_var.view(-1,3,112,112).data.cpu():
            input_imgs_pil.append(tensor2img(t.squeeze()))
            
        for t in source_mod.data.cpu():
            mod_imgs_pil.append(tensor2img(t.squeeze()))          
        
        input_imgs_npy = []
        mod_imgs_npy = []
        
        for _ in input_imgs_pil:
            input_imgs_npy.append(np.array(_))
            
        for _ in mod_imgs_pil:
            mod_imgs_npy.append(np.array(_))            
        
        sk_ssim = batch_sk_ssim(input_imgs_npy,
                                mod_imgs_npy
                               )        

        sk_hard_ssim = batch_sk_hard_ssim(input_imgs_npy,
                                          mod_imgs_npy,
                                          0.95)    
        
        bb_dists = []
        
        for t_mod,t_target in zip(source_mod.data.cpu(),target.view(-1,3,112,112)):
            t_mod = t_mod.numpy()[np.newaxis,:,:,:]
            t_target = t_target.numpy()[np.newaxis,:,:,:]
            
            mod_bb_des = bb_model.submit(t_mod)
            target_bb_des = bb_model.submit(t_target)
            
            bb_dist = float(euclid_dist(mod_bb_des, target_bb_des).mean())
            bb_dists.append(bb_dist)
            
        
        eval_ssims.update(sk_ssim, input.size(0))
        eval_l2s.update(distance_loss.data[0], input.size(0))
        eval_bb_l2s.update(sum(bb_dists)/len(bb_dists), len(bb_dists))
        eval_sk_hard_ssims.update(sk_hard_ssim, input.size(0))
                        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_eval_ssim': eval_ssims.val,
                'train_eval_l2': eval_l2s.val,
                'train_eval_bb_l2': eval_bb_l2s.val,
                'train_eval_sk_hard_ssim': eval_sk_hard_ssims.val, 

            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        
        if i % args.print_freq == 0:            
            #============ TensorBoard logging ============# 
            if args.tensorboard_images:
                    
                pannos = process_img_pannos(input_imgs_pil,
                                           mod_imgs_pil)
                
                info = {
                    'pannos': pannos,
                }
                
                for tag, images in info.items():
                    logger.image_summary(tag, images, valid_minib_counter)                    
     
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'SSIM   {eval_ssims.val:.5f} ({eval_ssims.avg:.5f})\t'                      
                  'L2     {eval_l2s.val:.5f} ({eval_l2s.avg:.5f})\t'
                  'L2 BB  {eval_bb_l2s.val:.5f} ({eval_bb_l2s.avg:.5f})\t'
                  'HSSIM  {eval_sk_hard_ssims.val:.5f} ({eval_sk_hard_ssims.avg:.5f})\t'.format(
                      i, len(val_loader_eval), batch_time=batch_time,
                      eval_ssims=eval_ssims,eval_l2s=eval_l2s,eval_sk_hard_ssims=eval_sk_hard_ssims,eval_bb_l2s=eval_bb_l2s))
            
        if i==3:
            break

    return eval_ssims.avg,eval_l2s.avg,eval_sk_hard_ssims.avg

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

def batch_sk_hard_ssim(batch1,batch2,ths):
    
    ssims = []
    
    for img1,img2 in zip(batch1,batch2):
        _ = sk_ssim(img1, img2, multichannel=True)>ths
        ssims.append(_)
    
    return sum(ssims)/len(ssims)

def batch_sk_ssim(batch1,batch2):
    
    ssims = []
    
    for img1,img2 in zip(batch1,batch2):
        _ = sk_ssim(img1, img2, multichannel=True)
        ssims.append(_)
    
    return sum(ssims)/len(ssims)

def process_img_pannos(batch_img,
                       batch_mod_img):
    
    pannos = []
    
    for img,mod_image in zip(batch_img,batch_mod_img):
        pannos.append(compare_img(img,mod_image))
            
    return np.asarray(pannos).transpose(0,3,1,2)

def img_to_crop(img):
    preprocessing = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.Resize(112),
                    ])
    return preprocessing(img)

def crop_to_tensor(img):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                    ])
    img_arr = preprocessing(img).unsqueeze(0).numpy()
    return img_arr

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

def get_model(model_name, checkpoint_path):
    '''
    Model architecture choosing
    '''
    # do transfer learning
    model = load_model(model_name,
                       pretrained=True)
    model = FineTuneModel(model,model_name,512)

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    model=model.module
    
    return model

def reverse_normalize(tensor, mean, std):
    '''reverse normalize to convert tensor -> PIL Image'''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def euclid_dist(x,y, axis=1): 
    return np.sqrt(((x - y) ** 2).sum(axis=axis))

def tensor2img(tensor, on_cuda=True):
    tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
    # clipping
    tensor[tensor > 1] = 1
    tensor[tensor < 0] = 0
    tensor = tensor.squeeze(0)
    if on_cuda:
        tensor = tensor.cpu()
    return transforms.ToPILImage()(tensor)

if __name__ == '__main__':
    main()