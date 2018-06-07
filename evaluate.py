'''
Evaluate script before submit
'''

import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import argparse
from skimage.measure import compare_ssim
from torchvision import transforms
from tqdm import tqdm

import MCS2018

SSIM_THR = 0.95
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='pre-submit evaluate')
parser.add_argument('--original_root',
                    type=str, 
                    help='original data root path',
                    default='data/imgs/')
parser.add_argument('--attack_root',
                    required=True,
                    type=str, 
                    help='changed data root path')
parser.add_argument('--submit_list',
                    type=str, 
                    help='path of datalist',
                    default='data/submit_list.csv')
parser.add_argument('--target_dscr',
                    required=True,
                    type=str,
                    help="target descriptors path (.npy),"\
                         " will be created if file doesn't exist")
parser.add_argument('--submit_name',
                     required=True,
                     type=str)
parser.add_argument('--gpu_id',
                     type=int,
                     help='GPU ID for black box, default = -1 (CPU)',
                     default=-1)
parser.add_argument('--pairs_list',
                     type=str,
                     help='attack pairs list',
                     default='data/pairs_list.csv')
args = parser.parse_args()


def euclid_dist(x,y, axis=1): 
    return np.sqrt(((x - y) ** 2).sum(axis=axis))

def main(args):
    # loading black-box model
    net = MCS2018.Predictor(args.gpu_id)

    img_list = pd.read_csv(args.submit_list)
    descriptors = np.ones((len(img_list), 512))

    cropping = transforms.Compose([transforms.CenterCrop(224),
                                   transforms.Resize(112)])
    cropped_img_preprocessing = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=MEAN, std=STD)
                                ])
    # SSIM checking
    for idx, img_name in tqdm(enumerate(img_list.path.values),
                              total=len(img_list.path.values),
                              desc='SSIM'):

        img = Image.open(os.path.join(args.attack_root, img_name))
        org_img = Image.open(os.path.join(args.original_root, 
                                          img_name.replace('.png', '.jpg')))
        org_img = cropping(org_img)

        ssim = compare_ssim(np.array(img), np.array(org_img), multichannel=True)
        assert ssim >= SSIM_THR, '{0}\n ssim {1} < {2}'.format(img_name, ssim, SSIM_THR)

        # Creating batch with one element
        batch_from_one_img = np.array(cropped_img_preprocessing(img).unsqueeze(0),
                                      dtype=np.float32)

        # Getting batch result from black-box
        res = net.submit(batch_from_one_img).squeeze()
        descriptors[idx] = res

    # Saving descriptors for submit
    descriptor_path = os.path.join(args.attack_root, 'descriptors.npy')
    np.save(descriptor_path, descriptors)

    
    # axis 0 - number of imgs for each class
    # axis 1 - number of classes
    # axis 2 - descriptor size
    descriptors = descriptors.reshape((5,1000,512), order='F')

    if not os.path.isfile(args.target_dscr):
        pairs_list = pd.read_csv(args.pairs_list)
        preprocessing = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                ])

        val_img_list = []
        for target_imgs in pairs_list.target_imgs.values:
            for img_name in target_imgs.split('|'):
                img_name = os.path.join(args.original_root, img_name)
                val_img_list.append(img_name)

        target_descriptors = np.ones((5000, 512), dtype=np.float32)
        for idx, img_name in tqdm(enumerate(val_img_list), 
                                  total=len(val_img_list),
                                  desc='get descriptors'):
            img = Image.open(img_name)
            img_arr = preprocessing(img).unsqueeze(0).numpy()
            res = net.submit(img_arr).squeeze()
            target_descriptors[idx] = res
        target_descriptors = target_descriptors.reshape((5,1000,512), 
                                                        order='F')
        np.save(args.target_dscr, target_descriptors)

    #target descriptors shape: (5,1000,512)
    target_descriptors = np.load(args.target_dscr)

    # axis 0 - img number for source class
    # axis 1 - img number for target class
    # axis 2 - number of classes
    dist_all = np.zeros((5,5,1000))

    for idx, dscr_row in enumerate(descriptors):
        for jdx, target_dscr_row in enumerate(target_descriptors):
            assert(dscr_row.shape == target_dscr_row.shape)
            dist = euclid_dist(target_dscr_row, dscr_row)
            dist_all[idx][jdx] = dist

    score_value = dist_all.mean(axis=1).mean(axis=0).mean()
    print ('Validation score: {0:.6f}'.format(score_value))

    # Submit zip archive creating
    submit_directory = 'submits'
    if not os.path.isdir(submit_directory):
        os.makedirs(submit_directory)

    submit_file = os.path.join(submit_directory, args.submit_name + '.zip')
    with zipfile.ZipFile(submit_file,'w') as myzip:
        for img_name in tqdm(img_list.path.values,
                             desc='archive images'):
            img_path = os.path.join(args.attack_root, img_name)
            myzip.write(img_path, arcname=img_name)
        myzip.write(descriptor_path, arcname='descriptors.npy') 

if __name__ == '__main__':
    main(args)
