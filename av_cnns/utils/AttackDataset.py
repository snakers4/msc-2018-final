import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from random import shuffle
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import KFold

class AttackDataset(data.Dataset):
    def __init__(self,
                 mode = 'train',
                 df_path='../data/pairs_list.csv',
                 folder='../data/imgs',
                 random_state=42,
                 test_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 do_augs=False,
                 do_repeat=False,
                 do_combinations=False,
                 num_folds=4,
                 fold=0,
                 shuffle_img_list=False # to produce random image pairs, for simplicity we just shuffle lists
                ):
        
        process_paths = lambda x: [os.path.join('../data/imgs',_) for _ in x.split('|')]
        
        self.source_imgs = list(pd.read_csv(df_path).source_imgs.apply(process_paths).values)
        self.target_imgs = list(pd.read_csv(df_path).target_imgs.apply(process_paths).values)

        self.mode = mode
        self.mean = mean
        self.std = std
    
        self.fold = fold
        self.do_augs = do_augs
        self.shuffle_img_list = shuffle_img_list
        self.do_repeat = do_repeat
        self.do_combinations = do_combinations
        
        if self.mode in ['train','val']:
            skf = KFold(n_splits = num_folds,
                        shuffle = True,
                        random_state = random_state)
            
            f1, f2, f3, f4 = skf.split(self.source_imgs)
            
            folds = [f1, f2, f3, f4]
            self.train_idx = folds[self.fold][0]
            self.val_idx = folds[self.fold][1] 

        elif self.mode == 'test':
            self.submission_list_jpg,self.submission_list_png,self.test_idx = self.prepare_test_idx()

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
        elif self.mode == 'test':
            return len(self.test_idx)
        
    def prepare_test_idx(self):

        sub_df = pd.read_csv('../data/submit_list.csv')
        df_path='../data/pairs_list.csv'
        folder='../data/imgs'

        process_paths = lambda x: [_.split('.')[0] for _ in x.split('|')]

        submission_set = set(sub_df.path.apply(lambda x: x.split('.')[0]).values)
        submission_list = list(sub_df.path.apply(lambda x: x.split('.')[0]).values)

        source_imgs = list(pd.read_csv(df_path).source_imgs.apply(process_paths).values)
        target_imgs = list(pd.read_csv(df_path).target_imgs.apply(process_paths).values)

        flat_source_imgs = [item for sublist in source_imgs for item in sublist]
        flat_target_imgs = [item for sublist in target_imgs for item in sublist]

        # make sure that the intersection between target imgs and source imgs is zero
        assert len(set(flat_source_imgs).intersection(set(flat_target_imgs))) == 0

        # submit imgs come only from srouce imgs
        assert len(set(flat_source_imgs).intersection(submission_set)) == 5000
        assert len(set(flat_target_imgs).intersection(submission_set)) == 0

        def find_idx(img_name,source_imgs):
            for i,img_set in enumerate(source_imgs):
                if img_name in img_set:
                    return i

        test_idx = [find_idx(_,source_imgs) for _ in submission_list]

        submission_list_jpg = [os.path.join('../data/imgs',_)+'.jpg' for _ in submission_list]
        submission_list_png = [os.path.join('../data/imgs',_)+'.png' for _ in submission_list]

        return submission_list_jpg,submission_list_png,test_idx        
        
    def __getitem__(self, idx):
        
        if self.mode in ['train','val']:
            
            if self.mode == 'train':
                idx = self.train_idx[idx]
            elif self.mode == 'val':
                idx = self.val_idx[idx]
            
            if self.do_repeat:
                # transform Bx5/Bx5 image sets into
                # B/Bx5, i.e. just repeat each target tensor 5 times
                sources = self.process_5_imgs(self.source_imgs[idx])
                targets = self.process_5_imgs(self.target_imgs[idx])
                targets = targets[np.newaxis,:]
                targets = np.repeat(targets, 5, axis=0)

                return (sources,targets)

            elif self.do_combinations==True:
                # transform Bx5/Bx5 into
                # B/B, where batch_size 

                sources = self.process_5_imgs(self.source_imgs[idx],is_list=True)
                targets = self.process_5_imgs(self.target_imgs[idx],is_list=True)

                sources_25 = []
                targets_25 = []
                targets_25_5 = []

                for source in sources:
                    for target in targets:
                        sources_25.append(source)
                        targets_25.append(target)
                        targets_25_5.append(targets)

                return (np.asarray(sources_25),np.asarray(targets_25),np.asarray(targets_25_5))            

            else:
                return_tuple = (self.process_5_imgs(self.source_imgs[idx]),
                                self.process_5_imgs(self.target_imgs[idx]))

                return return_tuple
            
        elif self.mode == 'test':
            test_idx = self.test_idx[idx]

            self.submission_list_jpg,self.submission_list_png,self.test_idx

            sources = self.process_5_imgs([self.submission_list_jpg[idx]],is_list=False)
            targets = self.process_5_imgs(self.target_imgs[test_idx],is_list=False)
            
            return_tuple = (sources,
                            targets,
                            self.submission_list_jpg[idx],
                            self.submission_list_png[idx])
            
            return return_tuple
            
        
    def process_5_imgs(self,
                       img_list,
                       is_list=False):
        
        if self.shuffle_img_list:
            shuffle(img_list)
            
        imgs = []
        for img in img_list:
            imgs.append(self.preprocess_img(Image.open(img)))

        if is_list==False:
            return np.asarray(imgs)
        else:
            return imgs
        
    def preprocess_img(self,
                       img):

        if self.do_augs==False:
            preprocessing = transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.Resize(112),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
            img_arr = preprocessing(img).numpy()            
        else:
            preprocessing1 = transforms.Compose([
                            transforms.RandomCrop(224),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            transforms.RandomGrayscale(p=0.25),
                            transforms.Resize(112),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
            
            preprocessing2 = transforms.Compose([
                        transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0), ratio=(1, 1)),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        transforms.RandomGrayscale(p=0.25),
                        transforms.Resize(112),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean,
                                             std=self.std),
                        ])    

            if random.choice([0,1])==0:
                img_arr = preprocessing1(img).numpy()
            else:
                img_arr = preprocessing2(img).numpy()
        return img_arr