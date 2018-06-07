import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import train_test_split

class StudentDataset(data.Dataset):
    def __init__(self,
                 mode = 'train', # 'train' or val'
                 df_path='../data/img_list_1M.csv',
                 descriptors_path='../data/img_descriptors_1M.npy',
                 folder='../data/student_model_imgs',
                 random_state=42,
                 test_size=0.1,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 do_augs=False
                ):
        
        self.img_list = list(pd.read_csv('../data/img_list_1M.csv').img_name.values)
        self.img_list = [os.path.join(folder,_) for _ in self.img_list]
        self.descriptors = np.load('../data/img_descriptors_1M.npy')   
        self.mode = mode
        self.mean = mean
        self.std = std
        self.do_augs=do_augs
        self.train_idx, self.val_idx = train_test_split(self.img_list,
                                                        test_size=test_size,
                                                        random_state=random_state)
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
    def __getitem__(self, idx):

        return_tuple = (self.preprocess_img(Image.open(self.img_list[idx])),
                        self.descriptors[idx])
       
        return return_tuple 
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
                            transforms.Resize(112),                  
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            transforms.RandomGrayscale(p=0.25),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])
            
            preprocessing2 = transforms.Compose([
                        transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0), ratio=(1, 1)),
                        transforms.Resize(112),                  
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        transforms.RandomGrayscale(p=0.25),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean,
                                             std=self.std),
                        ])
            
            preprocessing3 = transforms.Compose([
                            transforms.RandomCrop(224),
                            transforms.Resize(112),                  
                            transforms.RandomGrayscale(p=0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.mean,
                                                 std=self.std),
                            ])

            if random.choice([0,1])==0:
                img_arr = preprocessing3(img).numpy()
            else:
                img_arr = preprocessing3(img).numpy()
        return img_arr