'''Custom dataset for loading imgs and descriptors
'''
import os.path

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def np_loader(path):
    return np.load(path)

def build_dataset_lists(list_path,split):
    im_list = os.path.join(list_path, 'im_'+split+'.txt')
    at_list = os.path.join(list_path, 'at_'+split+'.npy')
    images = pd.read_csv(im_list, header=None, names=['impath'])
    targets = np.load(at_list)
    return images.impath.values,targets

class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    root -- path to images
    list_path -- path to image lists
    split -- train|val| - name of the dataset part (default train)
    transform -- transform for images
    """
    def __init__(self, root, list_path, split = 'train', 
                 transform=None, loader=default_loader):
        
        images, targets = build_dataset_lists(list_path,split)
        self.root = root
        self.images = root + images
        self.targets = targets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) 
        """
        path = self.images[index]
        target = self.targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)
