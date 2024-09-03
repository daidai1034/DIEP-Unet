import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import cv2
from glob import glob
import imageio
import torch


class COVID19_CT_Lung_and_Infection_Segmentation_Dataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'dataset/COVID-19 CT Lung and Infection Segmentation Dataset/'
        # self.root = r'dataset/COVID-19 CT segmentation dataset/'
        self.img_paths = []
        self.mask_paths = []
        self.train_img_paths, self.val_img_paths, self.test_img_paths = [], [], []
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = [], [], []
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        ct_folder = os.path.join(self.root, 'images')
        mask_folder = os.path.join(self.root, 'masks')

        for i in range(1, 21):
            ct_subfolder = os.path.join(ct_folder, f"{i:02}")
            mask_subfolder = os.path.join(mask_folder, f"{i:02}")

            ct_images = sorted(glob(os.path.join(ct_subfolder, '*.png')))
            mask_images = sorted(glob(os.path.join(mask_subfolder, '*.png')))
            
        # ct_images = sorted(glob(os.path.join(ct_folder, '*.png')))
        # mask_images = sorted(glob(os.path.join(mask_folder, '*.png')))

            self.img_paths.extend(ct_images)
            self.mask_paths.extend(mask_images)
       
    
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=40)
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = \
            train_test_split(self.val_img_paths, self.val_mask_paths, test_size=0.5, random_state=40)

        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':#1475 #2816 
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':#184 #352
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':#185 #352
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
      
        pic = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
        pic = cv2.resize(pic, (352, 352), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        
        # pic = pic.astype('float32') / 255
        # mask = mask.astype('float32') / 255    
     
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
            
        return img_x, img_y

    def __len__(self):
        return len(self.pics)
