#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
#from F_try import FrequencyIndex

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask,fre):
        image = (image - self.mean)/self.std
        mask /= 255
        fre /= 255
        # fre_ = torch.tensor(fre)
        # fre_ = fre_.view(-1)
        # max,_ = torch.topk(fre_,100)
        # print(max)
        return image, mask, fre

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask,mask1,mask2):
        if np.random.randint(2)==0:
            return image[:, ::-1].copy(), mask[:, ::-1].copy(), mask1[:, ::-1].copy(), mask2[:, ::-1].copy()
        else:
            return image, mask,mask1,mask2

class RandomRotate(object):
    def __call__(self, image, mask):
        degree = 10
        rows, cols, channels = image.shape
        random_rotate = random.random() * 2 * degree - degree
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), random_rotate, 1)
        '''
        第一个参数：旋转中心点
        第二个参数：旋转角度
        第三个参数：缩放比例
        '''
        image = cv2.warpAffine(image, rotate, (cols, rows))
        mask = cv2.warpAffine(mask, rotate, (cols, rows))
        # contour = cv2.warpAffine(contour, rotate, (cols, rows))

        return image,mask



class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,fre):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask1  = cv2.resize( mask, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        mask2  = cv2.resize( mask, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        fre  = cv2.resize( fre, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        return image, mask1,mask2,fre

class ToTensor(object):
    def __call__(self, image, mask,mask1,mask2):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask1  = torch.from_numpy(mask1)
        mask2  = torch.from_numpy(mask2)
        return image, mask,mask1,mask2


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.randomrotate = RandomRotate()
        self.resize     = Resize(512, 512)
        #self.fre = FrequencyIndex()
        #self.resize1     = cv2.resize((352, 352), interpolation=cv2.INTER_NEAREST)
        self.totensor   = ToTensor()

        self.root = cfg.datapath

        img_path = os.path.join(self.root, 'Image')
        fre_path = os.path.join(self.root, 'Frequency_2')
        gt_path = os.path.join(self.root, 'Masks')
        self.samples = [os.path.splitext(f)[0]
                    for f in os.listdir(gt_path) if f.endswith('.png')]



    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.root+'/Image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.root+'/Masks/' +name+'.png', 0).astype(np.float32)
        fre  = cv2.imread(self.root+'/Frequency_2/' +name+'.jpg', 0).astype(np.float32)
        #print(fre.shape)

        shape = mask.shape

        if self.cfg.mode=='train':
            image, mask,fre = self.normalize(image, mask,fre)
            image, mask1,mask2,fre = self.resize(image, mask,fre)
#             # image, mask = self.randomcrop(image, mask)
            image, mask1,mask2,fre = self.randomflip(image, mask1,mask2,fre)
#             image, mask = self.randomrotate(image, mask)
            image, mask1,mask2,fre = self.totensor(image, mask1,mask2,fre)
            #print(name)
            #_ = self.fre(image.unsqueeze(0).float())
            return image, mask1,mask2,fre
        else:
            image, mask,fre = self.normalize(image, mask,fre)
            image, mask1,mask2,fre = self.resize(image,mask,fre)
            image, mask1,mask2,fre = self.totensor(image, mask1,mask2,fre)
            return image, mask1, fre, name

    def __len__(self):
        return len(self.samples)


# train_path = 'train'
# cfg = Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
# data = Data(cfg)
# print(data[0][-2].size())


# def check(a):
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             if a[i][j] != 0 and a[i][j] != 1:
#                 print(a[i][j])
#             else:
#                 print('love')
                
# for i in range(len(data)):
#     a = data[i][1]
#     check(a)

