from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
# from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 读取中文路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

class BasicDataset(Dataset):
    def __init__(self, img_path, mask_path, patch=512):
        self.img = self.preprocess(img_path,'img')
        self.mask = self.preprocess(mask_path,'mask')
        h, w = min(self.img.shape[1],self.mask.shape[1]),min(self.img.shape[2],self.mask.shape[2])
        self.img,self.mask = self.img[:,:h,:w],self.mask[:,:h,:w]
        self.patch = patch
        self.cols = w//patch
        self.rows = h//patch
        self.ids = self.filt_idx(range(self.cols*self.rows))
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def filt_idx(self,ids):
        r = []
        for id in ids:
            mask = self._read_mask(id)
            if (mask==1).sum()/self.patch**2 > 0.2:
                r.append(id)
        return r


    def _read_mask(self,idx):
        row = idx // self.cols
        col = idx % self.cols
        mask = self.mask[:,row*self.patch:row*self.patch+self.patch,col*self.patch:col*self.patch+self.patch]
        return mask

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, path,type):
        img_nd = cv_imread(path)
        w, h = img_nd.shape[:2]
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if type == 'mask':
            img_trans[img_trans==255]=0
        else:
            img_trans = img_trans / 255

        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        row = idx//self.cols
        col = idx%self.cols
        mask = self.mask[:,row*self.patch:row*self.patch+self.patch,col*self.patch:col*self.patch+self.patch]
        img = self.img[:,row*self.patch:row*self.patch+self.patch,col*self.patch:col*self.patch+self.patch]

        if 0:
            cv2.imshow('mask',np.transpose(mask,(1,2,0))*255)
            cv2.imshow('img',np.transpose(img,(1,2,0)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
