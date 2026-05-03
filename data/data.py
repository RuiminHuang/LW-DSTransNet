import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from natsort import natsorted

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random

import numpy as np

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())




class GrokCV_NUAA_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data2/hrm/Datasets/GrokCV/NUAA-SIRST/NUAA'

        self.mean = [0.442, 0.442, 0.442]
        self.std = [0.111, 0.111, 0.111]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        # self.base_size = 512 # default
        # self.crop_size = 512 # default
        self.base_size = 512 # for MLP_Net
        self.crop_size = 512 # for MLP_Net

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        # for img in natsorted( os.listdir(self.imgs_dir) ):
        for img in sorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class GrokCV_NUDT_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data2/hrm/Datasets/GrokCV/NUDT-SIRST/NUDT'

        self.mean = [0.424, 0.424, 0.424]
        self.std = [0.130, 0.130, 0.130]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 256
        self.crop_size = 256

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        # for img in natsorted( os.listdir(self.imgs_dir) ):
        for img in sorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class GrokCV_IRSTD_1k(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data2/hrm/Datasets/GrokCV/IRSTD-1k/IRSTD-1k'

        self.mean = [0.343, 0.343, 0.343]
        self.std = [0.157, 0.157, 0.157]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 512
        self.crop_size = 512

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        # for img in natsorted( os.listdir(self.imgs_dir) ):
        for img in sorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask
