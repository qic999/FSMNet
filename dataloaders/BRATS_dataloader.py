from __future__ import print_function, division
import numpy as np
import pandas as pd
from glob import glob
import random
from skimage import transform
from PIL import Image

import os
import torch
from torch.utils.data import Dataset

class Hybrid(Dataset):

    def __init__(self, base_dir=None, split='train', MRIDOWN='4X', SNR=15, transform=None):

        super().__init__()
        self._base_dir = base_dir
        self._MRIDOWN = MRIDOWN
        self.im_ids = []
        self.t2_images = []
        self.t1_undermri_images, self.t2_undermri_images = [], []
        self.splits_path = "/home/xiaohan/datasets/BRATS_dataset/BRATS_2020_images/cv_splits/"

        if split=='train':
            self.train_file = self.splits_path + 'train_data.csv'
            train_images = pd.read_csv(self.train_file).iloc[:, -1].values.tolist()
            self.t1_images = [image for image in train_images if image.split('_')[-1]=='t1.png']

        elif split=='test':
            self.test_file = self.splits_path + 'test_data.csv'
            test_images = pd.read_csv(self.test_file).iloc[:, -1].values.tolist()
            # test_images = os.listdir(self._base_dir)
            self.t1_images = [image for image in test_images if image.split('_')[-1]=='t1.png']

        
        for image_path in self.t1_images:
            t2_path = image_path.replace('t1', 't2')
            if SNR == 0:
                # t1_under_path = image_path.replace('t1', 't1_' + self._MRIDOWN + '_undermri')
                t1_under_path = image_path
                t2_under_path = image_path.replace('t1', 't2_' + self._MRIDOWN + '_undermri')
            else:
                # t1_under_path = image_path.replace('t1', 't1_' + self._MRIDOWN + '_' + str(SNR) + 'dB_undermri')
                t1_under_path = image_path.replace('t1', 't1_' + str(SNR) + 'dB')
                if MRIDOWN == "False":
                    t2_under_path = image_path.replace('t1', 't2_' + str(SNR) + 'dB')
                else:
                    t2_under_path = image_path.replace('t1', 't2_' + self._MRIDOWN + '_' + str(SNR) + 'dB_undermri')

            # print("image paths:", image_path, t1_under_path, t2_path, t2_under_path)

            self.t2_images.append(t2_path)
            self.t1_undermri_images.append(t1_under_path)
            self.t2_undermri_images.append(t2_under_path)

        # print("t1 images:", self.t1_images)
        # print("t2 images:", self.t2_images)
        # print("t1_undermri_images:", self.t1_undermri_images)
        # print("t2_undermri_images:", self.t2_undermri_images)

        self.transform = transform

        assert (len(self.t1_images) == len(self.t2_images))
        assert (len(self.t1_images) == len(self.t1_undermri_images))
        assert (len(self.t1_images) == len(self.t2_undermri_images))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.t1_images)))

    def __len__(self):
        return len(self.t1_images)


    def __getitem__(self, index):
        ### 两种settings. 
        ### 1. T1 fully-sampled 不加noise, T2 down-sampled, 做MRI acceleration.
        ### 2. T1 fully-sampled 但是加noise, T2 down-sampled同时也加noise, 同时做MRI acceleration and enhancement.
        ### T1, T2两个模态的输入都是low-quality images.
        sample = {'image_in': np.array(Image.open(self._base_dir + self.t1_undermri_images[index]))/255.0, 
                  'image': np.array(Image.open(self._base_dir + self.t1_images[index]))/255.0, 
                  'target_in': np.array(Image.open(self._base_dir + self.t2_undermri_images[index]))/255.0, 
                  'target': np.array(Image.open(self._base_dir + self.t2_images[index]))/255.0}


        # ### 2023/05/23, Xiaohan, 把T1模态的输入改成high-quality图像（和ground truth一致，看能否为T2提供更好的guidance）。
        # sample = {'image_in': np.array(Image.open(self._base_dir + self.t1_images[index]))/255.0, 
        #           'image': np.array(Image.open(self._base_dir + self.t1_images[index]))/255.0, 
        #           'target_in': np.array(Image.open(self._base_dir + self.t2_undermri_images[index]))/255.0, 
        #           'target': np.array(Image.open(self._base_dir + self.t2_images[index]))/255.0}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class RandomPadCrop(object):
    def __call__(self, sample):
        new_w, new_h = 256, 256
        crop_size = 240
        pad_size = (256-240)//2
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        img_in = np.pad(img_in, pad_size, mode='reflect')
        img = np.pad(img, pad_size, mode='reflect')
        target_in = np.pad(target_in, pad_size, mode='reflect')
        target = np.pad(target, pad_size, mode='reflect')

        ww = random.randint(0, np.maximum(0, new_w - crop_size))
        hh = random.randint(0, np.maximum(0, new_h - crop_size))

        # print("img_in:", img_in.shape)
        img_in = img_in[ww:ww+crop_size, hh:hh+crop_size]
        img = img[ww:ww+crop_size, hh:hh+crop_size]
        target_in = target_in[ww:ww+crop_size, hh:hh+crop_size]
        target = target[ww:ww+crop_size, hh:hh+crop_size]

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        return sample


class RandomResizeCrop(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        new_w, new_h = 270, 270
        crop_size = 256
        img_in = sample['image_in']
        img = sample['image']
        target_in = sample['target_in']
        target = sample['target']

        img_in = transform.resize(img_in, (new_h, new_w), order=3)
        img = transform.resize(img, (new_h, new_w), order=3)
        target_in = transform.resize(target_in, (new_h, new_w), order=3)
        target = transform.resize(target, (new_h, new_w), order=3)

        ww = random.randint(0, np.maximum(0, new_w - crop_size))
        hh = random.randint(0, np.maximum(0, new_h - crop_size))

        img_in = img_in[ww:ww+crop_size, hh:hh+crop_size]
        img = img[ww:ww+crop_size, hh:hh+crop_size]
        target_in = target_in[ww:ww+crop_size, hh:hh+crop_size]
        target = target[ww:ww+crop_size, hh:hh+crop_size]

        sample = {'image_in': img_in, 'image': img, 'target_in': target_in, 'target': target}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['image_in'][:, :, None].transpose((2, 0, 1))
        img = sample['image'][:, :, None].transpose((2, 0, 1))
        target_in = sample['target_in'][:, :, None].transpose((2, 0, 1))
        target = sample['target'][:, :, None].transpose((2, 0, 1))
        img_in = torch.from_numpy(img_in).float()
        img = torch.from_numpy(img).float()
        target_in = torch.from_numpy(target_in).float()
        target = torch.from_numpy(target).float()

        return {'ct_in': img_in,
                'ct': img,
                'mri_in': target_in,
                'mri': target}
