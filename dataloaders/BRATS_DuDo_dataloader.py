"""
dual-domain network的dataloader, 读取两个模态的under-sampled和fully-sampled kspace data, 以及high-quality image作为监督信号。
"""


from __future__ import print_function, division
import numpy as np
import pandas as pd
from glob import glob
import random
from skimage import transform
from PIL import Image

import cv2
import os
import torch
from torch.utils.data import Dataset

from .kspace_subsample import undersample_mri, mri_fft


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data (torch.Tensor): Input data to be normalized.
        mean (float): Mean value.
        stddev (float): Standard deviation.
        eps (float, default=0.0): Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std



def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.

    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)



def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.

    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to ifftshift.

    Returns:
        torch.Tensor: ifftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]

    return roll(x, shift, dim)



def ifft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )

    return data



class Hybrid(Dataset):

    def __init__(self, base_dir=None, HF_refine = 'False', split='train', MRIDOWN='4X', SNR=15, \
                 transform=None, input_round = None, input_normalize=None):

        super().__init__()
        self._base_dir = base_dir
        self.HF_refine = HF_refine
        self.input_round = input_round
        self._MRIDOWN = MRIDOWN
        self._SNR = SNR
        self.im_ids = []
        self.t2_images = []
        self.splits_path = "/data/xiaohan/BRATS_dataset/cv_splits_100patients/"

        if split=='train':
            self.train_file = self.splits_path + 'train_data.csv'
            train_images = pd.read_csv(self.train_file).iloc[:, -1].values.tolist()
            self.t1_images = [image for image in train_images if image.split('_')[-1]=='t1.png']

        elif split=='test':
            self.test_file = self.splits_path + 'test_data.csv'
            # self.test_file = self.splits_path + 'train_data.csv'
            test_images = pd.read_csv(self.test_file).iloc[:, -1].values.tolist()
            # test_images = os.listdir(self._base_dir)
            self.t1_images = [image for image in test_images if image.split('_')[-1]=='t1.png']

        
        for image_path in self.t1_images:
            t2_path = image_path.replace('t1', 't2')
            self.t2_images.append(t2_path)


        self.transform = transform
        self.input_normalize = input_normalize

        assert (len(self.t1_images) == len(self.t2_images))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.t1_images)))

    def __len__(self):
        return len(self.t1_images)


    def __getitem__(self, index):

        image_name = self.t1_images[index].split('t1')[0]
        # print("image name:", image_name)

        t1 = np.array(Image.open(self._base_dir + self.t1_images[index]))/255.0
        t2 = np.array(Image.open(self._base_dir + self.t2_images[index]))/255.0
        # print("images:", t1_in.shape, t1.shape, t2_in.shape, t2.shape)
        # print("t1 before standardization:", t1.max(), t1.min(), t1.mean())
        # print("loaded t1 range:", t1.max(), t1.min())
        # print("loaded t2 range:", t2.max(), t2 .min())

        ### normalize the MRI image by divide_max
        t1_max, t2_max = t1.max(), t2.max()
        t1 = t1/t1_max
        t2 = t2/t2_max
        sample_stats = {"t1_max": t1_max, "t2_max": t2_max, "image_name": image_name}

        # sample_stats = {"t1_max": 1.0, "t2_max": 1.0}

        ### convert images to kspace and perform undersampling.
        t1_kspace_in, t1_in, t1_kspace, t1_img = mri_fft(t1, _SNR = self._SNR)
        t2_kspace_in, t2_in, t2_kspace, t2_img, mask = undersample_mri(
            t2, _MRIDOWN = self._MRIDOWN, _SNR = self._SNR)
        

        # print("loaded t2 range:", t2.max(), t2.min())
        # print("t2_under_img range:", t2_under_img.max(), t2_under_img.min())
        # print("t2_kspace real_part range:", t2_kspace.real.max(), t2_kspace.real.min())
        # print("t2_kspace imaginary_part range:", t2_kspace.imag.max(), t2_kspace.imag.min())
        # print("t2_kspace_in real_part range:", t2_kspace_in.real.max(), t2_kspace_in.real.min())
        # print("t2_kspace_in imaginary_part range:", t2_kspace_in.imag.max(), t2_kspace_in.imag.min())

        if self.HF_refine == "False":
            sample = {'t1': t1_img, 't1_in': t1_in, 't1_kspace': t1_kspace, 't1_kspace_in': t1_kspace_in, \
                        't2': t2_img, 't2_in': t2_in, 't2_kspace': t2_kspace, 't2_kspace_in': t2_kspace_in, \
                        't2_mask': mask}

        elif self.HF_refine == "True":
            ### 读取上一步重建的kspace data.
            t1_krecon_path = self._base_dir + self.t1_images[index].replace(
                't1.png', 't1_' + str(self._SNR) + 'dB_recon_kspace_' + self.input_round + '_DudoLoss.npy')
            t2_krecon_path = self._base_dir + self.t1_images[index].replace('t1.png', 't2_' + self._MRIDOWN + \
                '_' + str(self._SNR) + 'dB_recon_kspace_' + self.input_round + '_DudoLoss.npy')  

            t1_krecon = np.load(t1_krecon_path)
            t2_krecon = np.load(t2_krecon_path)
            # print("t1 and t2 recon kspace:", t1_krecon.shape, t2_krecon.shape) 
            #  
            sample = {'t1': t1_img, 't1_in': t1_in, 't1_kspace': t1_kspace, 't1_kspace_in': t1_kspace_in, \
                        't2': t2_img, 't2_in': t2_in, 't2_kspace': t2_kspace, 't2_kspace_in': t2_kspace_in, \
                        't2_mask': mask, 't1_krecon': t1_krecon, 't2_krecon': t2_krecon}
    

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, sample_stats


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image'][:, :, None].transpose((2, 0, 1))
        target = sample['target'][:, :, None].transpose((2, 0, 1))
        # print("img_in before_numpy range:", img_in.max(), img_in.min())
        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).float()
        # print("img_in range:", img_in.max(), img_in.min())

        return {'ct': img, 'mri': target}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         img_in = sample['image_in'][:, :, None].transpose((2, 0, 1))
#         img = sample['image'][:, :, None].transpose((2, 0, 1))
#         target_in = sample['target_in'][:, :, None].transpose((2, 0, 1))
#         target = sample['target'][:, :, None].transpose((2, 0, 1))
#         # print("img_in before_numpy range:", img_in.max(), img_in.min())
#         img_in = torch.from_numpy(img_in).float()
#         img = torch.from_numpy(img).float()
#         target_in = torch.from_numpy(target_in).float()
#         target = torch.from_numpy(target).float()
#         # print("img_in range:", img_in.max(), img_in.min())

#         return {'ct_in': img_in,
#                 'ct': img,
#                 'mri_in': target_in,
#                 'mri': target}
