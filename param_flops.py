"""
Compare with state-of-the-art methods.
Load models from the folder networks/compare_models.
"""

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from networks.compare_models import build_model_from_name
from dataloaders.BRATS_dataloader_new import Hybrid as MyDataset
from dataloaders.BRATS_dataloader_new import RandomPadCrop, ToTensor, AddNoise
from networks.mynet import TwoBranch
# from dataloaders.BRATS_dataloader import Hybrid as MyDataset
# from dataloaders.BRATS_dataloader import RandomPadCrop, ToTensor
from option import args
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


# torch.backends.cudnn.benchmark = True


train_data_path = args.root_path
test_data_path = args.root_path
snapshot_path = "model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr



def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(
        img1 **2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()



def gradient_calllback(network):
    """
    记录Unet_restormer网络中各层特征参数的gradient.
    """
    for name, param in network.named_parameters():
        if param.grad is not None:
            # print("Gradient of {}: {}".format(name, param.grad.abs().mean()))

            if param.grad.abs().mean() == 0:
                print("Gradient of {} is 0".format(name))

        else:
            print("Gradient of {} is None".format(name))

class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)


class PhaLoss(nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)


from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # if args.baseline == 'swinIR':
    #     network = build_model().cuda()
    # elif args.baseline == 'MINet':
    #     network = MINet().cuda()
    # elif args.baseline == 'MCCA':
    #     network = MCCA().cuda()
    # elif args.baseline == 'DCAMSR':
    #     network = DCAMSR().cuda()
    # elif args.baseline == 'MTrans':
    #     cfg = RMC
    #     network = RMC_Model(cfg).cuda()
    # elif args.baseline == 'UNet':
    #     network = Unet().cuda()
    network = TwoBranch(args).cuda()
    # network = build_model_from_name(args).cuda()
    device = torch.device('cpu')
    network.to(device)

    # 创建输入网络的tensor
    tensor = (torch.rand(1, 1, 320, 320),torch.rand(1, 1, 320, 320))
    # tensor = (torch.rand(1, 2, 240, 240))


    # 分析FLOPs
    flops = FlopCountAnalysis(network, tensor)
    print("FLOPs: ", flops.total()/(1024*1024*1024))

    # 分析parameters
    print(parameter_count_table(network))

    from ptflops import get_model_complexity_info
    def prepare_input(resolution):
        x1 = torch.FloatTensor(1, *resolution)
        x2 = torch.FloatTensor(1, *resolution)
        return dict(main=x1, aux=x2)
    macs, params = get_model_complexity_info(network, (1, 240, 240), input_constructor=prepare_input,
                                             as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('FLOPs',macs)
    print('params',params)



