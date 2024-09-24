import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import numpy as np
from skimage import io
from scipy.ndimage import zoom

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from networks.compare_models import build_model_from_name
from dataloaders.BRATS_dataloader_new import Hybrid as MyDataset
from dataloaders.BRATS_dataloader_new import ToTensor
from networks.mynet import TwoBranch
from utils import bright, trunc
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/xiaohan/datasets/BRATS_dataset/BRATS_2020_images/selected_images/')
parser.add_argument('--MRIDOWN', type=str, default='4X', help='MRI down-sampling rate')
parser.add_argument('--low_field_SNR', type=int, default=15, help='SNR of the simulated low-field image')
parser.add_argument('--phase', type=str, default='test', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--base_lr', type=float, default=0.0002, help='maximum epoch numaber to train')

parser.add_argument('--model_name', type=str, default='unet_single', help='model_name')
parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')
parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')

# args = parser.parse_args()
from option import args
test_data_path = args.root_path
snapshot_path = "model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def normalize_output(out_img):
    out_img = (out_img - out_img.min())/(out_img.max() - out_img.min() + 1e-8)
    return out_img


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    network = TwoBranch(args).cuda()
    device = torch.device('cuda')
    network.to(device)

    if len(args.gpu.split(',')) > 1:
        network = nn.DataParallel(network)

    db_test = MyDataset(split='test', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if args.phase == 'test':

        save_mode_path = os.path.join(snapshot_path, 'best_checkpoint.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'])
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case/'
        feature_save_path = snapshot_path + '/feature_visualization/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(feature_save_path):
            os.makedirs(feature_save_path)


        t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
        t2_MSE_all, t2_PSNR_all, t2_SSIM_all = [], [], []

        for (sampled_batch, sample_stats) in tqdm(testloader, ncols=70):
            cnt += 1

            print('processing ' + str(cnt) + ' image')
            t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                   sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
            t1_krecon, t2_krecon = sampled_batch['image_krecon'].cuda(), sampled_batch['target_krecon'].cuda()

            t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
            t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]


            t1_out, t2_out = None, None
                

            t2_out = network(t2_in, t1_in)['img_out']
            t2_out_2 = network(t2_in, t1_in)['img_out']

            t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
            t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
            t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
            t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

            if t1_out is not None:
                t1_img = (np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                t1_krecon_img = (np.clip(t1_krecon.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)

            t1_img = (np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
            t2_in_img = (np.clip(t2_in.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
            t2_img = (np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
            t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
            t2_krecon_img = (np.clip(t2_krecon.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
            t2_out_2_img = (np.clip(t2_out_2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)

            io.imsave(save_path + str(cnt) + '_t1.png', bright(t1_img,0,0.8))
            io.imsave(save_path + str(cnt) + '_t2.png', bright(t2_img,0,0.8))
            io.imsave(save_path + str(cnt) + '_t2_in.png', bright(t2_in_img,0,0.8))
            io.imsave(save_path + str(cnt) + '_t2_out.png', bright(t2_out_img,0,0.8))
            io.imsave(save_path + str(cnt) + '_t2_out2.png', bright(t2_out_2_img,0,0.8))


            if t2_out is not None:
                t2_out_img[t2_out_img < 0.0] = 0.0
                t2_img[t2_img < 0.0] = 0.0
                MSE = mean_squared_error(t2_img, t2_out_img)
                PSNR = peak_signal_noise_ratio(t2_img, t2_out_img)
                SSIM = structural_similarity(t2_img, t2_out_img)
                t2_MSE_all.append(MSE)
                t2_PSNR_all.append(PSNR)
                t2_SSIM_all.append(SSIM)
                print("[t2 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM)

            # if cnt > 20:
            #     break

        print("[T2 MRI:] average MSE:", np.array(t2_MSE_all).mean(), "average PSNR:", np.array(t2_PSNR_all).mean(), "average SSIM:", np.array(t2_SSIM_all).mean())
        print("[T2 MRI:] average MSE:", np.array(t2_MSE_all).std(), "average PSNR:", np.array(t2_PSNR_all).std(), "average SSIM:", np.array(t2_SSIM_all).std())

