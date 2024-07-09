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

### Xiaohan, add evaluation metrics
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

# parser.add_argument('--input_dim', type=int, default=1, help='number of channels of the input image')
# parser.add_argument('--output_dim', type=int, default=1, help='number of channels of the reconstructed image')
parser.add_argument('--model_name', type=str, default='unet_single', help='model_name')
parser.add_argument('--use_multi_modal', type=str, default='False', help='whether use multi-modal data for MRI reconstruction')
parser.add_argument('--modality', type=str, default='t2', help='MRI modality')
parser.add_argument('--input_modality', type=str, default='t2', help='input MRI modality')

parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')

parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')
parser.add_argument('--kspace_refine', type=str, default='False', \
                    help='use the original under-sampled input or the kspace-interpolated input')

parser.add_argument('--kspace_round', type=str, default='round4', help='use which round of kspace_recon as model input')


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

    # network = build_model_from_name(args).cuda()
    network = TwoBranch(args).cuda()
    device = torch.device('cuda')
    network.to(device)

    if len(args.gpu.split(',')) > 1:
        network = nn.DataParallel(network)

    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round, 
                        split='test', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if args.phase == 'test':

        save_mode_path = os.path.join(snapshot_path, 'best_checkpoint.pth')
        # save_mode_path = os.path.join(snapshot_path, 'best.pth')
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

        # for name, param in network.named_parameters():
        #     print(name, param)

        t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
        t2_MSE_all, t2_PSNR_all, t2_SSIM_all = [], [], []

        for (sampled_batch, sample_stats) in tqdm(testloader, ncols=70):
            cnt += 1
            if cnt > 300 and cnt < 310:
                pass
            elif cnt > 310:
                break
            else:
                continue

            print('processing ' + str(cnt) + ' image')
            t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                   sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
            t1_krecon, t2_krecon = sampled_batch['image_krecon'].cuda(), sampled_batch['target_krecon'].cuda()

            # t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
            # t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
            t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
            t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]


            t1_out, t2_out = None, None

            if args.use_multi_modal == 'True':
                if args.modality == "both":
                    # t1_out, t2_out = network(t1_in, t2_in)
                    t1_out, t2_out = network(t1_in, t2_in, t1_krecon, t2_krecon)

                elif args.modality == "t1":
                    t1_out = network(t1_in, t2_in)
                elif args.modality == "t2":
                    t2_out = network(t2_in, t1_in)

                if args.input_normalize == "mean_std":
                    ### 按照 x*std + mean把图像变回原来的特征范围
                    t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
                    t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
                    t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
                    t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

                    t1_img = (normalize_output(np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1)) * 255).astype(np.uint8)
                    t2_img = (normalize_output(np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1)) * 255).astype(np.uint8)
                    t1_out_img = (normalize_output(np.clip(t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1)) * 255).astype(np.uint8)
                    t2_out_img = (normalize_output(np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1)) * 255).astype(np.uint8)

                    print("t1_img range:", t1_img.max(), t1_img.min())
                    print("t1_out_img range:", t1_out_img.max(), t1_out_img.min())
                    print("t2_img range:", t2_img.max(), t2_img.min())
                    print("t2_out_img range:", t2_out_img.max(), t2_out_img.min())
           
                
                
            elif args.use_multi_modal == 'False':
                if args.modality == "t1":
                    t1_out = network(t1_in)
                    t1_out[t1_out < 0.0] = 0.0
                    t1_out_img = (t1_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)
                    io.imsave(save_path + str(cnt) + '_t1_out.png', bright(t1_out_img,0,0.8))


                elif args.modality == "t2":
                    if args.input_modality == "t1":
                        t2_out = network(t1_in)
                    elif args.input_modality == "t2":
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


    elif args.phase == "diff":
        save_mode_path = os.path.join(snapshot_path, 'iter_100000.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        network.load_state_dict(checkpoint['network'])
        network.eval()
        cnt = 0
        save_path = snapshot_path + '/result_case_diff/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for sampled_batch in tqdm(testloader, ncols=70):
            print('processing ' + str(cnt) + ' image')
            ct_in, ct, mri_in, mri = sampled_batch['ct_in'].cuda(), sampled_batch['ct'].cuda(), \
                                     sampled_batch['mri_in'].cuda(), sampled_batch['mri'].cuda()

            for idx, lam in enumerate([0, 0.3, 0.5, 0.7, 1]):
                domainness = [torch.tensor(lam).cuda().float().reshape((1, 1))]
                with torch.no_grad():
                    fusion_out = network(ct_in, mri_in, domainness)[0][0]
                fusion_out[fusion_out < 0.0] = 0.0
                fusion_img = (fusion_out.data.cpu().numpy()[0, 0] * 255).astype(np.uint8)

                diff_ct = fusion_out.data.cpu().numpy()[0, 0] - ct.data.cpu().numpy()[0, 0]
                diff_ct = (trunc(diff_ct*255 +135)).astype(np.uint8)

                diff_mri = fusion_out.data.cpu().numpy()[0, 0] - mri.data.cpu().numpy()[0, 0]
                diff_mri = (trunc(diff_mri*255 +135)).astype(np.uint8)

                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_ct.png', diff_ct)
                io.imsave(save_path + 'diff_' + str(cnt) + '_'+ str(lam) + '_mri.png', diff_mri)

            cnt = cnt + 1
            if cnt > 3:
                break
