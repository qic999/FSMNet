import os
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
from option import args
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


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

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    network = TwoBranch(args).cuda()
    # network = build_model_from_name(args).cuda()
    device = torch.device('cuda')
    network.to(device)

    if len(args.gpu.split(',')) > 1:
        network = nn.DataParallel(network)
        # network = nn.SyncBatchNorm.convert_sync_batchnorm(network)
    
    # print("network architecture:", network)
    
    n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('number of params: %.2f M' % (n_parameters / 1024 / 1024))

    db_train = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                         split='train', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        #  transform=transforms.Compose([RandomPadCrop(), ToTensor(), AddNoise()]),
                         transform=transforms.Compose([RandomPadCrop(), ToTensor()]),
                         base_dir=train_data_path, input_normalize = args.input_normalize)
    
    db_test = MyDataset(kspace_refine=args.kspace_refine, kspace_round = args.kspace_round,
                        split='test', MRIDOWN=args.MRIDOWN, SNR=args.low_field_SNR, 
                        transform=transforms.Compose([ToTensor()]),
                        base_dir=test_data_path, input_normalize = args.input_normalize)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    fixtrainloader = DataLoader(db_train, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.phase == 'train':
        network.train()

        params = list(network.parameters())
        optimizer1 = optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20000, gamma=0.5)

        writer = SummaryWriter(snapshot_path + '/log')

        iter_num = 0
        max_epoch = max_iterations // len(trainloader) + 1


        best_status = {'T1_NMSE': 10000000, 'T1_PSNR': 0, 'T1_SSIM': 0,
                       'T2_NMSE': 10000000, 'T2_PSNR': 0, 'T2_SSIM': 0}
        fft_weight=0.01
        criterion = nn.L1Loss().to(device, non_blocking=True)
        amploss = AMPLoss().to(device, non_blocking=True)
        phaloss = PhaLoss().to(device, non_blocking=True)
        for epoch_num in tqdm(range(max_epoch), ncols=70):
            time1 = time.time()
            for i_batch, (sampled_batch, sample_stats) in enumerate(trainloader):
                time2 = time.time()
                # print("time for data loading:", time2 - time1)

                t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                       sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
                
                t1_krecon, t2_krecon = sampled_batch['image_krecon'].cuda(), sampled_batch['target_krecon'].cuda()
                

                time3 = time.time()


                outputs = network(t2_in, t1_in)

                loss = criterion(outputs['img_out'], t2) + \
                        fft_weight * amploss(outputs['img_fre'], t2) + fft_weight * phaloss(
                        outputs['img_fre'],
                        t2) + \
                        criterion(outputs['img_fre'], t2)

                time4 = time.time()
                optimizer1.zero_grad()
                loss.backward()

                if args.clip_grad == "True":
                    ### clip the gradients to a small range.
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.01)

                optimizer1.step()
                scheduler1.step()

                time5 = time.time()

                # summary
                iter_num = iter_num + 1
                # writer.add_scalar('lr', scheduler1.get_lr(), iter_num)
                # writer.add_scalar('loss/loss', loss, iter_num)

                if iter_num % 100 == 0:
                    logging.info('iteration %d : learning rate : %f loss : %f ' % (iter_num, scheduler1.get_lr()[0], loss.item()))
    
                if iter_num % 20000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save({'network': network.state_dict()}, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num > max_iterations:
                    break
                time1 = time.time()
            
            
            ## ================ Evaluate ================
            logging.info(f'Epoch {epoch_num} Evaluation:')
            # print()
            t1_MSE_all, t1_PSNR_all, t1_SSIM_all = [], [], []
            t2_MSE_all, t2_PSNR_all, t2_SSIM_all = [], [], []

            t1_MSE_krecon, t1_PSNR_krecon, t1_SSIM_krecon = [], [], []
            t2_MSE_krecon, t2_PSNR_krecon, t2_SSIM_krecon = [], [], []

            for (sampled_batch, sample_stats) in testloader:
                
                t1_in, t1, t2_in, t2 = sampled_batch['image_in'].cuda(), sampled_batch['image'].cuda(), \
                                       sampled_batch['target_in'].cuda(), sampled_batch['target'].cuda()
                
                t1_krecon, t2_krecon = sampled_batch['image_krecon'].cuda(), sampled_batch['target_krecon'].cuda()
                t_merge = torch.cat([t1_in, t2_in], dim=1)

                t2_out = network(t2_in, t1_in)['img_out']
                t1_out = None

                if args.input_normalize == "mean_std":
                    t1_mean = sample_stats['t1_mean'].data.cpu().numpy()[0]
                    t1_std = sample_stats['t1_std'].data.cpu().numpy()[0]
                    t2_mean = sample_stats['t2_mean'].data.cpu().numpy()[0]
                    t2_std = sample_stats['t2_std'].data.cpu().numpy()[0]

                    if t1_out is not None:
                        t1_img = (np.clip(t1.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                        t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)
                        t1_krecon_img = (np.clip(t1_krecon.data.cpu().numpy()[0, 0] * t1_std + t1_mean, 0, 1) * 255).astype(np.uint8)

 
                    t2_img = (np.clip(t2.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)
                    t2_krecon_img = (np.clip(t2_krecon.data.cpu().numpy()[0, 0] * t2_std + t2_mean, 0, 1) * 255).astype(np.uint8)

                else:
                    if t1_out is not None:
                        t1_img = (np.clip(t1.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                        t1_out_img = (np.clip(t1_out.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                        t1_krecon_img = (np.clip(t1_krecon.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
 
                    t2_img = (np.clip(t2.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                    t2_out_img = (np.clip(t2_out.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
                    t2_krecon_img = (np.clip(t2_krecon.data.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)


                if t1_out is not None:

                    MSE = mean_squared_error(t1_img, t1_out_img)
                    PSNR = peak_signal_noise_ratio(t1_img, t1_out_img)
                    SSIM = structural_similarity(t1_img, t1_out_img)
                    t1_MSE_all.append(MSE)
                    t1_PSNR_all.append(PSNR)
                    t1_SSIM_all.append(SSIM)

                    MSE = mean_squared_error(t1_img, t1_krecon_img)
                    PSNR = peak_signal_noise_ratio(t1_img, t1_krecon_img)
                    SSIM = structural_similarity(t1_img, t1_krecon_img)
                    t1_MSE_krecon.append(MSE)
                    t1_PSNR_krecon.append(PSNR)
                    t1_SSIM_krecon.append(SSIM)


                if t2_out is not None:
                    MSE = mean_squared_error(t2_img, t2_out_img)
                    PSNR = peak_signal_noise_ratio(t2_img, t2_out_img)
                    SSIM = structural_similarity(t2_img, t2_out_img)
                    t2_MSE_all.append(MSE)
                    t2_PSNR_all.append(PSNR)
                    t2_SSIM_all.append(SSIM)
                    # print("[t2 MRI] MSE:", MSE, "PSNR:", PSNR, "SSIM:", SSIM)
                    
                    MSE = mean_squared_error(t2_img, t2_krecon_img)
                    PSNR = peak_signal_noise_ratio(t2_img, t2_krecon_img)
                    SSIM = structural_similarity(t2_img, t2_krecon_img)
                    t2_MSE_krecon.append(MSE)
                    t2_PSNR_krecon.append(PSNR)
                    t2_SSIM_krecon.append(SSIM)

            if t1_out is not None:
                t1_mse = np.array(t1_MSE_all).mean()
                t1_psnr = np.array(t1_PSNR_all).mean()
                t1_ssim = np.array(t1_SSIM_all).mean()

                t1_krecon_mse = np.array(t1_MSE_krecon).mean()
                t1_krecon_psnr = np.array(t1_PSNR_krecon).mean()
                t1_krecon_ssim = np.array(t1_SSIM_krecon).mean()

            t2_mse = np.array(t2_MSE_all).mean()
            t2_psnr = np.array(t2_PSNR_all).mean()
            t2_ssim = np.array(t2_SSIM_all).mean()
                
            t2_krecon_mse = np.array(t2_MSE_krecon).mean()
            t2_krecon_psnr = np.array(t2_PSNR_krecon).mean()
            t2_krecon_ssim = np.array(t2_SSIM_krecon).mean()


            if t2_psnr > best_status['T2_PSNR']:
                best_status = {'T2_NMSE': t2_mse, 'T2_PSNR': t2_psnr, 'T2_SSIM': t2_ssim}

                best_checkpoint_path = os.path.join(snapshot_path, 'best_checkpoint.pth')

                torch.save({'network': network.state_dict()}, best_checkpoint_path)
                print('New Best Network:')

            logging.info(f"[T2 MRI:] average MSE: {t2_mse} average PSNR: {t2_psnr} average SSIM: {t2_ssim}")

            if args.kspace_refine == "True":
                logging.info(f"[T1 MRI (krecon_input):] average MSE: {t1_krecon_mse} average PSNR: {t1_krecon_psnr} average SSIM: {t1_krecon_ssim}")
                logging.info(f"[T2 MRI (krecon_input):] average MSE: {t2_krecon_mse} average PSNR: {t2_krecon_psnr} average SSIM: {t2_krecon_ssim}")

            if iter_num > max_iterations:
                break
        print(best_status)
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save({'network': network.state_dict()},
                   save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        writer.close()
