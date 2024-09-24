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
from dataloaders.fastmri import build_dataset


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
    for name, param in network.named_parameters():
        if param.grad is not None:
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

from metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    input_dic = defaultdict(dict)

    for data in data_loader:
        pd, pdfs, _ = data
        target = pdfs[1]

        mean = pdfs[2]
        std = pdfs[3]

        fname = pdfs[4]
        slice_num = pdfs[5]

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        mean = mean.to(device)
        std = std.to(device)

        pd_img = pd[1].unsqueeze(1)
        pdfs_img = pdfs[0].unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)


        outputs = model(pdfs_img, pd_img)['img_out']
        outputs = outputs.squeeze(1)

        outputs = outputs * std + mean
        target = target * std + mean
        inputs = pdfs_img.squeeze(1) * std + mean

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = outputs[i]
            target_dic[f][slice_num[i]] = target[i]
            input_dic[f][slice_num[i]] = inputs[i]


    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)

    print("==> Evaluate Metric")
    print("Results ----------")
    print("NMSE: {:.4}".format(nmse_meter.avg))
    print("PSNR: {:.4}".format(psnr_meter.avg))
    print("SSIM: {:.4}".format(ssim_meter.avg))
    print("------------------")
    model.train()
    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}



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

    db_train = build_dataset(args, mode='train')
    db_test = build_dataset(args, mode='val')

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.phase == 'train':
        network.train()

        params = list(network.parameters())
        optimizer1 = optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20000, gamma=0.5)

        writer = SummaryWriter(snapshot_path + '/log')

        iter_num = 0
        max_epoch = max_iterations // len(trainloader) + 1


        best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0}
        fft_weight=0.01
        criterion = nn.L1Loss().to(device, non_blocking=True)
        amploss = AMPLoss().to(device, non_blocking=True)
        phaloss = PhaLoss().to(device, non_blocking=True)
        for epoch_num in tqdm(range(max_epoch), ncols=70):
            time1 = time.time()
            for i_batch, sampled_batch in enumerate(trainloader):
                time2 = time.time()
                # print("time for data loading:", time2 - time1)

                pd, pdfs, _ = sampled_batch
                target = pdfs[1]

                pd_img = pd[1].unsqueeze(1)
                pdfs_img = pdfs[0].unsqueeze(1)
                target = target.unsqueeze(1)

                pd_img = pd_img.to(device) # [4, 1, 320, 320]
                pdfs_img = pdfs_img.to(device) # [4, 1, 320, 320]
                target = target.to(device) # [4, 1, 320, 320]

                time3 = time.time()
                # breakpoint()
                outputs = network(pdfs_img, pd_img)

                loss = criterion(outputs['img_out'], target) + \
                        fft_weight * amploss(outputs['img_fre'], target) + fft_weight * phaloss(
                        outputs['img_fre'],
                        target) + \
                        criterion(outputs['img_fre'], target)

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
            eval_result = evaluate(network, testloader, device)

            if eval_result['PSNR'] > best_status['PSNR']:
                best_status = {'NMSE': eval_result['NMSE'], 'PSNR': eval_result['PSNR'], 'SSIM': eval_result['SSIM']}
                best_checkpoint_path = os.path.join(snapshot_path, 'best_checkpoint.pth')

                torch.save({'network': network.state_dict()}, best_checkpoint_path)
                print('New Best Network:')
            logging.info(f"average MSE: {eval_result['NMSE']} average PSNR: {eval_result['PSNR']} average SSIM: {eval_result['SSIM']}")

            if iter_num > max_iterations:
                break
        print(best_status)
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
        torch.save({'network': network.state_dict()},
                   save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        writer.close()
        