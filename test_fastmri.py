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

from option import args

test_data_path = args.root_path
snapshot_path = "model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def normalize_output(out_img):
    out_img = (out_img - out_img.min())/(out_img.max() - out_img.min() + 1e-8)
    return out_img

from metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict
@torch.no_grad()
def evaluate(model, data_loader, device, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    input_dic = defaultdict(dict)
    flag=0
    last_name='no'
    for data in data_loader:
        pd, pdfs, _ = data
        name = os.path.basename(pdfs[4][0]).split('.')[0]
        if not last_name == name:
            last_name = name
            flag+=1
        if flag < 3:
            continue
        elif flag >= 4:
            break
        else:
            pass

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

        outputs = network(pdfs_img, pd_img)['img_out']
        outputs = outputs.squeeze(1)

        outputs_save = outputs[0].cpu().numpy()/6.0
        outputs_save = np.clip(outputs_save, a_min=-1, a_max=1)
        io.imsave(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '.png', target[0].cpu().numpy()/6.0)
        io.imsave(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_in.png', pdfs_img[0][0].cpu().numpy()/6.0)
        io.imsave(save_path + str(name) + '_' + str(slice_num[0].cpu().numpy()) + '_out.png', outputs_save)

        outputs = outputs * std + mean
        target = target * std + mean
        inputs = pdfs_img.squeeze(1) * std + mean

        output_dic[fname[0]][slice_num[0]] = outputs[0]
        target_dic[fname[0]][slice_num[0]] = target[0]
        input_dic[fname[0]][slice_num[0]] = inputs[0]
        our_nmse = nmse(target[0].cpu().numpy(), outputs[0].cpu().numpy())
        our_psnr = psnr(target[0].cpu().numpy(), outputs[0].cpu().numpy())
        our_ssim = ssim(target[0].cpu().numpy(), outputs[0].cpu().numpy())
            
        print('name:{}, slice:{}, psnr:{}, ssim:{}'.format(name, slice_num[0], our_psnr, our_ssim))


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
    print("NMSE: {:.4}".format(np.array(nmse_meter.score).mean()))
    print("PSNR: {:.4}".format(np.array(psnr_meter.score).mean()))
    print("SSIM: {:.4}".format(np.array(ssim_meter.score).mean()))
    print("NMSE: {:.4}".format(np.array(nmse_meter.score).std()))
    print("PSNR: {:.4}".format(np.array(psnr_meter.score).std()))
    print("SSIM: {:.4}".format(np.array(ssim_meter.score).std()))
    print("------------------")
    model.train()
    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}

from dataloaders.fastmri import build_dataset
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


    db_test = build_dataset(args, mode='val')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if args.phase == 'test':

        save_mode_path = os.path.join(snapshot_path, 'best_checkpoint.pth')
        print('load weights from ' + save_mode_path)
        checkpoint = torch.load(save_mode_path)
        
        
        weights_dict = {}
        for k, v in checkpoint['network'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        # breakpoint()
        network.load_state_dict(weights_dict)
        network.eval()

        eval_result = evaluate(network, testloader, device, save_path = snapshot_path + '/result_case/')



