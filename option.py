import argparse
# import os

parser = argparse.ArgumentParser(description='MRI recon')
parser.add_argument('--root_path', type=str, default='/home/xiaohan/datasets/BRATS_dataset/BRATS_2020_images/selected_images/')
parser.add_argument('--MRIDOWN', type=str, default='4X', help='MRI down-sampling rate')
parser.add_argument('--low_field_SNR', type=int, default=15, help='SNR of the simulated low-field image')
parser.add_argument('--phase', type=str, default='train', help='Name of phase')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--exp', type=str, default='msl_model', help='model_name')
parser.add_argument('--max_iterations', type=int, default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0002, help='maximum epoch numaber to train')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--resume', type=str, default=None, help='resume')

parser.add_argument('--model_name', type=str, default='unet_single', help='model_name')
parser.add_argument('--relation_consistency', type=str, default='False', help='regularize the consistency of feature relation')
parser.add_argument('--clip_grad', type=str, default='True', help='clip gradient of the network parameters')


parser.add_argument('--norm', type=str, default='False', help='Norm Layer between UNet and Transformer')
parser.add_argument('--input_normalize', type=str, default='mean_std', help='choose from [min_max, mean_std, divide]')

parser.add_argument("--dist_url", default="63654")

parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')
parser.add_argument('--base_num_every_group', type=int, default=2,
                    help='super resolution scale')


parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--augment', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftloss', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftd', action='store_true',
                    help='use data augmentation')
parser.add_argument('--fftd_weight', type=float, default=0.1,
                    help='use data augmentation')
parser.add_argument('--fft_weight', type=float, default=0.01)

# Model specifications
parser.add_argument('--model', type=str, default='MYNET')
parser.add_argument('--act', type=str, default='PReLU')
parser.add_argument('--data_range', type=float, default=1)
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--num_features', type=int, default=64)

parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')

parser.add_argument('--MASKTYPE', type=str, default='random') # "random" or "equispaced"
parser.add_argument('--CENTER_FRACTIONS', nargs='+', type=float)
parser.add_argument('--ACCELERATIONS', nargs='+', type=int)



args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))


# for arg in vars(args):
#     if vars(args)[arg] == 'True':
#         vars(args)[arg] = True
#     elif vars(args)[arg] == 'False':
#         vars(args)[arg] = False

