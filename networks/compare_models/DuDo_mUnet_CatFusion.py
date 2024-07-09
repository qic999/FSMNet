"""
Xiaohan Xing, 2023/11/10
在image domain network的前面加上kspace interpolation. 
kspace用Unet做interpolation.
image domain用mUnet + multi layer concat fusion模型。
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .ARTUnet import TransformerBlock

from .Kspace_mUnet import kspace_mmUnet
from .DataConsistency import DataConsistency
from dataloaders.BRATS_kspace_dataloader import ifft2c


################ Feature Extractor ##############

class Encoder(nn.Module):
    def __init__(self, num_pool_layers, in_chans, chans, drop_prob):
        super(Encoder, self).__init__()
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.ch = ch

    def forward(self, x):
        stack = []
        output = x
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        return stack, output




class Decoder(nn.Module):
    def __init__(self, num_pool_layers, ch, out_chans, drop_prob):
        super(Decoder, self).__init__()
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, out_chans, kernel_size=1, stride=1),
                # nn.Tanh(),
            )
        )

    def forward(self, x, stack):
        output = x
        # print("decoder layer output:", output.shape)
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            # print("output after transpose conv:", output.shape)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
    



class mUnet_CATfusion(nn.Module):
    """
    整体框架是multi-modal Unet. 两个模态分别提取各层特征，然后通过ART block融合各层特征。
    """

    def __init__(
        self, args, 
        input_dim: int = 1,
        output_dim: int = 1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
    # def __init__(self, args):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = input_dim
        self.out_chans = output_dim
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.kspace_net = kspace_mmUnet(args)

        self.encoder1 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        self.encoder2 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder1.ch

        # print("encoder layers:", self.encoder1.down_sample_layers)

        self.fuse_conv_layers1 = nn.ModuleList()
        self.fuse_conv_layers2 = nn.ModuleList()
        for l in range(self.num_pool_layers):
            # print("input and output channels:", chans*(2**(l+1)), chans*(2**l))
            self.fuse_conv_layers1.append(ConvBlock(chans*(2**(l+1)), chans*(2**l), drop_prob))
            self.fuse_conv_layers2.append(ConvBlock(chans*(2**(l+1)), chans*(2**l), drop_prob))

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)


    def complex_abs(self, data):
        """
        data: [B, 2, H, W]
        """
        if data.size(1) == 2:
            return (data ** 2).sum(dim=1).sqrt()
        elif data.size(-1) == 2:
            return (data ** 2).sum(dim=-1).sqrt()
    

    def normalize(self, data, mean, stddev, eps=0.0):
        """
        Normalize the given tensor.
        Applies the formula (data - mean) / (stddev + eps).
        """
        return (data - mean) / (stddev + eps)
    

    def forward(self, args, kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, \
                aux_image: torch.Tensor, t2_gt: torch.Tensor, t1_max: torch.Tensor, t2_max: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if args.domain == "dual":
            recon_kspace, masked_kspace = self.kspace_net(kspace, ref_kspace, mask)
            # print("input kspace range:", kspace.max(), kspace.min())
            # print("recon kspace range:", recon_kspace.max(), recon_kspace.min())
            image = self.complex_abs(ifft2c(recon_kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()

        elif args.domain == "image":
            image = self.complex_abs(ifft2c(kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()
            recon_kspace, masked_kspace = None, None

        image = image * t2_max.to(torch.float32).view(-1, 1, 1, 1)
        aux_image = aux_image * t1_max.to(torch.float32).view(-1, 1, 1, 1)
        t2_image = t2_gt * t2_max.to(torch.float32).view(-1, 1, 1, 1)

        LR_image = self.complex_abs(ifft2c(kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()
        LR_image = LR_image * t2_max.to(torch.float32).view(-1, 1, 1, 1)

        # # print("aux_image range:", aux_image.max(), aux_image.min())
        # print("LR_image range:", LR_image.max(), LR_image.min())
        # print("kspace recon_img range:", image.max(), image.min())

        ### normalize the input data with (x-mean)/std, and save the mean and std for recovery.
        t1_mean = aux_image.view(aux_image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        t1_std = aux_image.view(aux_image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()
        t2_mean = image.view(image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        t2_std = image.view(image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()

        # t2_LR_mean = LR_image.view(LR_image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        # t2_LR_std = LR_image.view(LR_image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()

        # print("t2_kspace_recon mean and std:", t2_mean.view(-1), t2_std.view(-1))
        # print("t2_LR mean and std:", t2_LR_mean.view(-1), t2_LR_std.view(-1))

        aux_image = self.normalize(aux_image, t1_mean, t1_std, eps=1e-11)
        image = self.normalize(image, t2_mean, t2_std, eps=1e-11)
        LR_image = self.normalize(LR_image, t2_mean, t2_std, eps=1e-11)
        t2_gt_image = self.normalize(t2_image, t2_mean, t2_std, eps=1e-11)

        aux_image = torch.clamp(aux_image, -6, 6)
        image = torch.clamp(image, -6, 6)
        LR_image = torch.clamp(LR_image, -6, 6)
        t2_gt_image = torch.clamp(t2_gt_image, -6, 6)


        # print("normalized t1 range:", aux_image.max(), aux_image.min())
        # print("normalized kspace_recon_t2 range:", image.max(), image.min())
        # print("normalized LR_t2 range:", LR_image.max(), LR_image.min())

        data_stats = {"t1_mean": t1_mean, "t1_std": t1_std, "t2_mean": t2_mean, "t2_std": t2_std}

        # output1, output2 = aux_image.detach(), torch.cat((image, LR_image), 1).detach()
        # output1, output2 = aux_image.detach(), LR_image.detach()
        output1, output2 = aux_image.detach(), image.detach()
        stack1, stack2 = [], []
        t1_features, t2_features = [], []

        l = 0
        bs, _, H, W = aux_image.shape

        for net1_layer, net2_layer, net1_fuse_layer, net2_fuse_layer in zip(
            self.encoder1.down_sample_layers, self.encoder2.down_sample_layers, self.fuse_conv_layers1, self.fuse_conv_layers2):
            ### 将encoder中multi-modal fusion之前的特征通过skip connection连接到decoder.
            output1 = net1_layer(output1)
            stack1.append(output1)
            t1_features.append(output1)
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2, padding=0)

            output2 = net2_layer(output2)
            stack2.append(output2)
            t2_features.append(output2)
            output2 = F.avg_pool2d(output2, kernel_size=2, stride=2, padding=0)

            ### 两个模态的特征concat
            output1 = net1_fuse_layer(torch.cat((output1, output2), 1))
            output2 = net2_fuse_layer(torch.cat((output1, output2), 1))    


        # output = torch.cat((output1, output2), 1)
        output1 = self.conv1(output1)
        output2 = self.conv2(output2)
        recon_t1 = self.decoder1(output1, stack1)
        recon_t2 = self.decoder2(output2, stack2)

        return recon_t1, recon_t2, aux_image, t2_gt_image, image, LR_image, recon_kspace, masked_kspace, data_stats
    


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)



def build_model(args):
    return mUnet_CATfusion(args)