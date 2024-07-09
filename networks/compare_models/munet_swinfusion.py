"""
Xiaohan Xing, 2023/06/12
两个模态分别用CNN提取多个层级的特征, 每个层级都用TransFuse layer融合。融合后的特征和encoder原始的特征相加。
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from .SwinFuse_layer import SwinFusion_layer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
    



class mUnet_SwinFuse(nn.Module):
    """
    整体框架是multi-modal Unet. 两个模态分别提取各层特征，然后通过SwinFusion的方式融合。
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
        self.fuse_type = 'swinfuse'
        self.img_size = 240
        # self.fuse_type = 'sum'

        self.encoder1 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        self.encoder2 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder1.ch

        # print("encoder layers:", self.encoder1.down_sample_layers)

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)

        # Swin transformer fusion modules

        self.fuse_layer1 = SwinFusion_layer(img_size=(self.img_size//2, self.img_size//2), patch_size=4, window_size=5, Fusion_depths=[2, 2, 2, 2], \
                                            embed_dim=self.chans, Fusion_num_heads=[4, 4, 4, 4])
        self.fuse_layer2 = SwinFusion_layer(img_size=(self.img_size//4, self.img_size//4), window_size=5, Fusion_depths=[2, 2, 2, 2], \
                                            embed_dim=2*self.chans, Fusion_num_heads=[4, 4, 4, 4])
        self.fuse_layer3 = SwinFusion_layer(img_size=(self.img_size//8, self.img_size//8), window_size=5, Fusion_depths=[2, 2, 2, 2], \
                                            embed_dim=4*self.chans, Fusion_num_heads=[4, 4, 4, 4])
        self.fuse_layer4 = SwinFusion_layer(img_size=(self.img_size//16, self.img_size//16), window_size=5, Fusion_depths=[2, 2, 2, 2], \
                                            embed_dim=8*self.chans, Fusion_num_heads=[4, 4, 4, 4])
        self.swinfusion_layers = nn.Sequential(self.fuse_layer1, self.fuse_layer2, self.fuse_layer3, self.fuse_layer4)

        # print("length of encoder layers:", len(self.encoder1.down_sample_layers), len(self.encoder1.down_sample_layers), len(self.transfuse_layers))



    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # output = self.encoder1.(output)
        # output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output1, output2 = image1, image2
        stack1, stack2 = [], []

        ### 两个模态分别用CNN layer提取特征，然后用transfuse layer融合，将融合前后的特征相加送到后续的layers.
        for net1_layer, net2_layer, swinfuse_layer in zip(self.encoder1.down_sample_layers, self.encoder2.down_sample_layers, self.swinfusion_layers):
            output1 = net1_layer(output1)
            stack1.append(output1)
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2, padding=0)

            output2 = net2_layer(output2)
            stack2.append(output2)
            output2 = F.avg_pool2d(output2, kernel_size=2, stride=2, padding=0)

            # print("output of encoders:", output1.shape, output2.shape)
            # print("CNN feature range:", output1.max(), output2.min())

            ### Swin transformer-based multi-modal fusion.
            feature1, feature2 = swinfuse_layer(output1, output2)
            output1 = (output1 + feature1 + output2 + feature2)/4.0
            output2 = (output1 + feature1 + output2 + feature2)/4.0          

        # output = torch.cat((output1, output2), 1)
        output1 = self.conv1(output1)
        output2 = self.conv2(output2)
        output1 = self.decoder1(output1, stack1)
        output2 = self.decoder2(output2, stack2)

        return output1, output2



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
    return mUnet_SwinFuse(args)
