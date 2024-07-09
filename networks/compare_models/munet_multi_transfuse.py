"""
Xiaohan Xing, 2023/06/12
两个模态分别用CNN提取多个层级的特征, 每个层级都用TransFuse layer融合。融合后的特征和encoder原始的特征相加。
2023/06/23
每个层级的特征用TransFuse layer融合之后, 将两个模态的original features和transfuse features concat, 
然后在每个模态中经过conv层变换得到下一层的输入。
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from .TransFuse import TransFuse_layer

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
    



class mUnet_TransFuse(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234 241.
    Springer, 2015.
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

        self.encoder1 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        self.encoder2 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder1.ch

        # print("encoder layers:", self.encoder1.down_sample_layers)

        ### conv layer to transform the features after Transfuse layer.
        self.fuse_conv_layers1 = nn.ModuleList()
        self.fuse_conv_layers2 = nn.ModuleList()
        for l in range(self.num_pool_layers):
            # print("input and output channels:", chans*(2**(l+1)), chans*(2**l))
            self.fuse_conv_layers1.append(ConvBlock(chans*(2**(l+2)), chans*(2**l), drop_prob))
            self.fuse_conv_layers2.append(ConvBlock(chans*(2**(l+2)), chans*(2**l), drop_prob))

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)

        # transformer fusion modules
        self.n_anchors = 15
        self.avgpool = nn.AdaptiveAvgPool2d((self.n_anchors, self.n_anchors))
        self.transfuse_layer1 = TransFuse_layer(n_embd=self.chans, n_head=4, block_exp=4, n_layer=8, num_anchors=self.n_anchors)
        self.transfuse_layer2 = TransFuse_layer(n_embd=2*self.chans, n_head=4, block_exp=4, n_layer=8, num_anchors=self.n_anchors)
        self.transfuse_layer3 = TransFuse_layer(n_embd=4*self.chans, n_head=4, block_exp=4, n_layer=8, num_anchors=self.n_anchors)
        self.transfuse_layer4 = TransFuse_layer(n_embd=8*self.chans, n_head=4, block_exp=4, n_layer=8, num_anchors=self.n_anchors)

        self.transfuse_layers = nn.Sequential(self.transfuse_layer1, self.transfuse_layer2, self.transfuse_layer3, self.transfuse_layer4)

        # print("length of encoder layers:", len(self.encoder1.down_sample_layers), len(self.encoder1.down_sample_layers), len(self.transfuse_layers))



    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:

        output1, output2 = image1, image2
        stack1, stack2 = [], []

        ### 两个模态分别用CNN layer提取特征，然后用transfuse layer融合，将融合前后的特征相加送到后续的layers.
        for net1_layer, net2_layer, transfuse_layer, net1_fuse_layer, net2_fuse_layer in zip(self.encoder1.down_sample_layers, \
                self.encoder2.down_sample_layers, self.transfuse_layers, self.fuse_conv_layers1, self.fuse_conv_layers2):
            
            ### extract multi-level features from the encoder of each modality.
            output1 = net1_layer(output1)
            output2 = net2_layer(output2)

            ### Transformer-based multi-modal fusion layer
            feature1 = self.avgpool(output1)
            feature2 = self.avgpool(output2)
            feature1, feature2 = transfuse_layer(feature1, feature2)
            feature1 = F.interpolate(feature1, scale_factor=output1.shape[-1]//self.n_anchors, mode='bilinear')
            feature2 = F.interpolate(feature2, scale_factor=output2.shape[-1]//self.n_anchors, mode='bilinear')

            ### 两个模态的特征concat
            output1 = net1_fuse_layer(torch.cat((output1, output2, feature1, feature2), 1))
            output2 = net2_fuse_layer(torch.cat((output1, output2, feature1, feature2), 1))    
            
            stack1.append(output1)            
            stack2.append(output2)

            ### downsampling features
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2, padding=0)
            output2 = F.avg_pool2d(output2, kernel_size=2, stride=2, padding=0)


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
    return mUnet_TransFuse(args)
