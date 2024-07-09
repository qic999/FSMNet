"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Xiaohan Xing, 2023/05/24
concat the input images and utilize CNN to extract multi-modal features, reconstruct multi-modal images with conditional decoder.
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from .modules import *



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
                nn.Tanh(),
            )
        )

    def forward(self, x, stack):
        output = x
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

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
            # print("decoder output:", output.shape)

        return output
    



class MyNet_fusion(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234 241.
    Springer, 2015.
    """

    def __init__(
        self, 
        input_dim: int = 2,
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

        self.encoder = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder.ch
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        # self.decoder = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)

        # nlatent = 16
        # self.decoder = Decoder_CIN(n_upsample=4, n_res=1, dim=ch*2, output_dim=1, nlatent=nlatent, pad_type='zero')
        # self.G_D = nn.Linear(1, nlatent, bias=True)

        self.decoder1 = Decoder(n_upsample=4, n_res=1, dim=ch*2, output_dim=1, pad_type='zero')
        self.decoder2 = Decoder(n_upsample=4, n_res=1, dim=ch*2, output_dim=1, pad_type='zero')


    def forward(self, image, aux_image, domainness=[]):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        output = torch.cat((image, aux_image), 1)  ### shape: (N, in_chans, H, W)`.
        # print("image shape:", image.shape)

        stack, output = self.encoder(output)
        feature = self.conv(output) ### from [4, 256, 15, 15] to [4, 512, 15, 15]
    
        # output = []
        # for item in domainness:
        #     Z = torch.unsqueeze(torch.unsqueeze(self.G_D(item), 2), 3)
        #     output += [self.decoder(feature, Z)]

        output = []
        output += [self.decoder1(feature)]
        output += [self.decoder2(feature)]
        

        # print("output shape:", output[0].shape)
        return output



class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, pad_type='zero'):
        super(Decoder, self).__init__()

        norm_layer = nn.InstanceNorm2d
        use_dropout = False

        self.n_res = n_res

        self.model = []
        for i in range(n_res):
            self.model += [ResnetBlock(dim=dim, padding_type=pad_type, norm_layer=norm_layer, 
                                       use_dropout=use_dropout, use_bias=True)]

        for i in range(n_upsample):
            self.model += [
                nn.ConvTranspose2d(dim, dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                norm_layer(dim//2),
                nn.ReLU(True)]
            dim //= 2

        self.model += [nn.ReflectionPad2d(3), nn.Conv2d(dim, output_dim, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        return self.model(input)



class Decoder_CIN(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, nlatent=16, pad_type='zero'):
        super(Decoder_CIN, self).__init__()

        norm_layer = CondInstanceNorm
        use_dropout = False

        self.n_res = n_res

        self.model = []
        for i in range(n_res):
            self.model += [CINResnetBlock(x_dim=dim, z_dim=nlatent, padding_type=pad_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        for i in range(n_upsample):
            self.model += [
                nn.ConvTranspose2d(dim, dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                norm_layer(dim//2, nlatent),
                nn.ReLU(True)]
            dim //= 2

        self.model += [nn.ReflectionPad2d(3), nn.Conv2d(dim, output_dim, kernel_size=7, padding=0), nn.Tanh()]
        self.model = TwoInputSequential(*self.model)

    def forward(self, input, noise):
        return self.model(input, noise)




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
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
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


