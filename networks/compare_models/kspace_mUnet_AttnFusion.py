"""
Xiaohan Xing, 2023/11/22
两个模态分别用CNN提取多个层级的特征, 每个层级都把特征沿着宽度拼接起来，得到(h, 2w, c)的特征。
然后学习得到(h, 2w)的attention map, 和原始的特征相乘送到后面的层。
"""

# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


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
    



class mUnet_multi_fuse(nn.Module):
    """
    整体框架是multi-modal Unet. 两个模态分别提取各层特征，然后通过求平均的方式融合各层特征。
    """

    def __init__(
        self, args, 
        input_dim: int = 2,
        output_dim: int = 2,
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

        if args.HF_refine == "True":
            self.in_chans = self.in_chans * 2

        self.encoder1 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        self.encoder2 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder1.ch

        # print("encoder layers:", self.encoder1.down_sample_layers)

        self.attn_layers = nn.ModuleList()

        for l in range(self.num_pool_layers):

            self.attn_layers.append(nn.Sequential(ConvBlock(chans*(2**(l+1)), chans*(2**l), drop_prob),
                                                  nn.Conv2d(chans*(2**l), 2, kernel_size=1, stride=1),
                                                  nn.Sigmoid()))

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)


    def forward(self, kspace, ref_kspace, recon_kspace, recon_ref_kspace) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # output = self.encoder1.(output)
        # output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        # print("kspace and ref_kspace:", kspace.shape, ref_kspace.shape)
        # if recon_kspace is not None:
        #     print("recon_kspace and recon_ref_kspace:", recon_kspace.shape, recon_ref_kspace.shape)


        # print("T2 input_kspace range:", kspace.max(), kspace.min())
        # print("T2 recon_kspace range:", recon_kspace.max(), recon_kspace.min())
        # print("T1 input_kspace range:", ref_kspace.max(), ref_kspace.min())
        # print("T1 recon_kspace range:", recon_ref_kspace.max(), recon_ref_kspace.min())

        if recon_kspace is not None:
            output1 = torch.cat((kspace, recon_kspace), 1)
            output2 = torch.cat((ref_kspace, recon_ref_kspace), 1)
        else:
            output1, output2 = kspace, ref_kspace
            
        stack1, stack2 = [], []
        t1_features, t2_features = [], []
        relation_stack1, relation_stack2 = [], []

        ### 两个模态分别用CNN layer提取特征，然后用transfuse layer融合，将融合前后的特征相加送到后续的layers.
        for net1_layer, net2_layer, attn_layer in zip(
            self.encoder1.down_sample_layers, self.encoder2.down_sample_layers, self.attn_layers):
            ### 将encoder中multi-modal fusion之前的特征通过skip connection连接到decoder.
            output1 = net1_layer(output1)
            stack1.append(output1)
            t1_features.append(output1)
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2, padding=0)

            output2 = net2_layer(output2)
            stack2.append(output2)
            t2_features.append(output2)
            output2 = F.avg_pool2d(output2, kernel_size=2, stride=2, padding=0)

            ### 两个模态的特征concat之后求attention, 然后进行modulation.
            attn = attn_layer(torch.cat((output1, output2), 1))  
            # print("spatial attention:", attn[:, 0, :, :].unsqueeze(1).shape)
            # print("output1:", output1.shape)
            output1 = output1 * (1 + attn[:, 0, :, :].unsqueeze(1))
            output2 = output2 * (1 + attn[:, 1, :, :].unsqueeze(1))

            # print("range of output1:", output1.max(), output1.min())
            # print("range of output2:", output2.max(), output2.min())

        # output = torch.cat((output1, output2), 1)
        output1 = self.conv1(output1)
        output2 = self.conv2(output2)
        output1 = self.decoder1(output1, stack1)
        output2 = self.decoder2(output2, stack2)

        return output1, output2 #, relation_stack1, relation_stack2 #, t1_features, t2_features



def get_relation_matrix(feature):
    """
    将各层的特征都变换成5*5的尺寸, 然后计算25*25个位置之间的relation matrix.
    """
    bs, c, h, w = feature.shape
    feature = feature.view(bs, c, h//15, 15, w//15, 15).permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, -1, 15, 15)
    avg_pool = nn.AdaptiveAvgPool2d(5)
    feature = avg_pool(feature).view(bs, feature.shape[1], -1).permute(0, 2, 1) ### (bs, 5*5, c)
    # print("intermediate feature:", feature.shape)

    feature_norm = torch.norm(feature, p=2, dim=-1, keepdim=True) ### (bs, 5*5, 1)
    relation_matrix = torch.bmm(feature, feature.permute(0, 2, 1))/torch.bmm(feature_norm, feature_norm.permute(0, 2, 1)) # (bs, 25, 25)

    return relation_matrix
    




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
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
    return mUnet_multi_fuse(args)
