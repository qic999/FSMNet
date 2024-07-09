"""
Xiaohan Xing, 2023/06/25
两个模态分别用CNN提取多个层级的特征, 每个层级的特征都经过concat之后得到fused feature, 
然后每个模态都通过一层conv变换得到2C*h*w的特征, 其中C个channels作为common feature, 另C个channels作为specific features. 
利用CDDFuse论文中的decomposition loss约束特征解耦。
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

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
    



class mUnet_multi_fuse(nn.Module):
    """
    整体框架是multi-modal Unet. 两个模态分别提取各层特征，然后通过求平均的方式融合各层特征。
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

        self.transform_layers1 = nn.ModuleList()
        self.transform_layers2 = nn.ModuleList()
        self.fuse_conv_layers1 = nn.ModuleList()
        self.fuse_conv_layers2 = nn.ModuleList()
        for l in range(self.num_pool_layers):
            # print("input and output channels:", chans*(2**(l+1)), chans*(2**l))
            self.transform_layers1.append(ConvBlock(chans*(2**l), chans*(2**l)*2, drop_prob))
            self.transform_layers2.append(ConvBlock(chans*(2**l), chans*(2**l)*2, drop_prob))
            self.fuse_conv_layers1.append(ConvBlock(chans*(2**l)*3, chans*(2**l), drop_prob))
            self.fuse_conv_layers2.append(ConvBlock(chans*(2**l)*3, chans*(2**l), drop_prob))

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)


    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:

        output1, output2 = image1, image2
        stack1, stack2 = [], []
        stack1_common, stack1_specific, stack2_common, stack2_specific = [], [], [], []
        t1_features, t2_features = [], []
        relation_stack1, relation_stack2 = [], []

        ### 两个模态分别用CNN layer提取特征，然后用transfuse layer融合，将融合前后的特征相加送到后续的layers.
        for net1_layer, net2_layer, net1_transform_layer, net2_transform_layer, net1_fuse_layer, net2_fuse_layer in zip(
            self.encoder1.down_sample_layers, self.encoder2.down_sample_layers, self.transform_layers1, self.transform_layers2, \
            self.fuse_conv_layers1, self.fuse_conv_layers2): 

            ### 将encoder中multi-modal fusion之前的特征通过skip connection连接到decoder.
            output1 = net1_layer(output1)
            output2 = net2_layer(output2)
            # print("original feature:", output1.shape)

            ### 分别提取两个模态的common feature和specific feature
            output1 = net1_transform_layer(output1)
            # print("transformed feature:", output1.shape)
            output1_common = output1[:, :(output1.shape[1]//2), :, :]
            output1_specific = output1[:, (output1.shape[1]//2):, :, :]
            output2 = net2_transform_layer(output2)
            output2_common = output2[:, :(output2.shape[1]//2), :, :]
            output2_specific = output2[:, (output2.shape[1]//2):, :, :]
            
            stack1_common.append(output1_common)
            stack1_specific.append(output1_specific)
            stack2_common.append(output2_common)
            stack2_specific.append(output2_specific)

            ### 将一个模态的两组feature和另一模态的common feature合并， 然后经过conv变换得到和初始特征相同维度的fused feature送到下一层。
            output1 = net1_fuse_layer(torch.cat((output1_common, output1_specific, output2_common), 1))
            output2 = net2_fuse_layer(torch.cat((output2_common, output2_specific, output1_common), 1))  
            # print("fused feature:", output1.shape)  
            
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

        return output1, output2, stack1_common, stack1_specific, stack2_common, stack2_specific
        #, relation_stack1, relation_stack2 #, t1_features, t2_features



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
    return mUnet_multi_fuse(args)
