"""
Xiaohan Xing, 2023/07/26
两个模态分别用Unet提取多个层级的特征, 每个层级都用ART block融合多模态特征.
将fused feature送到encoder的下一个层级和decoder.
"""
# coding: utf-8
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .ARTUnet_new import ConcatTransformerBlock, TransformerBlock


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




class mUnet_ARTfusion_SeqConcat(nn.Module):
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
        num_blocks=[2, 4, 4, 6],
        heads=[1, 2, 4, 8],
        window_size=[10, 10, 10, 10], 
        interval=[24, 12, 6, 3],
        mlp_ratio=4.
    ):

        super().__init__()

        self.in_chans = input_dim
        self.out_chans = output_dim
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        
        self.vis_feature_maps = False
        self.vis_attention_maps = False
        # self.vis_feature_maps = args.MODEL.VIS_FEAT_MAPS
        # self.vis_attention_maps = args.MODEL.VIS_ATTN_MAPS
        
        self.encoder1 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        self.encoder2 = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder1.ch

        self.fuse_layers = nn.ModuleList()
        for l in range(self.num_pool_layers):
            if l < 2:
                self.fuse_layers.append(nn.ModuleList([TransformerBlock(dim=int(chans * (2 ** (l + 1))), num_heads=heads[l],
                                        window_size=window_size[l], interval=interval[l], ds_flag=0 if i % 2 == 0 else 1,
                                        mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0.,
                                        attn_drop=0., drop_path=0.,visualize_attention_maps=self.vis_attention_maps) for i in range(num_blocks[l])]))
            else:
                self.fuse_layers.append(nn.ModuleList([ConcatTransformerBlock(dim=int(chans * (2 ** l)), num_heads=heads[l],
                                        window_size=window_size[l], interval=interval[l], ds_flag=0 if i % 2 == 0 else 1,
                                        mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0.,
                                        attn_drop=0., drop_path=0.,visualize_attention_maps=self.vis_attention_maps) for i in range(num_blocks[l])]))

        # print(self.fuse_layers)

        self.conv1 = ConvBlock(ch, ch * 2, drop_prob)
        self.conv2 = ConvBlock(ch, ch * 2, drop_prob)

        self.decoder1 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)
        self.decoder2 = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        output1, output2 = image1, image2
        stack1, stack2 = [], []
        t1_features, t2_features = [], []
        relation_stack1, relation_stack2 = [], []

        l = 0
        bs, _, H, W = image1.shape

        feature_maps = {
            'modal1': {'before': [], 'after': []},
            'modal2': {'before': [], 'after': []}
        }

        attention_maps = []
        """
        attention_maps = [ 
            unet layer 1 attention maps (x2): [map from TransBlock1, TransBlock2, ...],
            unet layer 2 attention maps (x4): [map from TransBlock1, TransBlock2, ...],
            unet layer 3 attention maps (x4): [map from TransBlock1, TransBlock2, ...],
            unet layer 4 attention maps (x6): [map from TransBlock1, TransBlock2, ...],
        ]
        """

        ### 两个模态分别用CNN layer提取特征，然后用transfuse layer融合，将融合前后的特征相加送到后续的layers.
        for net1_layer, net2_layer, fuse_layer in zip(self.encoder1.down_sample_layers, self.encoder2.down_sample_layers, self.fuse_layers):

            # print("number of blocks in fuse layer:", len(fuse_layer))

            ### 将encoder中multi-modal fusion之前的特征通过skip connection连接到decoder.
            output1 = net1_layer(output1)
            output2 = net2_layer(output2)

            if self.vis_feature_maps:
                feature_maps['modal1']['before'].append(output1.detach().cpu().numpy())
                feature_maps['modal2']['before'].append(output2.detach().cpu().numpy())

            ### 两个模态的特征concat
            # output = torch.cat((output1, output2), 1)



            unet_layer_attn_maps = []
            """
            unet_layer_attn_maps: [
                attn_map from TransformerBlock1: (B, nHGroups, nWGroups, winSize, winSize),
                attn_map from TransformerBlock2: (B, nHGroups, nWGroups, winSize, winSize),
                ...
            ]
            """
            if l < 2:
                output = torch.cat((output1, output2), 1)
                output = rearrange(output, "b c h w -> b (h w) c").contiguous()
            else:
                output1 = rearrange(output1, "b c h w -> b (h w) c").contiguous()
                output2 = rearrange(output2, "b c h w -> b (h w) c").contiguous()

            for layer in fuse_layer:
                if l < 2:
                    if self.vis_attention_maps:
                        output, attn_map = layer(output, [H // (2**l), W // (2**l)])
                        unet_layer_attn_maps.append(attn_map)
                    else:
                        output = layer(output, [H // (2**l), W // (2**l)])

                else:
                    if self.vis_attention_maps:
                        output1, output2, attn_map = layer(output1, output2, [H // (2**l), W // (2**l)])
                        unet_layer_attn_maps.append(attn_map)
                    else:
                        output1, output2 = layer(output1, output2, [H // (2**l), W // (2**l)])

            attention_maps.append(unet_layer_attn_maps)

            if l < 2:
                output = rearrange(output, "b (h w) c -> b c h w", h=H // (2**l), w=W // (2**l)).contiguous()
                output1 = output[:, :output.shape[1]//2, :, :]
                output2 = output[:, output.shape[1]//2:, :, :]
            else:
                output1 = rearrange(output1, "b (h w) c -> b c h w", h=H // (2**l), w=W // (2**l)).contiguous()
                output2 = rearrange(output2, "b (h w) c -> b c h w", h=H // (2**l), w=W // (2**l)).contiguous()

                # if self.vis_attention_maps:
                #     output1, output2, attn_map = layer(output1, output2, [H // (2**l), W // (2**l)]) # attn_map: (B, nHGroups, nWGroups, winSize, winSize)
                #     unet_layer_attn_maps.append(attn_map)
                # else:
                    # output1, output2 = layer(output1, output2, [H // (2**l), W // (2**l)])

            # attention_maps.append(unet_layer_attn_maps)



            if self.vis_feature_maps:
                feature_maps['modal1']['after'].append(output1.detach().cpu().numpy())
                feature_maps['modal2']['after'].append(output2.detach().cpu().numpy())

            stack1.append(output1)
            stack2.append(output2)

            ### downsampling features
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2, padding=0)
            output2 = F.avg_pool2d(output2, kernel_size=2, stride=2, padding=0)

            l += 1


        # output = torch.cat((output1, output2), 1)
        output1 = self.conv1(output1)
        output2 = self.conv2(output2)
        output1 = self.decoder1(output1, stack1)
        output2 = self.decoder2(output2, stack2)

        if self.vis_feature_maps:
            return output1, output2, feature_maps#, relation_stack1, relation_stack2 #, t1_features, t2_features
        elif self.vis_attention_maps:
            return output1, output2, attention_maps
        else:
            return output1, output2



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
    return mUnet_ARTfusion_SeqConcat(args)