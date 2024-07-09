"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Xiaohan Xing, 2023/05/23
对于Unet中的high-level feature, 用Transformer进行特征交互，然后和Transformer之前的特征一起送到decoder重建图像。
"""
# coding: utf-8
from typing import Any
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


#######################################################
########## Transformer for feature enhancement ########
#######################################################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def get_feature(self, x):
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if idx == 0:
                return x
        #  return x
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



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

        return output
    



class Unet_Transformer(nn.Module):
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

        self.encoder = Encoder(self.num_pool_layers, self.in_chans, self.chans, self.drop_prob)
        ch = self.encoder.ch
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.decoder = Decoder(self.num_pool_layers, ch, self.out_chans, self.drop_prob)


        # transformer fusion modules
        fmp_size = 15 # feature map after encoder [bs,encoder.output_dim,fmp_size,fmp_size]=[4,256,96,96]
        num_patch = fmp_size * fmp_size
        patch_dim = ch

        self.to_patch_embedding = Rearrange('b e (h) (w) -> b (h w) e')
 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, patch_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(patch_dim, depth=2, heads=8, dim_head=64, mlp_dim=3072, dropout=0.1)



    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        output = image
        # print("image shape:", image.shape)

        ### CNN encoder提取high-level feature
        stack, output = self.encoder(output) ### output size: [4, 256, 15, 15]

        ### 用transformer进行feature enhancement, 然后和前面提取的特征concat.
        m_embed = self.to_patch_embedding(output) # [4, 225, 256], 225 is the number of patches.
        patch_embed_input = F.normalize(m_embed, p=2.0, dim=-1, eps=1e-12)
        b, n, _ = m_embed.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        patch_embed_input = torch.cat((cls_tokens, patch_embed_input), 1) 
        # print("patch embedding:", patch_embed_input.shape, "pos embedding:", self.pos_embedding.shape)
        patch_embed_input += self.pos_embedding
        patch_embed_input = self.dropout(patch_embed_input)  # [4, 1152+1, 1024]

        feature_output = self.transformer(patch_embed_input)[:, 1:, :]  # [4, 225, 256]
        # print("output of the transformer:", feature_output.shape)
        h, w = int(np.sqrt(feature_output.shape[1])), int(np.sqrt(feature_output.shape[1]))
        feature_output = feature_output.contiguous().view(b, feature_output.shape[-1], h, w) # [4,512*2,24,24]

        output = torch.cat((output, feature_output), 1)
        # print("feature:", output.shape)

        ### 用CNN和transformer联合的特征送到decoder进行图像重建
        output = self.decoder(output, stack)

        return output



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



def build_model(args):
    return Unet_Transformer(args)
