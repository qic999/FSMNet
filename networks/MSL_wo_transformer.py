"""
2023/05/23
之前lequan's method中transformer可能会造成重建效果差。
现在把transformer去掉，改成用CNN encoder分别提取两个模态的特征之后，求平均得到multi-modal representations, 然后用conditional decoder进行图像重建。
"""


import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .modules import *
import numpy as np

## sum two modality features
class MyNet_fusion(nn.Module):
    def __init__(self, nlatent=16, gd_bias=True):
        super(MyNet_fusion, self).__init__()
        self.n_res = 3

        self.encoder1 = ContentEncoder_expand(n_downsample=4, n_res=self.n_res, input_dim=1, dim=64, norm='in', activ='relu',
                                              pad_type='reflect')
        self.encoder2 = ContentEncoder_expand(n_downsample=4, n_res=self.n_res, input_dim=1, dim=64, norm='in', activ='relu',
                                              pad_type='reflect')

        self.decoder = Decoder_CIN(n_upsample=4, n_res=1, dim=self.encoder1.output_dim, output_dim=1, nlatent=nlatent, pad_type='zero')

        self.G_D = nn.Linear(1, nlatent, bias=gd_bias)


    def forward(self, m, n, domainness=[]):
        #4,1,384,384
        # print("m:", m.shape)
        m_out = self.encoder1.model(m) # [4, 256, 96, 96]   
        n_out = self.encoder2.model(n)
        # print("m_out:", m_out.shape)
        fusion_feat = (m_out + n_out)/2.0

        output = []
        for item in domainness:
            Z = torch.unsqueeze(torch.unsqueeze(self.G_D(item), 2), 3)
            output += [self.decoder(fusion_feat, Z)]

        # print("output shape:", output[0].shape)
        return output


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

##################################################################################
# Encoder and Decoders
##################################################################################
class ContentEncoder_expand(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder_expand, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model = nn.Sequential(*self.model)

        self.resblocks =[]
        for i in range(n_res):
            self.resblocks += [ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)]

        self.model2 = nn.Sequential(*self.resblocks)
        self.output_dim = dim

        # print("content_encoder model_1:", self.model)
        # print("content_encoder model_2:", self.model2)

    def forward(self, x):
        out = self.model(x)
        out = self.model2(out)
        return out

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
    
    def get_feature(self, input, noise):
        fuse = input
        for i in range(self.n_res):
            fuse = self.model[i](fuse, noise)
        
        for i in range(6):
            if isinstance(self.model[self.n_res+i], TwoInputModule):
                fuse = self.model[self.n_res+i].forward(fuse, noise)
            else:
                fuse = self.model[self.n_res+i].forward(fuse)
            if i == 5:
                return fuse


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
