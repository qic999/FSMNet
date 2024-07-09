"""
在our method中去掉多模态，直接用CNN提取图像特征，然后切patch得到patch embeddings. 送到transformer网络中进行MRI重建。
"""

import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ..modules import *
import numpy as np

## sum two modality features
class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()
        self.n_res = 3

        self.encoder = ContentEncoder_expand(n_downsample=2, n_res=self.n_res, input_dim=1, dim=64, norm='in', activ='relu',
                                             pad_type='reflect')

        self.decoder = Decoder(n_upsample=2, n_res=1, dim=self.encoder.output_dim, output_dim=1, pad_type='zero')

        # transformer fusion modules
        fmp_size = 60 # feature map after encoder [bs,encoder.output_dim,fmp_size,fmp_size]=[4,256,96,96]
        patch_size = 4
        num_patch = (fmp_size // patch_size) * (fmp_size // patch_size)
        patch_dim = 512*2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(self.encoder.output_dim, patch_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, patch_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(patch_dim, depth=2, heads=8, dim_head=64, mlp_dim=3072, dropout=0.1)

        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(patch_dim, patch_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(patch_dim//2),
            nn.ReLU(True)
        )
        self.upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(patch_dim//2, patch_dim//4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(patch_dim//4),
            nn.ReLU(True)
        )

    def forward(self, m):
        #4,1,240,240
        m_out = self.encoder.model(m) # [4, 256, 60, 60]   

        m_embed = self.to_patch_embedding(m_out) # [4, 225, 512*2], 225 is the number of patches.
        patch_embed = F.normalize(m_embed, p=2.0, dim=-1, eps=1e-12)
        
        b, n, _ = patch_embed.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        patch_embed_input = torch.cat((cls_tokens, patch_embed),1) 
        # print("patch embedding:", patch_embed_input.shape, "pos embedding:", self.pos_embedding.shape)
        patch_embed_input += self.pos_embedding
        patch_embed_input = self.dropout(patch_embed_input)  # [4, 225+1, 1024]

        feature_output = self.transformer(patch_embed_input)[:, 1:, :].transpose(1,2) # [4, 1024, 225]

        h, w = int(np.sqrt(feature_output.shape[-1])), int(np.sqrt(feature_output.shape[-1]))
        feature_output = feature_output.contiguous().view(b, feature_output.shape[1], h, w) # [4,512*2,15,15]

        feature_output = self.upsampling1(feature_output) # [4,512,30,30]
        feature_output = self.upsampling2(feature_output) # [4,256,60,60]

        output = self.decoder(feature_output)

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
    

def build_model(args):
    return CNN_Transformer()
