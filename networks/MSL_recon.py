"""
2023/06/01, Xiaohan Xing,
在MSL代码的基础上，保留encoder部分，把conditional decoder改成普通的decoder，看效果如何。
"""

import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .modules import *
import numpy as np

## sum two modality features
class MyNet_fusion(nn.Module):
    def __init__(self):
        super(MyNet_fusion, self).__init__()
        self.n_res = 3

        self.encoder1 = Encoder(num_pool_layers=4, in_chans=1, chans=32, drop_prob=0.0)
        self.encoder2 = Encoder(num_pool_layers=4, in_chans=1, chans=32, drop_prob=0.0)
        ch = self.encoder1.ch
        self.decoder = Decoder(n_upsample=4, n_res=self.n_res, dim=ch, output_dim=1)

        # transformer fusion modules
        fmp_size = 15 # feature map after encoder [bs,encoder.output_dim,fmp_size,fmp_size]=[4,256,96,96]
        num_patch = fmp_size * fmp_size
        patch_dim = 256

        self.to_patch_embedding = Rearrange('b e (h) (w) -> b (h w) e')
 
        self.pos_embedding = nn.Parameter(torch.randn(1, 4*num_patch + 1, patch_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(patch_dim, depth=2, heads=8, dim_head=64, mlp_dim=3072, dropout=0.1)


    def forward(self, m, n):
        #4,1,384,384
        # print("m:", m.shape)
        m_out = self.encoder1(m) # [4, 256, 96, 96]   
        n_out = self.encoder2(n)
        # print("m_out:", m_out.shape)

        m_embed = self.to_patch_embedding(m_out) # [4, 225, 256], 225 is the number of patches.
        n_embed = self.to_patch_embedding(n_out)

        m_embed = F.normalize(m_embed, p=2.0, dim=-1, eps=1e-12)
        n_embed = F.normalize(n_embed, p=2.0, dim=-1, eps=1e-12)
        
        patch_embed_ct = torch.cat((m_embed, m_embed),1) # [4, 450, 256]
        patch_embed_mri = torch.cat((n_embed, n_embed),1) 

        patch_embed_input = torch.cat((patch_embed_ct, patch_embed_mri),1) 

        b, n, _ = m_embed.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        patch_embed_input = torch.cat((cls_tokens, patch_embed_input), 1) 
        # print("patch embedding:", patch_embed_input.shape, "pos embedding:", self.pos_embedding.shape)
        patch_embed_input += self.pos_embedding
        patch_embed_input = self.dropout(patch_embed_input)  # [4, 1152+1, 1024]

        feature_output = self.transformer(patch_embed_input)[:, 1:, :]

        patch_num = feature_output.shape[1]

        feat_ct_ct = feature_output[:,:patch_num//4,:]
        feat_ct_mri = feature_output[:,patch_num//4:patch_num//2,:]
        feat_mri_ct = feature_output[:,patch_num//2:3*patch_num//4,:]
        feat_mri_mri = feature_output[:,3*patch_num//4:,:]

        fusion_feat = (feat_ct_ct + feat_mri_mri + feat_ct_mri + feat_mri_ct).transpose(1,2)
        fusion_feat /= 4

        h, w = int(np.sqrt(fusion_feat.shape[-1])), int(np.sqrt(fusion_feat.shape[-1]))
        fusion_feat = fusion_feat.contiguous().view(b, fusion_feat.shape[1], h, w) # [4,512*2,24,24]

        # fusion_feat = self.upsampling1(fusion_feat)
        # fusion_feat = self.upsampling2(fusion_feat)
        
        rnd = torch.rand(1)[0]
        if rnd < 0.5:
            fusion_feat_com = (feat_ct_mri + feat_ct_ct).transpose(1,2)
            fusion_feat_com /= 2
        else:
            fusion_feat_com = (feat_mri_mri + feat_mri_ct).transpose(1,2)
            fusion_feat_com /= 2

        fusion_feat_com = fusion_feat_com.contiguous().view(b, fusion_feat_com.shape[1], h, w) # [4,512*2,24,24]

        output = self.decoder(fusion_feat)
        output_com = self.decoder(fusion_feat_com)

        return output, output_com, (feat_ct_ct, feat_mri_ct), (feat_ct_mri, feat_mri_mri)



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

            # print(output.shape)
        # return stack, output
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
