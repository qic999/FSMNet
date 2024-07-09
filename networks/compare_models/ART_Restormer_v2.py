"""
2023/07/24, Xiaohan Xing
U-shape structure, 每个block都是由ART和Restormer并联而成。
输入特征分别经过ART和Restormer两个分支提取特征, 将两者的特征concat之后，用conv层变换得到该block的输出特征。
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers

from einops import rearrange
import math

from .restormer import TransformerBlock as RestormerBlock

NEG_INF = -1000000


##########################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


#########################################
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        # print("input of the Attention block:", x.max(), x.min())
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("output of the Attention block:", x.max(), x.min())
        return x


##########################################################################
class ARTBlock(nn.Module):
    r""" ART Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size: window size of dense attention
        interval: interval size of sparse attention
        ds_flag (int): use Dense Attention or Sparse Attention, 0 for DAB and 1 for SAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        # act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 interval=8,
                 ds_flag=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.interval = interval
        self.ds_flag = ds_flag
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.ds_flag = 0
            self.window_size = min(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        size_par = self.interval if self.ds_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # partition the whole feature map into several groups
        if self.ds_flag == 0:  # Dense Attention
            G = Gh = Gw = self.window_size
            x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
            nP = Hd * Wd // G ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nP, 1, G * G)
                attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        if self.ds_flag == 1:  # Sparse Attention
            I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
            x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B * I * I, Gh * Gw, C)
            nP = I ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nP, 1, Gh * Gw)
                attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # MSA
        # print("attn mask:", attn_mask)
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nP*B, Gh*Gw, C
        # print("output of the Attention block:", x.max(), x.min())

        # merge embeddings
        if self.ds_flag == 0:
            x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hd, Wd, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x:", x.shape)

        return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, ds_flag={self.ds_flag}, mlp_ratio={self.mlp_ratio}"


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class ART_Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_art_blocks=[2, 4, 4, 6],
                 num_restormer_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 window_size=[8, 8, 8, 8], mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,
                 interval=[32, 16, 8, 4],
                 ffn_expansion_factor = 2.66,
                 LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                 bias=False
                 ):

        super(ART_Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.ART_encoder_level1 = nn.ModuleList([ARTBlock(dim=dim, num_heads=heads[0], window_size=window_size[0], interval=interval[0],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[0])])
        self.Restormer_encoder_level1 = nn.Sequential(*[RestormerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[0])])
        self.enc_fuse_level1 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)


        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.ART_encoder_level2 = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=window_size[1], interval=interval[1],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[1])])
        
        self.Restormer_encoder_level2 = nn.Sequential(*[RestormerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[1])])
        self.enc_fuse_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)


        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.ART_encoder_level3 = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], window_size=window_size[2], interval=interval[2],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[2])])

        self.Restormer_encoder_level3 = nn.Sequential(*[RestormerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[2])])
        self.enc_fuse_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=1, bias=bias)


        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.ART_latent = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], window_size=window_size[3], interval=interval[3],
                                    ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[3])])

        self.Restormer_latent = nn.Sequential(*[RestormerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, \
                                                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[3])])
        self.enc_fuse_level4 = nn.Conv2d(int(dim * 2 ** 4), int(dim * 2 ** 3), kernel_size=3, stride=1, padding=1, bias=bias)


        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.ART_decoder_level3 = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], window_size=window_size[2], interval=interval[2],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[2])])

        self.Restormer_decoder_level3 = nn.Sequential(*[RestormerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[2])])
        self.dec_fuse_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=3, stride=1, padding=1, bias=bias)


        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.ART_decoder_level2 = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=window_size[1], interval=interval[1],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[1])])

        self.Restormer_decoder_level2 = nn.Sequential(*[RestormerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[1])])
        self.dec_fuse_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)


        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.ART_decoder_level1 = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], window_size=window_size[0], interval=interval[0],
                                            ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_art_blocks[0])])

        self.Restormer_decoder_level1 = nn.Sequential(*[RestormerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, \
                                                                       bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_restormer_blocks[0])])
        self.dec_fuse_level1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)


        self.ART_refinement = nn.ModuleList([ARTBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], window_size=window_size[0], interval=interval[0],
                                        ds_flag=0 if i % 2 == 0 else 1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) for i in range(num_refinement_blocks)])

        # self.Restormer_refinement = nn.Sequential(*[RestormerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, \
        #                                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        # self.fuse_refinement = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=3, stride=1, padding=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    # def forward(self, inp_img, aux_img):
    #     stack = []
    #     bs, _, H, W = inp_img.shape

    #     fuse_img = torch.cat((inp_img, aux_img), 1)
    #     inp_enc_level1 = self.patch_embed(fuse_img)  # b,hw,c

    def forward(self, inp_img):
        bs, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c

        ######################################### Encoder level1 ############################################
        art_enc_level1 = inp_enc_level1
        restor_enc_level1 = inp_enc_level1

        ### ART encoder block
        for layer in self.ART_encoder_level1:
            art_enc_level1 = layer(art_enc_level1, [H, W])
        
        ### Restormer encoder block
        restor_enc_level1 = rearrange(restor_enc_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        restor_enc_level1 = self.Restormer_encoder_level1(restor_enc_level1) ### (b, c, h, w)

        art_enc_level1 = rearrange(art_enc_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        out_enc_level1 = torch.cat((art_enc_level1, restor_enc_level1), 1)
        out_enc_level1 = self.enc_fuse_level1(out_enc_level1)
        out_enc_level1 = rearrange(out_enc_level1, "b c h w -> b (h w) c").contiguous()
        # print("out_enc_level1:", out_enc_level1.max(), out_enc_level1.min())


        ######################################### Encoder level2 ############################################
        ### ART encoder block
        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        art_enc_level2 = inp_enc_level2
        restor_enc_level2 = inp_enc_level2

        for layer in self.ART_encoder_level2:
            art_enc_level2 = layer(art_enc_level2, [H // 2, W // 2])
        
        ### Restormer encoder block
        restor_enc_level2 = rearrange(restor_enc_level2, "b (h w) c -> b c h w", h=H//2, w=W//2).contiguous()
        restor_enc_level2 = self.Restormer_encoder_level2(restor_enc_level2)
        
        art_enc_level2 = rearrange(art_enc_level2, "b (h w) c -> b c h w", h=H//2, w=W//2).contiguous()
        out_enc_level2 = torch.cat((art_enc_level2, restor_enc_level2), 1)
        out_enc_level2 = self.enc_fuse_level2(out_enc_level2)
        out_enc_level2 = rearrange(out_enc_level2, "b c h w -> b (h w) c").contiguous()
        
        # print("out_enc_level2:", out_enc_level2.max(), out_enc_level2.min())

        ######################################### Encoder level3 ############################################
        ### ART encoder block
        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        art_enc_level3 = inp_enc_level3
        restor_enc_level3 = inp_enc_level3

        for layer in self.ART_encoder_level3:
            art_enc_level3 = layer(art_enc_level3, [H // 4, W // 4])

        ### Restormer encoder block
        restor_enc_level3 = rearrange(restor_enc_level3, "b (h w) c -> b c h w", h=H//4, w=W//4).contiguous()
        restor_enc_level3 = self.Restormer_encoder_level3(restor_enc_level3)

        art_enc_level3 = rearrange(art_enc_level3, "b (h w) c -> b c h w", h=H//4, w=W//4).contiguous()
        out_enc_level3 = torch.cat((art_enc_level3, restor_enc_level3), 1)
        out_enc_level3 = self.enc_fuse_level3(out_enc_level3)
        out_enc_level3 = rearrange(out_enc_level3, "b c h w -> b (h w) c").contiguous()

        # print("out_enc_level3:", out_enc_level3.max(), out_enc_level3.min())


        ######################################### Encoder level4 ############################################
        ### ART encoder block
        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        art_latent = inp_enc_level4
        restor_latent = inp_enc_level4

        for layer in self.ART_latent:
            art_latent = layer(art_latent, [H // 8, W // 8])

        ### Restormer encoder block
        restor_latent = rearrange(restor_latent, "b (h w) c -> b c h w", h=H//8, w=W//8).contiguous()
        restor_latent = self.Restormer_latent(restor_latent)
        
        art_latent = rearrange(art_latent, "b (h w) c -> b c h w", h=H//8, w=W//8).contiguous()
        latent = torch.cat((art_latent, restor_latent), 1)
        latent = self.enc_fuse_level4(latent)
        latent = rearrange(latent, "b c h w -> b (h w) c").contiguous()

        # print("latent feature:", latent.max(), latent.min())


        ######################################### Decoder level3 ############################################
        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c

        ### ART decoder block
        art_dec_level3 = inp_dec_level3
        restor_dec_level3 = inp_dec_level3

        for layer in self.ART_decoder_level3:
            art_dec_level3 = layer(art_dec_level3, [H // 4, W // 4])

        ### Restormer decoder block
        restor_dec_level3 = rearrange(restor_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        restor_dec_level3 = self.Restormer_decoder_level3(restor_dec_level3)

        art_dec_level3 = rearrange(art_dec_level3, "b (h w) c -> b c h w", h=H//4, w=W//4).contiguous()
        out_dec_level3 = torch.cat((art_dec_level3, restor_dec_level3), 1)
        out_dec_level3 = self.dec_fuse_level3(out_dec_level3)
        out_dec_level3 = rearrange(out_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c


        ######################################### Decoder level2 ############################################
        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c

        ### ART decoder block
        art_dec_level2 = inp_dec_level2
        restor_dec_level2 = inp_dec_level2

        for layer in self.ART_decoder_level2:
            art_dec_level2 = layer(art_dec_level2, [H // 2, W // 2])

        ### Restormer decoder block
        restor_dec_level2 = rearrange(restor_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        restor_dec_level2 = self.Restormer_decoder_level2(restor_dec_level2)

        art_dec_level2 = rearrange(art_dec_level2, "b (h w) c -> b c h w", h=H//2, w=W//2).contiguous()
        out_dec_level2 = torch.cat((art_dec_level2, restor_dec_level2), 1)
        out_dec_level2 = self.dec_fuse_level2(out_dec_level2)
        out_dec_level2 = rearrange(out_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c


        ######################################### Decoder level1 ############################################
        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)

        ### ART decoder block
        art_dec_level1 = inp_dec_level1
        restor_dec_level1 = inp_dec_level1

        for layer in self.ART_decoder_level1:
            art_dec_level1 = layer(art_dec_level1, [H, W])

        ### Restormer decoder block
        restor_dec_level1 = rearrange(restor_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        restor_dec_level1 = self.Restormer_decoder_level1(restor_dec_level1)

        art_dec_level1 = rearrange(art_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        out_dec_level1 = torch.cat((art_dec_level1, restor_dec_level1), 1)
        out_dec_level1 = self.dec_fuse_level1(out_dec_level1)
        out_dec_level1 = rearrange(out_dec_level1, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c


        ######################################### final Refinement ############################################
        for layer in self.ART_refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1 #, stack



# def build_model(args):
#     return ART_Restormer(
#         inp_channels=2,
#         out_channels=1,
#         dim=32,
#         num_art_blocks=[2, 4, 4, 6],
#         num_restormer_blocks=[2, 2, 2, 2],
#         num_refinement_blocks=4,
#         heads=[1, 2, 4, 8],
#         window_size=[10, 10, 10, 10], mlp_ratio=4.,
#         qkv_bias=True, qk_scale=None,
#         interval=[24, 12, 6, 3],
#         ffn_expansion_factor = 2.66,
#         LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
#         bias=False)



def build_model(args):
    return ART_Restormer(
        inp_channels=1,
        out_channels=1,
        dim=32,
        num_art_blocks=[2, 4, 4, 6],
        num_restormer_blocks=[2, 2, 2, 2],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        window_size=[10, 10, 10, 10], mlp_ratio=4.,
        qkv_bias=True, qk_scale=None,
        interval=[24, 12, 6, 3],
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        bias=False)