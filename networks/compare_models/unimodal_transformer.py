import torch

from torch import nn
from .transformer_modules import build_transformer
from einops.layers.torch import Rearrange
from .MINet_common import default_conv as conv, Upsampler


# add by YunluYan


class ReconstructionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReconstructionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        return out


def build_head():
    return ReconstructionHead(input_dim=1, hidden_dim=12)



class CMMT(nn.Module):

    def __init__(self, args):
        super(CMMT, self).__init__()

        self.head = build_head()

        HEAD_HIDDEN_DIM = 12
        PATCH_SIZE = 16
        INPUT_SIZE = 240
        OUTPUT_DIM = 1 

        patch_dim = HEAD_HIDDEN_DIM* PATCH_SIZE ** 2
        num_patches = (INPUT_SIZE // PATCH_SIZE) ** 2

        self.transformer = build_transformer(patch_dim)

        self.patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, patch_dim))


        self.p1 = PATCH_SIZE
        self.p2 = PATCH_SIZE

        self.tail = nn.Conv2d(HEAD_HIDDEN_DIM, OUTPUT_DIM, 1)


    def forward(self, x):

        b,_ , h, w = x.shape

        x = self.head(x)

        x= self.patch_embbeding(x)

        x += self.pos_embedding

        x = self.transformer(x)  # b HW p1p2c

        c = int(x.shape[2]/(self.p1*self.p2))
        H = int(h/self.p1)
        W = int(w/self.p2)

        x = x.reshape(b, H, W, self.p1, self.p2, c)  # b H W p1 p2 c
        x = x.permute(0, 5, 1, 3, 2, 4)  # b c H p1 W p2
        x = x.reshape(b, -1, h, w,)
        x = self.tail(x)

        return x


def build_model(args):
    return CMMT(args)








