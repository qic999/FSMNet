"""
在lequan的代码上，去掉transformer-based feature fusion部分。
直接用CNN提取特征之后concat作为multi-modal representation. 用普通的decoder进行图像重建。
把T1作为guidance modality, T2作为target modality.
"""
# coding: utf-8
import torch
from torch import nn
from torch.nn import functional as F


class mmUnet(nn.Module):
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

        self.down_sample_layers1 = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        self.down_sample_layers2 = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers1.append(ConvBlock(ch, ch * 2, drop_prob))
            self.down_sample_layers2.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch * 2, ch * 2, drop_prob)

        # self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        # ch = chans
        # for _ in range(num_pool_layers - 1):
        #     self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
        #     ch *= 2
        # self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv = nn.Sequential(nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1), nn.Tanh())


    def forward(self, image: torch.Tensor, aux_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # output = torch.cat([image, aux_image], 1)
        # for layer in self.down_sample_layers:
        #     output = layer(output)
        #     output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)


        # apply down-sampling layers
        output = image
        for layer in self.down_sample_layers1:
            output = layer(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            # print("down-sampling layer output:", output.shape)
        output_m1 = output

        output = aux_image
        for layer in self.down_sample_layers2:
            output = layer(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)        
        output_m2 = output

        # print("two modality outputs:", output_m1.shape, output.shape)
        output = torch.cat([output_m1, output_m2], 1)


        output = self.conv(output) ### from [4, 256, 15, 15] to [4, 512, 15, 15]

        # apply up-sampling layers
        for transpose_conv in self.up_transpose_conv:
            output = transpose_conv(output)

        output = self.up_conv(output)
        # print("model output:", output.shape)

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
    return mmUnet(args)
