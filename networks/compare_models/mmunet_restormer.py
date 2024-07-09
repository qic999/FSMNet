"""
2023/07/12, 
将两个模态的图像concat作为input image, 在Unet的每层特征后面都连接一个restormer block提取channel-wise long-range dependent feature. 
"""

import torch
from torch import nn
from torch.nn import functional as F
from .restormer_block import TransformerBlock


class Unet_ART(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """
    

    def __init__(
        self, args, 
        input_dim: int = 2,
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
        self.img_size = 240

        self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)


        num_blocks=[2, 2, 2, 2]
        heads=[1, 2, 4, 8]
        ffn_expansion_factor = 2.66

        self.restor_block1 = nn.ModuleList([TransformerBlock(dim=chans, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, \
            bias=False, LayerNorm_type='WithBias', pre_norm=False) for i in range(num_blocks[0])])
        self.restor_block2 = nn.ModuleList([TransformerBlock(dim=int(chans * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, \
            bias=False, LayerNorm_type='WithBias', pre_norm=False) for i in range(num_blocks[1])])
        self.restor_block3 = nn.ModuleList([TransformerBlock(dim=int(chans * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, \
            bias=False, LayerNorm_type='WithBias', pre_norm=False) for i in range(num_blocks[2])])
        self.restor_block4 = nn.ModuleList([TransformerBlock(dim=int(chans * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, \
            bias=False, LayerNorm_type='WithBias', pre_norm=False) for i in range(num_blocks[3])])
        self.restormer_blocks = nn.Sequential(self.restor_block1, self.restor_block2, self.restor_block3, self.restor_block4)


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
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                # nn.Tanh(),
            )
        )

    # def forward(self, image: torch.Tensor) -> torch.Tensor:
    def forward(self, image: torch.Tensor, aux_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        # output = image
        output = torch.cat((image, aux_image), 1)
        # print("image shape:", image.shape)
        unet_features_stack, restormer_features_stack = [], []

        # print("input image range:", output.max(), output.min())

        # apply down-sampling layers
        for layer, restor_block in zip(self.down_sample_layers, self.restormer_blocks):
            output = layer(output)
                        
            feature = output
            for restormer_layer in restor_block:
                feature = restormer_layer(feature)
                print("encoder features:", feature.shape)

            unet_features_stack.append(output)
            restormer_features_stack.append(feature)

            # print("Unet feature range:", output.max(), output.min())
            # print("Restormer feature range:", feature.max(), feature.min())

            output = feature
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            # print("downsampling layer:", output.shape)

        output = self.conv(output)
        print("intermediate layer feature:", output.shape)

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
            # print("unsampling layer:", output.shape)

        return output #, unet_features_stack, restormer_features_stack


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
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.BatchNorm2d(out_chans),
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
        # for layer in range(len(self.layers)):
        #     print(self.layers[layer])
        #     image = self.layers[layer](image)
        #     # print("feature range of this layer:", image.max(), image.min())
        # return image
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
            # nn.BatchNorm2d(out_chans),
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
    return Unet_ART(args)
