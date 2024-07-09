"""
2023/10/05
Xiaohan Xing
Build a simple network with several convolution layers. 
Input: concat (undersampled kspace of FS-PD, fully-sampled kspace of PD modality)
Output: interpolated kspace of FS-PD
"""

import torch
from torch import nn
from torch.nn import functional as F
from .DataConsistency import DataConsistency


class mmConvKSpace(nn.Module):
    """
    Interpolate the kspace data of the target modality with several conv layers.
    """
    def __init__(
        self, args,
        chans: int = 32,
        drop_prob: float = 0.0):
        
        super().__init__()

        self.in_chans = 4
        self.out_chans = 2
        self.chans = chans
        self.drop_prob = drop_prob

        self.small_conv = ConvBlock(self.in_chans, self.chans, 3, self.drop_prob)
        self.mid_conv = ConvBlock(self.in_chans, self.chans, 5, self.drop_prob)
        self.large_conv = ConvBlock(self.in_chans, self.chans, 7, self.drop_prob)

        self.freq_filter = nn.Sequential(nn.Conv2d(3 * self.chans, self.chans, kernel_size=1, padding=0, bias=False),
                                         nn.ReLU(),
                                         nn.Conv2d(self.chans, 1, kernel_size=1, padding=0, bias=False),
                                         nn.Sigmoid())

        self.final_conv = ConvBlock(3 * self.chans, self.out_chans, 3, self.drop_prob)

        # self.conv_blocks = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        # self.conv_blocks.append(ConvBlock(self.chans, self.chans * 2, drop_prob))
        # self.conv_blocks.append(ConvBlock(self.chans * 2, self.chans, drop_prob))
        # self.conv_blocks.append(ConvBlock(self.chans, self.out_chans, drop_prob))

        self.dcs = DataConsistency()


    def forward(self, kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace: Input 4D tensor of shape `(N, H, W, 4)`.
            mask: Down-sample mask `(N, 1, len, 1)`

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        k_in = torch.cat((kspace, ref_kspace), 1)
        # k_in = k_in.permute(0, 3, 1, 2)
        output = k_in
        # print("image shape:", image.shape)

        output1 = self.small_conv(output)
        output2 = self.mid_conv(output)
        output3 = self.large_conv(output)
        # print("output of the three conv blocks:", output1.shape, output2.shape, output3.shape)
        # print("input kspace range:", kspace.max(), kspace.min())
        # print("conv1_output range:", output1.max(), output1.min())
        # print("conv2_output range:", output2.max(), output2.min())
        # print("conv3_output range:", output3.max(), output3.min())

        output = torch.cat((output1, output2, output3), 1)
        # print("output range before freq_filter:", output.max(), output.min())

        ### Element-wise multiplication in the Frequency domain = full image size conv in the image domain.
        spatial_weights = self.freq_filter(output)
        # print("spatial_weights range:", spatial_weights.max(), spatial_weights.min())
        output = spatial_weights * output
        # print("output range after freq_filter:", output.max(), output.min())

        output = self.final_conv(output)

        output = output.permute(0, 2, 3, 1)
        # print("output before DC layer:", output.shape)
        # print("output range before DC layer:", output.max(), output.min())
        # output = output + kspace  ## residual connection

        output, mask = self.dcs(output, kspace.permute(0, 2, 3, 1), mask)
        mask_output = mask * output
        
        # mask_output = output

        return output.permute(0, 3, 1, 2), mask_output.permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            kernel_size: size of the convolution kernel.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
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


def build_model(args):
    return mmConvKSpace(args)
