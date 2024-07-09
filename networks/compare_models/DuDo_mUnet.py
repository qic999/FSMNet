"""
Xiaohan Xing, 2023/10/25
传入两个模态的kspace data, 经过kspace network进行重建。
然后把重建之后的kspace data变换回图像，然后送到image domain的网络进行图像重建。
"""
# coding: utf-8
import torch
from torch import nn
from torch.nn import functional as F
from .DataConsistency import DataConsistency
from dataloaders.BRATS_kspace_dataloader import ifft2c

from .Kspace_mUnet import kspace_mmUnet


class mmConvKSpace(nn.Module):
    """
    Interpolate the kspace data of the target modality with several conv layers.
    """
    def __init__(
        self, 
        chans: int = 32,
        drop_prob: float = 0.0):
        
        super().__init__()

        self.in_chans = 4
        self.out_chans = 2
        self.chans = chans
        self.drop_prob = drop_prob

        self.small_conv = kspace_ConvBlock(self.in_chans, self.chans, 3, self.drop_prob)
        self.mid_conv = kspace_ConvBlock(self.in_chans, self.chans, 5, self.drop_prob)
        self.large_conv = kspace_ConvBlock(self.in_chans, self.chans, 7, self.drop_prob)

        self.freq_filter = nn.Sequential(nn.Conv2d(3 * self.chans, self.chans, kernel_size=1, padding=0, bias=False),
                                         nn.ReLU(),
                                         nn.Conv2d(self.chans, 1, kernel_size=1, padding=0, bias=False),
                                         nn.Sigmoid())

        self.final_conv = kspace_ConvBlock(3 * self.chans, self.out_chans, 3, self.drop_prob)

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

        output = torch.cat((output1, output2, output3), 1)
        # print("output range before freq_filter:", output.max(), output.min())

        ### Element-wise multiplication in the Frequency domain = full image size conv in the image domain.
        spatial_weights = self.freq_filter(output)
        # print("spatial_weights range:", spatial_weights.max(), spatial_weights.min())
        output = spatial_weights * output
        # print("output range after freq_filter:", output.max(), output.min())

        output = self.final_conv(output)

        output = output.permute(0, 2, 3, 1)

        output, mask = self.dcs(output, kspace.permute(0, 2, 3, 1), mask)
        mask_output = mask * output
        
        return output.permute(0, 3, 1, 2), mask_output.permute(0, 3, 1, 2)


class DuDo_mmUnet(nn.Module):

    def __init__(
        self, args, 
        input_dim: int = 3,
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

        # self.kspace_net = mmConvKSpace()
        self.kspace_net = kspace_mmUnet(args)
        
        self.down_sample_layers = nn.ModuleList([ConvBlock(self.in_chans, self.chans, self.drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

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
            )
        )


    def complex_abs(self, data):
        """
        data: [B, 2, H, W]
        """
        if data.size(1) == 2:
            return (data ** 2).sum(dim=1).sqrt()
        elif data.size(-1) == 2:
            return (data ** 2).sum(dim=-1).sqrt()
    

    def normalize(self, data, mean, stddev, eps=0.0):
        """
        Normalize the given tensor.
        Applies the formula (data - mean) / (stddev + eps).
        """
        return (data - mean) / (stddev + eps)


    def forward(self, args, kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, \
                aux_image: torch.Tensor, t1_max: torch.Tensor, t2_max: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """        
        if args.domain == "dual":
            recon_kspace, masked_kspace = self.kspace_net(kspace, ref_kspace, mask)
            # print("input kspace range:", kspace.max(), kspace.min())
            # print("recon kspace range:", recon_kspace.max(), recon_kspace.min())
            image = self.complex_abs(ifft2c(recon_kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()
        elif args.domain == "image":
            image = self.complex_abs(ifft2c(kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()
            recon_kspace, masked_kspace = None, None

        image = image * t2_max.to(torch.float32).view(-1, 1, 1, 1)
        aux_image = aux_image * t1_max.to(torch.float32).view(-1, 1, 1, 1)

        LR_image = self.complex_abs(ifft2c(kspace.permute(0, 2, 3, 1))).unsqueeze(1) #.detach()
        LR_image = LR_image * t2_max.to(torch.float32).view(-1, 1, 1, 1)

        ### normalize the input data with (x-mean)/std, and save the mean and std for recovery.
        t1_mean = aux_image.view(aux_image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        t1_std = aux_image.view(aux_image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()
        t2_mean = image.view(image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        t2_std = image.view(image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()

        # t2_LR_mean = LR_image.view(LR_image.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1).detach()
        # t2_LR_std = LR_image.view(LR_image.shape[0], -1).std(dim=1).view(-1, 1, 1, 1).detach()

        # print("t2_kspace_recon mean and std:", t2_mean.view(-1), t2_std.view(-1))
        # print("t2_LR mean and std:", t2_LR_mean.view(-1), t2_LR_std.view(-1))

        aux_image = self.normalize(aux_image, t1_mean, t1_std, eps=1e-11)
        image = self.normalize(image, t2_mean, t2_std, eps=1e-11)
        LR_image = self.normalize(LR_image, t2_mean, t2_std, eps=1e-11)


        # print("normalized t1 range:", aux_image.max(), aux_image.min())
        # print("normalized kspace_recon_t2 range:", image.max(), image.min())

        data_stats = {"t1_mean": t1_mean, "t1_std": t1_std, "t2_mean": t2_mean, "t2_std": t2_std}

        stack = []
        feature_stack = []
        # output = torch.cat((image, aux_image), 1)  ### shape: (N, in_chans, H, W)`.
        output = torch.cat((image, LR_image, aux_image), 1) #.detach()

        # print("LR image range:", LR_image.max(), LR_image.min())
        # print("kspace_recon image range:", image.max(), image.min())
        # print("aux_image range:", aux_image.max(), aux_image.min())

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            feature_stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            # print("down-sampling layer output:", output.shape)

        output = self.conv(output) ### from [4, 256, 15, 15] to [4, 512, 15, 15]

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

        return output, image, recon_kspace, masked_kspace, data_stats



class kspace_ConvBlock(nn.Module):
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
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.InstanceNorm2d(out_chans),
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
    return DuDo_mmUnet(args)
