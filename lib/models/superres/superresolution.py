# Modified from https://github.com/NVlabs/eg3d/blob/main/eg3d/training/superresolution.py
from torch import nn
from torch.nn import functional as F
import numpy as np

class SynthesisLayer(nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self.padding)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.up > 1:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=True)
            # x = F.upsample(x, scale_factor=self.up, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.act(x)
        return x

# for 256x256 generation
class SuperresolutionHybrid2X(nn.Module):
    def __init__(self, in_channels, hidden_channels, img_resolution):
        super().__init__()
        assert img_resolution == 512
        self.input_resolution = 256
        
        self.conv0 = SynthesisLayer(in_channels, hidden_channels, resolution=512, up=2)
        # self.conv1 = SynthesisLayer(hidden_channels, hidden_channels, resolution=256)
        self.torgb = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        # init torgb
        self.torgb.weight.data.uniform_(-1e-5, 1e-5)
        self.torgb.bias.data.zero_()


    def forward(self, x_in):
        # assert x_in.shape[-1] >= self.input_resolution
        final_res = self.input_resolution * 2
        x_in_up = F.interpolate(x_in, size=(final_res, final_res), mode='bilinear', align_corners=True)
        x = self.conv0(x_in)
        # x = self.conv1(x)

        y = self.torgb(x)
        out = x_in_up.add_(y)

        return out

class SuperresolutionHybrid4X(nn.Module):
    def __init__(self, in_channels, hidden_channels, img_resolution):
        super().__init__()
        assert img_resolution == 512
        self.input_resolution = 128
        
        self.conv0 = SynthesisLayer(in_channels, hidden_channels, resolution=256, up=2)
        self.conv1 = SynthesisLayer(in_channels, hidden_channels, resolution=512, up=2)
        self.torgb0 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.torgb1 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        # init torgb
        self.torgb0.weight.data.uniform_(-1e-5, 1e-5)
        self.torgb0.bias.data.zero_()
        self.torgb1.weight.data.uniform_(-1e-5, 1e-5)
        self.torgb1.bias.data.zero_()


    def forward(self, x_in):

        assert x_in.shape[-1] >= self.input_resolution
        final_res = self.input_resolution * 2
        x_in_up = F.interpolate(x_in, size=(final_res, final_res), mode='bilinear', align_corners=True)
        x = self.conv0(x_in)
        y = self.torgb0(x)
        x_in2 = x_in_up.add_(y)

        final_res *= 2
        x_in_up2 = F.interpolate(x_in2, size=(final_res, final_res), mode='bilinear', align_corners=True)
        x_ = self.conv1(x_in2)
        y_ = self.torgb1(x_)
        out = x_in_up2.add_(y_)

        return out


if __name__ == '__main__':
    import torch
    super_res_block = SuperresolutionHybrid2X(in_channels=18, hidden_channels=64, img_resolution=256)
    x = torch.randn((1, 18, 128, 128))
    x_up = super_res_block(x)
