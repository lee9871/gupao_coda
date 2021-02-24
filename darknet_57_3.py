import torch
import torch.nn as nn


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.sub_module = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),


        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sub_module = nn.Sequential(
            ConvolutionLayer(in_channels, in_channels//2, 1, 1, 0),
            ConvolutionLayer(in_channels//2, in_channels, 3, 1, 1),

        )

    def forward(self, x):
        return self.sub_module(x) + x


class DarknetLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sub_module = nn.Sequential(
            ResidualLayer(in_channels)
        )

    def forward(self, x):
        return self.sub_module(x)

# if __name__ =='__main__':
#     x = torch.randn(2)
#     ResidualLayer = ResidualLayer()