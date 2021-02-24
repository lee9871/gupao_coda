import torch
import torch.nn as nn


# [N, C ,H, W] -> [N, g,  C/g ,H, W]-> [N, C/g, g, H, W]->[N, C ,H, W]
class ShffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        c, n, h, w = x.shape
        y = x.reshape(c, self.groups, n//self.groups, h, w)
        z = y.permute(0, 2, 1, 3, 4)
        v = y.reshape(c, n ,h, w)
        return v

if __name__=="__main__":
    shffleBlock = ShffleBlock(3)
    x = torch.randn(1, 3, 32, 3)
    y = shffleBlock(x)
    print(y.shape)


# class ShffleNet(nn.Module):
#     def __init__(self):
#         super(ShffleNet, self).__init__()
#         conv = nn.Conv2d(4, 6, 3)
#
#     def forward(self, x):
#         x = shffleBlock(x)
#         y = conv(x)
