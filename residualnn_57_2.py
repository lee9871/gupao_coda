import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),
            nn.ReLU(),
            # nn.MaxPool2d(3)
        )

    def forward(self, x):
        out = self.conv(x)
        x = x.repeat(1, 2, 1, 1)
        print(x.size())
        return out + x


if __name__ == "__main__":
    x = torch.Tensor(4, 3, 28, 28)
    resNet = ResNet()
    y = resNet(x)

    # y = resNet.foward(x)
    print(y.shape)

