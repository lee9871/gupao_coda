import torch.nn as nn


class Mynet_cnn(nn.Module):
    def __init__(self):
        super(Mynet_cnn, self).__init__()
        # 图片一般用Conv2d，心电用Conv1d，视频用Conv3d；猫狗图片（100，100，3）,通道一般是2的n次方
        self.cov1 = nn.Sequential(
        # （98，98，16）
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
        # （49，49，16）
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cov2 = nn.Sequential(
        # （47，47，32）
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
        # （23，23，32）
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cov3 = nn.Sequential(
        # （21，21，64）
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
        # （10，10，64）
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cov4 = nn.Sequential(
        # （8，8，128）
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0,),
            nn.ReLU(),
        # （4，4，128）
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Linear(4*4*128, 64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax()
        )

# 交叉熵损失，里面带有softmax函数，均方差损失不带Softmax

    def forward(self, x):
        ou1 = self.cov1(x)
        ou2 = self.cov2(ou1)
        ou3 = self.cov3(ou2)
        ou4 = self.cov4(ou3)
        ou4 = ou4.reshape(-1,2048 )
        ou5 = self.layer1(ou4)
        ou6 = self.layer2(ou5)
        return ou6

