import torch
import torch.nn as nn
import torchvision
# [N, C ,H, W] -> [N, g,  C/g ,H, W]-> [N, C/g, g, H, W]->[N, C ,H, W]
x = torch.randn(4, 6, 3, 3)
y = x.reshape(4, 2, 3, 3, 3)
print(y.shape)
z = y.permute(0, 2, 1, 3, 4)
print(z.shape)
v = y.reshape(4, 6, 3, 3)
print(v.shape)

# torch 中存在封装好的残差网络和混洗网络
net = torchvision.models.shufflenet_v2_x0_5()
net2 = torchvision.models.resnet18()
net3 = torchvision.models.mobilenet_v2()

print(net)
print(net2)

#自带l2正则，weight_decay l2正则化的惩罚因子
opt = torch.optim.Adam(mynet.parameters(), weight_decay=0.2)
# 如果想只对某一层正则话
loss = ...+ 0.1*mynet.conv.paramaters()


## drop out，随机抑制百分之20的神经元，在当前批次不让参与训练
m = nn.Dropout(p=0.2)
input = torch.randn(24, 26)
output = m(input)