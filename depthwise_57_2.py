import torch
import torch.nn as nn

x = torch.randn(1, 6, 6, 6)
conv = nn.Conv2d(6, 6, 3)
group_conv = nn.Conv2d(6, 6, 3, groups=3)
dep_group_conv = nn.Conv2d(6, 6, 3, groups=6)

y = dep_group_conv(x)
print(y.shape)
sum = 0
params = dep_group_conv.parameters()
for i in params:
    print(i)
    sum += i.numel()
    print(sum)