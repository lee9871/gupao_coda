# from cat_dog import img
from cnn_56_2 import Mynet_cnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from mlp_55_2_2 import Mydataset
from torchvision import transforms, utils
import os
mydata = Mydataset("cat_dog/img")
batchsize = 120
train_data = DataLoader(mydata, batch_size=batchsize, shuffle=True)
if torch.cuda.is_available():
    mynet = Mynet_cnn().cuda()
else:
    mynet = Mynet_cnn()

loss_fn = nn.MSELoss()
lr = 0.001
opt = torch.optim.Adam(mynet.parameters(), lr=lr)
epochsize = 5
for epoch in range(epochsize):
    for i, (x, y) in enumerate(train_data):
        # print(x.size())
        # print(y)
        # 做轴交换 permute用多维的，transposet做二维，输入[ batch_size, channels, height_1, width_1 ]
        x = x.permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            x = x.cuda()
            y = torch.zeros(y.size(0), 2).scatter_(1, y.long().view(-1, 1), 1).cuda()
        else:
            x = x
            y = torch.zeros(y.size(0), 2).scatter_(1, y.long().view(-1, 1), 1)
        ouput = mynet(x)
        # y = y.long().reshape(batchsize, )
        loss = loss_fn(ouput, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print("epoch:{},batchsize:{},loss:{:.5}".format(epoch, i, loss.detach()))
            out = torch.argmax(ouput, 1)
            y_1 = torch.argmax(y, 1)
            acc = np.mean(np.array(out.cpu() == y_1.cpu()))
            print(acc)

    torch.save(mynet, 'models/mynet_cnn.pth')


