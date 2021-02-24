import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 定义与运算数据集，线性可分
class Mydataset(Dataset):
    def __init__(self):
        self.xs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).type(torch.FloatTensor)
        self.ys = torch.tensor([[0], [0], [0], [1]]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

mydataset = Mydataset()
print(mydataset[0])

#读取加载数据集
train_data = DataLoader(mydataset, batch_size=4, shuffle=False)

def __del__(self):
    print("%s 去了" % self.name)

    tom = Cat("Tom")
    print(tom.name)

#定义网络结构
class Mynet(nn.Module):
    def __init__(self):
        #这是对继承自父类的属性进行初始化，而且是用父类的初始化方法来初始化继承的属性；父类为nn.Module
        super(Mynet, self).__init__()
        self.w = nn.Parameter(torch.randn(2, 1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

# 问题：1、forward的形式，另外损失函数定义好了就可以直接到模型中
# 3、nn.Parameter

if torch.cuda.is_available():
    net = Mynet().cuda()
else:
    net = Mynet()

loss_fn = nn.MSELoss()
opt = torch.optim.SGD(net.parameters(), lr=0.1)
a1 = []
b1 = []
if __name__ == "__main__":
    for epoch in range(100):
        for i, (data_x, data_y) in enumerate(train_data):
            if torch.cuda.is_available():
                xs = data_x.cuda()
                ys = data_y.cuda()
            else:
                xs = data_x
                ys = data_y
            outputs = net(xs)
            # print(outputs)
            loss = loss_fn(outputs, ys)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.cpu().detach().numpy()
            output = outputs.cpu().detach().numpy()
            xs = xs.cpu().detach().numpy()
            ys = ys.cpu().detach().numpy()
            # print(output, "\n", ys)

        # 打开实时画图
        plt.ion()
        # 清楚之前画图
        plt.clf()
        window1 = plt.subplot(2, 1, 1)
        a1.append(epoch)
        b1.append(loss)
        plt.plot(a1, b1)
        window1 = plt.subplot(2, 1, 2)
        ## ob代表把blue
        plt.plot(xs[0][0], xs[0][1], "ob")
        plt.plot(xs[1][0], xs[1][1], "ob")
        plt.plot(xs[2][0], xs[2][1], "ob")
        plt.plot(xs[3][0], xs[3][1], "or")
        point = np.max(ys)
        plt.plot([point+0.5, 0], [0, point+0.5])
        point1 = np.max(outputs.cpu().detach().numpy())
        plt.plot([point1+0.5, 0], [0, point1+0.5], linestyle="--")

        #停顿0.1s
        plt.pause(0.1)
        # 关闭画图
        plt.ioff()

    plt.show()





