import torch
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

batchsize = 100
lr = 0.01
num_epoch = 5
#  下载数据
# Normalize 对原有数据标准化，原有数据0-1之前，故减0.5 除以0.5
# data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root="MNIST_data", train=True, transform=data_transform, download=False)
test_dataset = datasets.MNIST(root="MNIST_data", train=False, transform=data_transform, download=False)
# 加载数据
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batchsize)
# print(train_dataset.train_data.size())
# print(train_dataset.train_labels.size())
# # 打印数据
for i, (x, y) in enumerate(test_loader):
    print(x.size())
    print(x)



# 构建网络
# BatchNorm1d 对每个batch数据标准化，批量标准化在
class Mynet2(nn.Module):
    def __init__(self, input, hidden1, hidden2, hidden3, output):
        super(Mynet2, self).__init__()
        # self.layer = nn.Linear(n_features=input, out_features=hidden1, bias=True), nn.ReLU(inplace=True)
        self.layer = nn.Sequential(nn.Linear(in_features=input, out_features=hidden1, bias=True),
                                   nn.BatchNorm1d(hidden1),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=hidden1, out_features=hidden2, bias=True),
                                   nn.BatchNorm1d(hidden2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=hidden2, out_features=hidden3, bias=True),
                                   nn.BatchNorm1d(hidden3),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=hidden3, out_features=output, bias=True),
                                   )


    def forward(self, x):
        y = self.layer(x)
        return y


if torch.cuda.is_available():
    mynet = Mynet2(784, 256, 128, 64, 10).cuda()
else:
    mynet = Mynet2(784, 256, 128, 64, 10)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(mynet.parameters(), lr=lr)

if __name__ =="__main__":
    for epoch in range(num_epoch):
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                x = x
                y = y
            # print(x.size())
            x = x.reshape(batchsize, -1)
            out = mynet(x)
            ## 交叉熵损失函数会自动把y[100,1],100个数字变成one-hot
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i %10 ==0:
                print("epoch:{},batchsize:{},loss:{:.5}".format(epoch, i, loss.detach()))

            mynet.eval() ## 模型评估

        loss = 0
        Accuracy = 0
        for data in test_loader:
            x_test = data[0].reshape(batchsize, -1)
            y_test = data[1]
            if torch.cuda.is_available():
                x_test = x_test.cuda()
                y_test = y_test.cuda()
            else:
                x_test = x_test
                y_test = y_test
            out = mynet(x_test)
            ## 计算总损失
            loss += loss_fn(out, y).detach()*y_test.size()[0]
            y_pred = out.max(dim=1).indices
            # y_pred = torch.argmax(out, 1)
            Accuracy += (y_pred == y_test).sum()

        print(y_pred)
        print(y_test)
        print("test_loss:{},eval_acc:{}".format(loss/len(test_dataset), Accuracy/len(test_dataset)))

    # 保存模型，参数是pth的形式
    torch.save(mynet, "models/mynet.pth")




            # num +=1
            # if num >1:
            #     break
            # print(data)







