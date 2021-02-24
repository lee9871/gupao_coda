import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# class Rnn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
#         self.afine = nn.Linear(64, 10)
#
#     def forward(self, x):
#         #True： NSV N：batchsize S: sequence 序列长度(一共有多少个序列） V：input size（每个序列的维度）。False：SNV
#         x = x.reshape(-1, 28, 28)
#         batch = x.shape[0]
#         # 方向*层数， N：batchsize， hidden_size
#         h0 = torch.zeros(1, batch, 64)
#         output, _ = self.rnn(x, h0)
#         output = output[:, -1, :]
#         output = self.afine(output)
#         return output


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.afine = nn.Linear(64, 10)

    def forward(self, x):
        #True： NSV N：batchsize S: sequence 序列长度 V：input size。False：SNV
        x = x.reshape(-1, 28, 28)
        batch = x.shape[0]
        # 方向*层数， N：batchsize， hidden_size
        h0 = torch.zeros(1, batch, 64)
        c0 = torch.zeros(1, batch, 64)

        output, _ = self.rnn(x, (h0, c0))
        output = output[:, -1, :]
        output = self.afine(output)
        return output

# 数据加载
batchsize = 100
num_epoch = 14
lr = 0.01

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root="MNIST_data", train=True, transform=data_transform, download=False)
test_dataset = datasets.MNIST(root="MNIST_data", train=False, transform=data_transform, download=False)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batchsize)

# for i, (x, y) in enumerate(train_loader):
#     if i > 1:
#         break
#     print(x.size())
#     print(x, y)

# 模型加载
# mynet = Rnn()
mynet = LSTM()

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(mynet.parameters(), lr=lr)

# 训练测试数据
if __name__ =="__main__":
    for epoch in range(num_epoch):
        for i, (x, y) in enumerate(train_loader):
            # if i > 1:
            #     break

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                x = x
                y = y
            # print(x.size())
            # torch.Size([100, 1, 28, 28])
            out = mynet(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("epoch:{}, batch:{}, loss:{}".format(epoch, i, loss))
            mynet.eval()



        loss = 0
        Accuracy = 0
        for data in test_loader:
            # print(data[0])
            # print(data[0].shape)
            # x_test = data[0].reshape(batchsize, -1)

            x_test = data[0].reshape(-1, 28, 28)
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

