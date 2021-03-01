import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from seq2seq_59_2 import MyData




## 整个网络分为 encoder  decoder 和连贯主网络
class Encoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(128, 128, 2, True)

    def forward(self, x):
        # x的维度是 NCHW（n, 3, 60 ,120) 对W切，（n, 180 ,120)
        x = x.reshpe(-1, 180, 120)
        x = x.permute(0, 2, 1)  # （n ,120, 180)
        x = x.reshpe(-1, 180)   # （n*120, 180)
        fc1_out = self.fc1(x)   # （n*120, 128)
        fc1_out = fc1_out.reshape(-1, 120, 128)  # NSV （n， 120, 128)
        out, _ = self.lstm(fc1_out)     # N S H*direct（n， 120, 128)
        out = out[:, -1, :]     # 取最后一个隐藏单元输出值（n, 128)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.lstm2 = nn.LSTM(128, 128, 2, True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 128)
        batchsize = x.size()[0]
        x = x.expand(batchsize, 4, 128)
        lstm2_out = self.lstm2(x)   #NSH*direct （n, 4, 128)
        lstm2_out = lstm2_out.reshpe(-1, 128)  # （n*4, 128)
        fc2_out = self.fc2(lstm2_out)      # （n*4, 10)
        fc2_out = fc2_out.reshape(-1, 4, 10)
        return fc2_out


class Mynet(nn.Module):
    def __init__(self):
        super.__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

# 一种模型保存和加载的方式如下
# # save
# torch.save(model.state_dict(), PATH)
#
# # load
# model = MyModel(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()


if __name__ == '__main__':
    mynet = Mynet().cuda()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam([{"param": mynet.encoder.parameters()}, {'param': mynet.decoder.parameters()}])
    save_path ='models/seq2se2model'
    if os.path.exists(save_path):
        mynet.load_state_dict(torch.load(save_path))

    train_data = MyData("data")
    batch_size = 64
    num_works = 4
    epoch = 100
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_works= num_works)
    for epoch in range(epoch):
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                x = x
                y = y
            out = mynet(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                print("epoch:{},batchsize:{},loss:{:.5}".format(epoch, i, loss.detach()))
                out = torch.argmax(out, 2).detach().cpu().numpy()
                y_1 = torch.argmax(y.reshape(batch_size, 1), 2).detach().cpu().numpy()
                acc = np.mean(np.all(out.cpu() == y_1.cpu(), axis=1))
                print(acc)

        torch.save(mynet.state_dict(), save_path)




