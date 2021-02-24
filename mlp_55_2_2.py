import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, utils
import os
import numpy as np

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])


class Mydataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(self.path)

    def __getitem__(self, item):
        label = torch.Tensor([int(self.dataset[item].split('.')[0])])
        img_path = os.path.join(self.path, self.dataset[item])
        img = Image.open(img_path)
        # data = preprocess(img)
        data = torch.Tensor((np.array(img)/255 -0.5)/0.5)
        return data, label

    def __len__(self):
        return len(self.dataset)


# if __name__ == "__main__":
#     mydataset = Mydataset("cat_dog/img")
#     print(mydataset[3])
#     # 将第288张图片的tensor转为图片
#     # mydataset = np.array((mydataset[3][0]*0.5+0.5)*255, dtype=np.uint8)
#     # img = Image.fromarray(mydataset, "RGB")
#     # img.show()
mydata = Mydataset("cat_dog/img")
batchsize = 100
train_data = DataLoader(mydata, batch_size=batchsize, shuffle=True)

class Mynet3(nn.Module):
    def __init__(self, input, hidden1, hidden2, hidden3, output):
        super(Mynet3, self).__init__()
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
    mynet = Mynet3(30000, 512, 256, 128, 2).cuda()
else:
    mynet = Mynet3(30000, 512, 256, 128, 2)

loss_fn = nn.CrossEntropyLoss()
lr = 0.01
opt = torch.optim.Adam(mynet.parameters(), lr=lr)
epochsize = 3
if __name__ == "__main__":
    for epoch in range(epochsize):
        for i, (x, y) in enumerate(train_data):
            # print(x.size())
            print(y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                x = x
                y = y
            x_data = x.reshape(batchsize, -1)
            ouput = mynet(x_data)
            # 将y  reshape    变成      torch.Size([100])
            y = y.long().reshape(batchsize,)
            loss = loss_fn(ouput, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("epoch:{},batchsize:{},loss:{:.5}".format(epoch, i, loss.detach()))
                out = torch.argmax(ouput, 1)
                y_1= torch.argmax(y.reshape(batchsize,1) , 1)
                acc = np.mean(np.array(out.cpu() == y_1.cpu()))
                print(acc)
