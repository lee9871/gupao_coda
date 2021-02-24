import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, utils
import os
import numpy as np
filePath = 'cat_dog/img/'
for i, j, k in os.walk(filePath):
    img_name = k
y_data = [int(l[0]) for l in img_name]
x_data = [filePath+l for l in img_name]


preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])


def default_loader(path):
    img_pil = Image.open(path)
    # img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor


class Mydataset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = x_data
        self.target = y_data
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)


mydata = Mydataset()
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
    mynet = Mynet3(30000, 256, 128, 64, 10).cuda()
else:
    mynet = Mynet3(30000, 256, 128, 64, 10)

loss_fn = nn.CrossEntropyLoss()
lr = 0.01
opt = torch.optim.Adam(mynet.parameters(), lr=lr)
epochsize = 3
for epoch in range(epochsize):
    for i, (x, y) in enumerate(train_data):
        # print(x.size())
        # print(y)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        else:
            # x = torch.from_numpy(x)
            # y = np.array(y, dtype=float)
            y = torch.tensor(y)

        x_data = x.reshape(batchsize, -1)
        ouput = mynet(x_data)
        loss = loss_fn(ouput, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print("epoch:{},batchsize:{},loss:{:.5}".format(epoch, i, loss.detach()))

# path = 'cat_dog/img/0.3.jpeg'
# img_pil = Image.open(path)
# # img_pil = img_pil.resize((224, 224))
# img_tensor = preprocess(img_pil)