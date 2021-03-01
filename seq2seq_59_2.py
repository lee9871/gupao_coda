import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import sys

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class MyData(Dataset):
    def __init__(self, root):
        self.transform = data_transform
        self.list = []
        for filename in os.listdir(root):
            x = os.path.join(root, filename)
            ys = x.split('.')[1]
            y = self.one_hot(ys)
            self.list.append([x, np.array(y)])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        img_path, label = self.list[item]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def one_hot(self, x):
        z = np.zeros([4, 10])
        for i in range(4):
            x = int(x[i])
            z[i][x] += 1
        return z

sys.path.append('../')
# mydata = MyData('data')
data_loader = DataLoader(mydata, batch_size=1, shuffle=True)
for i,(x, y) in enumerate(data_loader):
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    if i > 1:
        break