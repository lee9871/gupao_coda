import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import PIL.ImageDraw as draw
from PIL import ImageFont
import matplotlib.pyplot as plt
from mlp_54_4 import Mynet2

# mynet = Mynet2(784, 256, 128, 64, 10)
net = torch.load("models/mynet.pth")
print(net)

batchsize = 100
#  下载数据
# Normalize 对原有数据标准化，原有数据0-1之前，故减0.5 除以0.5
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_dataset = datasets.MNIST(root="MNIST_data", train=False, transform=data_transform, download=False)
# 加载数据
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batchsize)

for data in test_loader:
    img, label = data[0], data[1]
    # 图片做了归一化 *255
    img2 = img.numpy()[0][0]*255
    # 将array转为图片
    img2 = Image.fromarray(img2)
    img2_draw = draw.ImageDraw(img2)

    img = img.reshape(batchsize, -1)
    if torch.cuda.is_available():
        img1 = img.cuda()
        label = label.cuda()
    else:
        img1 = img
        label = label
    out = net(img1)
    out2 = out.max(dim=1).indices[0].item()
    ## 字体
    font1 = ImageFont.truetype("arial.ttf", size=10)
    img2_draw.text(xy=(0, 0), text=str(out2), fill=255, font=font1)
    img2_draw.text(xy=(22, 0), text=str(label[0].item()), fill=255, font=font1)
    # print("网络输出：{},原图标签".format(out2, label[0].item()))


