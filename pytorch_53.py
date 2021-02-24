import torch

# Tensor生成浮点数float32
a = torch.Tensor([[1, 2, 3]])
# a = torch.FloatTensor([[1,2,3]]) 与上句一样
print(a)
print(a.dtype)
print(a.device)
print(a.shape)
print(a.size())


# tensor生成整型int64
a = torch.tensor([[1, 2, 3]])
#  tensor生成整型int32,dtype 修改数据类型的时候，只能在torch.tensor，不能torch.Tensor
a = torch.tensor([[1, 2, 3]], dtype=torch.int32)
print(a)
print(a.dtype)
print(a.size())
a = torch.tensor([[1], [2], [3]], dtype=torch.int32)
print(a.size())


# 随机生成一个3行2列的张量
b = torch.Tensor(3, 2)
print(b)
print(b.size())

# Tensor里面在加()生成一个张量
b = torch.Tensor((3, 2))
print(b)
print(b.size())


# requires_grad 需要浮点类型,tensor里面的数据需要是浮点型
b = torch.tensor((3, 2.5), requires_grad=True)
print(b)

#左开右闭，1-8，之间所有数
c = torch.arange(1,9)
print(c)
# tensor([1, 2, 3, 4, 5, 6, 7, 8])

#左开右闭，1-8之间所有数,步长为2
c = torch.arange(1,9,2)
print(c)
# tensor([1, 2, 3, 4, 5, 6, 7, 8])

#左开右闭，1-8之间所有数,步长为2，每步计算加上梯度；需要变为浮点型
c = torch.arange(1.0,9,2)
print(c)
# tensor([1, 2, 3, 4, 5, 6, 7, 8])


##左开右闭从2-6 平均取4个数
d = torch.linspace(2, 6, 4)
print(d)

##左开右闭从2-6 平均取4个数,是可以的，原因是 浮点型的
d = torch.linspace(2, 6, 4, requires_grad=True)
print(d)

## 左开右闭从10的2次方-10的6次方 平均取4个数
d = torch.logspace(1.0,9,2)
print(d)

## 全是1
e = torch.ones(3, 2)
print(e)
## 全是0
e = torch.zeros(3, 2)
print(e)
## 全是空
e = torch.empty(3, 2)
print(e)
## 对角矩阵
e = torch.eye(3, 2)
print(e)
## 0-1均匀分布之间随机生成3个
e = torch.rand(3)
print(e)
## 标准正态分布之间随机生成3个整数型
e = torch.randint(0, 4,(3,4))
print(e)
## 标准正态分布之间随机生成3个
e = torch.randn(3)
print(e)
## 标准正态分布之间随机生成3个4行5列
e = torch.randn(3, 4, 5)
print(e)


##torch.from_numpy将array转为torch,且内存共享
import numpy as np
f = np.array([1, 2, 3])
print(f)
f_1 = torch.from_numpy(f)
print(f)
f[0] = 55
print(f)
print(f_1)

##torch.tensor 可把标量、列表、元组，多维数组转为张量，但内存不共享
f = np.array([1, 2, 3])
f_1 = torch.tensor(f)
print(f)
f[0] = 55
print(f)
print(f_1)

# .numpy() 将tensor转为array
b = torch.tensor([1, 2, 3])
c = b.numpy()
b += 2
print(c)


#### 张量的运算
## 加法
a = torch.ones(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)
print(a + b)
print(torch.add(a, b))
result = torch.Tensor(2, 3)
print(result)
print(torch.add(a, b, out=result))

## 数乘
print(a*2)

## 点乘，内积
print(a*b)
print(torch.mul(a, b))

##多次点乘
print(torch.pow(a, 3))
print(torch.pow(b, 3))
print(a**3)

## 矩阵乘法
b = b.reshape(3, 2)
print(torch.matmul(a,b))


## 布尔运算
a = torch.ones(2, 3)
print(a > 3)

## 求平均值
a = torch.randn(2, 3)
print(a.mean())


## 将数据放在gpu上计算
a = torch.ones((2, 3))
print(a.device)
b = torch.randint(3,4, (2,3))
#放在GPU上跑，gpu上的速度比较慢原因是：
# 数据计算时首先传入主存再到CPU再到显存再到GPU,返回的路径是一样的；
# 进去的时间比较长，进去之后的速度就很快了
# a = a.cuda(0)
print(a)
print(b)

## 将数据放在cpu上计算
a = a.cpu()
print(a.device)

## 判断是否有cuda
if torch.cuda.is_available():
    a = a.cuda()
    b = b.cuda()
    print(a+b)
else:
    print(a.device)


## 导入数据集
from torch.utils.data import Dataset

class Mydataset(Dataset):
    def __init__(self):
        self.xs = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.ys = torch.Tensor([[0], [0], [0], [1]])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        x = self.xs[index]
        y = self.ys[index]
        return x,y

mydata = Mydataset()
print(mydata[1])
print(mydata[0][0])

##DataLoader 返回的迭代器, 可以把全部数据加载，并且打乱数据
# shuffle 打乱数据
# enumerate 将可遍历的数据对象，比如元组、列表、字符串 做个索引序列，列出索引、数据，一般和for一起用
from torch.utils.data import DataLoader
train_data = DataLoader(mydata, batch_size=2, shuffle=False)
for i ,(data_x, data_y) in enumerate(train_data):
    print(i)
    print("data_x:",data_x)
    print("data_y:",data_y)




