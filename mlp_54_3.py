import matplotlib.pyplot as plt
import numpy as np
import torch

x = torch.tensor([5.], requires_grad=True)
print(x)
b = x**2
opt = torch.optim.SGD([x], lr=0.1)# lr 学习率
for i in range(100):
    opt.zero_grad()## 清空梯度
    b = x ** 2
    b.backward(retain_graph=True)
    opt.step()#更新梯度
    print("第{}次".format(i), x)
    plt.ion()
    plt.clf()
    plt.plot(10, 10)
    m = np.linspace(-5, 5, 100)
    plt.plot(m, m**2, color="g")
    plt.plot(x.detach().numpy()[0], x.detach().numpy()[0]**2, "or")
    plt.pause(0.1)
    plt.ioff()
plt.show()

# loss = loss_fn(outputs, ys)
# opt.zero_grad()
# loss.backward()
# opt.step()