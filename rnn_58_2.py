import torch.nn as nn
import torch


class Rnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.afine = nn.Linear(64, 10)

    def forward(self, x):
        #True： NSW N：batchsize S: sequence 序列长度 V：input size（词向量维度）。False：SNV
        x = x.reshape(-1, 28, 28)
        batch = x.shape[0]
        # 方向*层数， N：batchsize， hidden_size
        h0 = torch.zeros(1, batch, 64)
        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]
        output = self.afine(output)
        return torch.softmax(x, dim=1)

# #Rnn 其中h0 可以不给
# rnn = nn.RNN(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)

# LSTM  其中h0 c0 可以不给
# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

# Rnn 和LSTM 区别在于，需要输入传c0，输出传出cn