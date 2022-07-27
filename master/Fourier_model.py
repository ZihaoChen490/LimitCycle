import torch
from torch.autograd import Variable
import torch.optim as optim
import argparse
from torch import Tensor, nn
from torch.nn import Linear, Sigmoid, Module, ReLU, Sigmoid
import torch, gensim

torch.manual_seed(3)
from Data_Utils import *
from Plot_Utils import *

signal_length = 1000
learning_rate = 0.0001
res_length = 1000
dims = 4
from torch.nn.init import kaiming_uniform_


class FT_model(Module):
    def __init__(self, dims, batch_size):
        super(FT_model, self).__init__()
        self.dims = dims
        self.batch_size = batch_size
        tvals = np.arange(signal_length).reshape([-1, 1])
        freqs = np.arange(signal_length).reshape([1, -1])
        self.arg_vals = torch.tensor([2 * np.pi * tvals * freqs / signal_length])
        self.y_pred = None
        self.h1 = Linear(self.batch_size * 2, self.batch_size * 50)
        kaiming_uniform_(self.h1.weight)
        self.act1 = Sigmoid()
        self.h2 = Linear(self.batch_size * 50, self.batch_size * 2)

    def forward(self, X):
        if self.y_pred is not None:
            y_pred_now = self.ecoder(X)
            y_pred = (y_pred_now + self.y_pred) / 2
            self.y_pred = y_pred.detach()
        else:
            y_pred = self.ecoder(X)
            self.y_pred = y_pred.detach()
        X = self.h1(y_pred)
        X = self.act1(X)
        X = self.h2(X)
        res = self.decoder(X)
        return res

    def ecoder(self, X):
        dim = 0
        Fc = [i for i in range(self.dims)]
        for i in range(self.dims):
            Fc[i] = Variable(
                torch.from_numpy(np.random.random([self.batch_size, self.batch_size * 2]) - 0.5).to(torch.float32),
                requires_grad=True)
        optimizer = optim.SGD([Fc[dim]], lr=0.1)
        for i in range(2000):
            optimizer.zero_grad()
            y_pred = torch.matmul(X, Fc[dim])
            y_real = y_pred[:, 0:signal_length]
            y_imag = y_pred[:, signal_length:]
            amplitudes = (torch.sqrt(y_real ** 2 + y_imag ** 2) / signal_length)
            phases = torch.atan2(y_imag, y_real)
            sinusoids = amplitudes * torch.cos(self.arg_vals + phases)
            reconstructed_signal = torch.sum(sinusoids, axis=1)
            encoder_loss = torch.sum(torch.square(X - reconstructed_signal))
            encoder_loss.backward(retain_graph=True)
            optimizer.step()
            if i % 100 == 1:
                print('In encoder, epoch', str(i), 'with loss ', encoder_loss)
                if encoder_loss.detach().numpy() < 1e-6 or isnan(encoder_loss):
                    return y_pred
        return y_real, y_imag

    def decoder(self, y_pred):
        y_real = y_pred[:, 0:signal_length]
        y_imag = y_pred[:, signal_length:]
        amplitudes = (torch.sqrt(y_real ** 2 + y_imag ** 2) / signal_length)
        phases = torch.atan2(y_imag, y_real)
        sinusoids = amplitudes * torch.cos(self.arg_vals + phases)
        reconstructed_signal = torch.sum(sinusoids, axis=1)
        return reconstructed_signal

data_path='../SNCRL_Dataset/CellCycle/'
DS=Dataset(data_path,1)
dim=0
FT = FT_model(dims, signal_length)
optimizer = optim.SGD(FT.parameters(), lr=0.01)
x = torch.from_numpy(DS.train_dl[:1000 * 200, :dims].T.reshape(dims, -1)).to(torch.float32)
losses = []
for epoch in range(300):
    t = 1000
    losses = []
    for i in range(10):
        optimizer.zero_grad()
        data = x[dim, i * t:(i + 1) * t].reshape(1, -1)
        reconstructed_signal = FT(data)
        target = x[dim, (i + 1) * t:(i + 2) * t].reshape(1, -1)
        loss = torch.sum(torch.square(target - reconstructed_signal))
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 2 == 0:
            print('now the training epoch', epoch, 'with step', str(i), ', loss ', loss)
            losses.append(loss.detach())
    print('Now the loss is', sum(losses))
    if sum(losses) < 1e-6 or isnan(losses):
        break