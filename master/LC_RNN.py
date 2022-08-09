# pytorch LSTM for regression
import numpy as np
from numpy import vstack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from sklearn.preprocessing import minmax_scale
from torch.nn.utils.rnn import pack_sequence
import torch
import gensim
torch.manual_seed(3)
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
torch.autograd.set_detect_anomaly(True)
#LSTM(sequence_len,batchsize,dim)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class RNN(nn.Module):
    """for rnn"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, h_state):
        rnn_out, h_state = self.rnn(x, h_state)
        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]
            out.append(self.output_layer(every_time_out))
        return torch.stack(out, dim=1), h_state       # torch.stack expansion to [1, output_size, 1]
    from LCattention.lcrnn_model import LCrnn,RNN
    from Data_Utils import *
    from Plot_Utils import *
    from Math_Utils import *
    def load_model(save_path):
        model = torch.load(save_path)
        return model
    steps,gap=200,10
    data_path='../SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    LC=LCrnn(DS,steps,gap)
    save_path='../SCNRL_Master/baseline/lcrnn_model_4.pkl'
    model = load_model(save_path)
    for name, param in model.named_parameters():
        print(name,param.shape)
    print('OK')
    for name in model.state_dict():
        print(name)
        print(model.state_dict()[name].shape)
    print('OKL')


LC = LCrnn(DS, steps, gap)
time = DS.time[0:len(DS.time):gap].reshape(-1, 1)
print(len(time))
ori_data = DS.ori_data
# ori_data=DS.transform_back(DS.ori_data,'ori2train')
# plt.show()
x_0 = DS.ori_data[0, :]
path_all = []
h_state = None
yhat = DS.ori_data[200:200 + gap, :]
for i in range(len(time)):
    x_0 = yhat
    # x_0 = DS.ori_data[i+200:i+200+gap,:]
    x_t = torch.from_numpy(x_0.reshape(-1, gap, DS.dim))
    yhat, h_state = model(x_t, h_state)
    h_state = h_state.detach()
    yhat = yhat.detach().numpy()
    yhat = yhat.reshape(-1, DS.dim)
    # print('x_0 shape is',x_0.shape)
    # jacobian(torch.from_numpy(x_0),torch.from_numpy(yhat),gap)
    # path_all.append(list(self.Dataset.transform_back(yhat[0],'train2back')))
    path_all.append(yhat[0])
print('OKL')
for dim in range(44):
    print(dim)
    plt.plot(time[:len(time)], ori_data[200:200 + len(time):1, dim])
    plt.show()
    plt.plot(time, [i[dim] for i in path_all])
    plt.show()
    plt.plot(time, [abs(ori_data[200 + i, dim] - path_all[i][dim]) for i in range(len(path_all))])
    plt.show()
    plt.plot([i[dim] for i in path_all], [i[dim + 1] for i in path_all], 'r')
    plt.plot(ori_data[200:200 + len(time):1, dim], ori_data[200:200 + len(time):1, dim + 1], 'y')
    plt.show()

from baseline.lcrnn_model import LCrnn,RNN
from Data_Utils import *
from Plot_Utils import *



def fftTransfer(timeseries, n=10, fmin=0):
    #print(len(timeseries))
    yf = abs(np.fft.fft(timeseries))#,axes=1))  # 取绝对值
    yfnormlize = yf / len(timeseries)  # 归一化处理
    #print('yfnormlize is',yfnormlize.shape)
    #conv1 = np.real(np.fft.ifftn(yf))
    #plt.plot(conv1 - 0.5)  # 为看清楚，将显示区域下拉0.5
    #plt.plot(yfnormlize - 1)
    #plt.show()
    yfhalf = yfnormlize[range(int(len(timeseries)/2))]  # 由于对称性，只取一半区间
    yfhalf = yfhalf * 2   # y 归一化
    xf = np.arange(len(timeseries))  # 频率
    xhalf = xf[range(int(len(timeseries) / 2))]  # 取一半区间
    x = np.arange(len(timeseries))  # x轴
    #plt.plot(x, timeseries)
    #plt.title('Original wave')
    #plt.plot(xhalf, yfhalf, 'r')
    #plt.title('FFT of Mixed wave(half side frequency range)', fontsize=10, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
    fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]
    xwbest = signal.argrelextrema(yfhalf, np.greater)
    #plt.plot(xwbest[0][:n], fwbest[:n], 'o', c='yellow')
    #plt.show(block=False)
    #plt.show()
    xorder = np.argsort(-fwbest)  # 对获取到的极值进行降序排序，也就是频率越接近，越排前
    #print('xorder = ', xorder)
    xworder = list()
    xworder.append(xwbest[x] for x in xorder)  # 返回频率从大到小的极值顺序
    fworder = list()
    fworder.append(fwbest[x] for x in xorder)  # 返回幅度
    if len(fwbest) <= n:
        fwbest = fwbest[fwbest >= fmin].copy()
        #print(fwbest)
        #print(len(timeseries)/xwbest[0][:len(fwbest)])
        return len(timeseries)/xwbest[0][:len(fwbest)], fwbest[:n]    #转化为周期输出
    else:
        fwbest = fwbest[fwbest >= fmin].copy()
        #print('len fwbest is',len(fwbest))
        #print('xwbest is also',xwbest)
        return len(timeseries)/xwbest[0][:n], fwbest[:n]  # 只返回前n个数   #转化为周期输出

def load_model(save_path):
    model = torch.load(save_path)
    return model

all_nt=[]
all_fshape=[]
for dim in range(44):
    nt,fshape=fftTransfer(DS.train_dl[:10000,dim],10,0)
    all_nt.append(nt.min())
    #print(nt.max())
    all_fshape.append(fshape)
print(len(all_fshape))
all_fshape=np.vstack(all_fshape)
print(len(all_nt))
all_nt=np.vstack(all_nt)
print('nt shape',all_nt.shape)
print(all_nt.max())
print('all_fshape',all_fshape.shape)
print(all_fshape.min())
