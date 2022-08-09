import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack, sqrt
import pandas as pd
import networkx as nx
from torch.autograd import Variable
import torch, time, sys,math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_tweedie_deviance
from torch import Tensor,nn
from torch.nn import Linear,Sigmoid,ReLU, Module,ELU
from torch.optim import SGD
from torch.nn import MSELoss,CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention
from baseline.lcrnn_model import *

import os
import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchdiffeq.torchdiffeq import odeint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchdiffeq import odeint
#data_sorce='C:/Aczh work place/3_paper/algo_new/data/'
#argv='C:/Aczh work place/3_paper/algonew/experiment-pend/'


def remap(x, out_min, out_max):
    in_min, in_max = np.min(x), np.max(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

beats = np.load('control_beats_6.npy')

class LcODE(Module):
    def __init__(self, n_inputs):
        super(LcODE, self).__init__()
        self.hidden1 = Linear(n_inputs, 60)
        kaiming_uniform_(self.hidden1.weight)
        self.act1 = ELU()
        self.hidden2 = Linear(60, 30)
        kaiming_uniform_(self.hidden2.weight)
        self.act2 = ELU()
        self.hidden3 = Linear(30, 10)
        kaiming_uniform_(self.hidden3.weight)
        self.act3 = ELU()
        self.hidden4 = Linear(10, n_inputs)
        kaiming_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
        self.nfe = 0
    def forward(self,t, X):
        self.nfe+=1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X=self.act3(X)
        X = self.hidden4(X)
        X=self.act4(X)
        return X

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


def generate_beats(beats_npy=beats,
                   ntotal=len(beats[0]) - 1,
                   nsample=200,
                   noise_std=0.00,
                   ):
    orig_ts = np.linspace(0, 1, num=len(beats_npy[0]) - 1)
    samp_ts = orig_ts[:nsample]
    xs = np.array([np.diff(single_beat) for single_beat in beats_npy])
    ys = np.array([single_beat[:-1] for single_beat in beats_npy])
    orig_trajs = np.stack((xs, ys), axis=2)  # [:1000]
    samp_trajs = []
    for i in range(len(orig_trajs)):
        orig_traj = orig_trajs[i]
        orig_traj = remap(orig_traj, -1, 1)
        orig_trajs[i] = orig_traj
        samp_traj = orig_traj.copy()
        idx0 = np.random.randint(ntotal - nsample)
        samp_traj = samp_traj[idx0:idx0 + nsample]
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)
    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    return orig_trajs, samp_trajs, orig_ts, samp_ts
N = 5000
latent_dim = 50
nhidden = 50
rnn_nhidden = 20
obs_dim = 2
nspiral = N #len(beats)
noise_std = 0.005
ntotal = 499
cutoff = 100
nsample = ntotal - cutoff

class SCNRL:
    def __init__(self,Dataset,steps,gap):
        self.Dataset=Dataset
        self.steps=steps
        self.gap=gap
        self.lam=0.5
        self.lr=0.000001
        self.weight_decay=1e-5
        self.epoch=15
        self.NewIndex, self.TargetSet = np.loadtxt('NewIndex.txt'),np.loadtxt('TargetSet.txt')
        if self.NewIndex is None:
            self.NewIndex,self.TargetSet=ReIndex(Dataset.train_dl[:len(Dataset.train_dl)//10],gap,num_model)

    def graph(self,Index):
        if Index>0:
            self.Node_list=list(set(self.NewIndex[Index-1]+self.NewIndex[Index]+self.NewIndex[Index+1]))
        else:
            self.Node_list=list(set(self.NewIndex[Index]+self.NewIndex[Index+1]))
        lis=[]
        for i in self.Node_list:
            for j in self.Node_list:
                if i!=j and (i in self.NewIndex[j] or j in self.NewIndex[i]):
                    lis.append((i,j,{'weight':(self.Dataset.train_dl[i,:]-self.Dataset.train_dl[j,:]).std()}))
        self.G=nx.Graph(lis)
        self.A=nx.adjacency_matrix(self.G)
        nx.degree(self.G)
        self.L=nx.normalized_laplacian_matrix(self.G)
        self.eig = np.linalg.eigvals(self.L.A)
        print(self.eig)
        return self.eig

    def creat_gauss_kernel(self,kernel_size=3, sigma=1, k=1):
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return gauss


    def train_sto_mlp_model(self,num_model,steps=200,gap=10):
        model = [MLP(DS.dim) for i in range(num_model)]
        train_dl=self.Dataset.train_dl
        print(train_dl.shape)
        data_all = []
        loss_res = []
        torch.manual_seed(8)
        np.random.seed(5)
        optimizer = [torch.optim.Adam(model[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) for i in range(num_model)]
        data=torch.from_numpy(train_dl)
        regularization_loss,t = 0,len(train_dl)//steps
        print('each length of the dataset is',t)
        for epoch in range(self.epoch):
            #for step in range(steps//num_model):
            for step in range(steps-70):
                print('......................................................................')
                print('Now we strat to train the',step,'th step in epoch:', epoch, 'with total step:', steps)
                for rd in range(t-gap):
                    inputs=[data[(step)*t+rd,:] for i in range(num_model)]
                    #NewIndex, TargetSet = ReIndex(train_dl, gap, num_model,(step)*t+rd)
                    targets=[torch.from_numpy(self.TargetSet[int(self.NewIndex[(step)*t+rd][i]),:]) for i in range(num_model)]
                    #inputs = [data[(step*num_model+i)*t+rd,:] for i in range(num_model)]
                    #targets=[data[(step*num_model+i)*t+rd+gap,:] for i in range(num_model)]
                    [optimizer[i].zero_grad() for i in range(num_model)]
                    yhat = [model[i](inputs[i]) for i in range(num_model)]
                    loss = [(1/(1+abs(inputs[i]-targets[i]).std()))*abs(L2_loss(yhat[i], targets[i]) -self.lam * torch.norm(yhat[i] - sum(yhat) /len(model))) for i in range(len(model))]
                    #loss = [(L2_loss(yhat[i], targets[i])) for i in range(num_model)]
                    [loss[i].backward(retain_graph=True) for i in range(num_model)]
                    [optimizer[i].step() for i in range(num_model)]
                print('Now the loss=',loss)
            loss_res.append([loss[i].detach().numpy() for i in range(len(model))])
            for i in range(len(model)):
                torch.save(model[i],'C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/model_ito_weight' + str(i) + '.pkl')
                print('model have been saved')
            if epoch>4 and (sum([(loss_res[epoch-1][i]-loss_res[epoch][i]) for i in range(len(model))]))<1e-10:
                break
        try:
            data_a=self.plot_model(1000,model,True)
            data_all.append(data_a[-1][-1])
        except:
            print('Unsave model')
        return data_all
    def plot_train(self,train_dl,dim):
        time=self.Dataset.time[0:len(self.Dataset.time):gap].reshape(-1,1)
        print(len(time))
        train_data=self.Dataset.transform_back(self.Dataset.train_dl,'train2back')
        plt.plot(time, train_data[:len(self.Dataset.train_data)//10, dim])
        plt.show()
        plot_vector_fields(train_dl,dim)


    def evaluate_model_on_test(self,model,step=60,gap=1):
        predictions, actuals = list(), list()
        data = torch.from_numpy(self.Dataset.test_dl)
        t=1
        print('test_dl is',len(self.Dataset.test_dl))
        for i in range((len(self.Dataset.test_dl)-gap)//10):
            inputs = data[i * t, :]
            targets = data[i * t + gap, :]
            yhat = sum([model[i](inputs) for i in range(len(model))])/len(model)
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
#        actual = actual.reshape((len(actual), 2))
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
        #MAE
        return mse
    def load_model(self,save_path):
        model = [MLP(DS.dim) for i in range(num_model)]
        for i in range(num_model):
            model[i] = torch.load(save_path+'model_ito_l1_' + str(i) + '.pkl')
        return model

    def evaluate_model_ori_data(self, path,gap=1):
        model=self.load_model(path)
        predictions, actuals = list(), list()
        data= torch.from_numpy(self.Dataset.ori_data)
        t = 1
        print('ori_data is', self.Dataset.transform_back(self.Dataset.ori_data.shape,'ori2back'))
        for i in range(self.Dataset.ori_data.shape[1]):
            inputs = data[i * t, :]
            targets = data[i * t + gap, :]
            yhat = sum([model[i](inputs) for i in range(len(model))]) / len(model)
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            #actual = actual.reshape((len(actual), 2))
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
        return mse

    def plot_model(self,step, model, flag=False):
        ax1 = plt.subplot(1, 1, 1)
        path_all = []
        print('step:', step)
        x_n = [[[0.05 * j, 0.05 * i] for i in range(21)] for j in range(21)]
        for i in range(len(x_n)):
            for j in range(len(x_n[0])):
                x_0 = x_n[j][i]
                print('第', i, j, '个', 'x_0 is :', x_0)
                pre_path = []
                for k in range(step):
                    pre_path.append(list(x_0))
                    yhat = predict(x_0, model)
                    x_0 = yhat[0]
                path_all.append(pre_path)
        for path in path_all:
            plt.plot([i[0] for i in path], [i[1] for i in path])
            print('res', path)
        if flag:
            plt.show()
        return path_all

    def plot_time_series_model(self, model,steps,gap,dim, flag=False):
        ax1 = plt.subplot(1, 1, 1)
        time=DS.time[0:len(DS.time):2*gap].reshape(-1,1)
        print(len(time))
        ori_data=self.Dataset.transform_back(self.Dataset.ori_data,'ori2back')
        #plt.show()
        x_0=self.Dataset.ori_data[0,:]
        path_all = []
        for i in range(len(time)):
            x_0 = np.hstack([self.Dataset.ori_data[i,:],time[i,:]])
            yhat = predict(x_0, model)
            #path_all.append(list(self.Dataset.transform_back(yhat[0],'train2back')))
            path_all.append(yhat[0])
        for dim in range(45):
            print(dim)
            plt.plot(time, ori_data[:len(self.Dataset.ori_data)//20:gap, dim])
            plt.show()
            plt.plot(time,[i[dim] for i in path_all])
            plt.show()
        print('Over Now')
        return path_all



    def load_model(save_path):
        model = torch.load(save_path)
        return model


if __name__=='__main__':
    num_model = 8
    data_path = 'C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS = Dataset(data_path, 1)
    steps, gap = 140, 10
    SC = SCNRL(DS, steps, gap)
    save_path = 'C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    # SC.train_sto_mlp_model(num_model,steps,gap)


    steps = 140
    gap = 10
    data_path = 'C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS = Dataset(data_path, 1)
    LC = LCrnn(DS, steps, gap)
    save_path = 'C:/Aczh work place/3_paper/SCNRL_Master/baseline/lcrnn_model_0.pkl'
    model = load_model(save_path)
    dim = 1
    # data_all=SC.plot_model(1000,model,True)
    data_all = LC.plot_time_series_model(model, steps, gap, dim)

