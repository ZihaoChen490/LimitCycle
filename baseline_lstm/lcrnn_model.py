# pytorch LSTM for regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch import Tensor,nn
from torch.nn import Linear, Sigmoid, Module
from torchkeras import summary
import torch, gensim
torch.manual_seed(3)
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
torch.autograd.set_detect_anomaly(True)
#LSTM(sequence_len,batchsize,dim)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.output_layer = Linear(in_features=hidden_size, out_features=output_size)


    def simple_batch_norm_1d(self,x, gamma, beta):
        eps = 1e-5
        x_mean = torch.mean(x, dim=0, keepdim=True)  # 保留维度进行 broadcast
        x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)
        for m in self.modules():
            if isinstance(m, Linear):
        #        gamma = torch.ones(m.weight.data.shape[1])
        #        beta = torch.zeros(m.weight.data.shape[1])
        #        m.weight.data = self.simple_batch_norm_1d(m.weight.data, gamma, beta)
                m.weight.data =torch.div(m.weight.data,torch.norm(m.weight.data))
        #        print(m.weight.data)
                #m.weight.data.normal_(0, 1)
        #    elif isinstance(m, nn.BatchNorm3d):
        #        m.weight.data.fill_(1)
        rnn_out, h_state = self.rnn(x, h_state)   # h_state是之前的隐层状态
        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]       # 相当于获取每个时间点上的输出，然后过输出层
            out.append(self.output_layer(every_time_out))
        return torch.stack(out, dim=1), h_state       # torch.stack扩成[1, output_size, 1]

class LCrnn:
    def __init__(self,Dataset,steps,gap):
        self.Dataset=Dataset
        self.steps=steps
        self.gap=gap
        self.lam=0.5
        self.lr=0.0001
        self.weight_decay=1e-5
        self.epoch=300
        self.InputSet,self.TargetSet=Id2Id(self.Dataset.train_dl,gap)

    def train_rnn(self,model,save_path):
        train_dl=self.Dataset.train_dl
        gap=self.gap
        steps=self.steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001 ,weight_decay=0.001)
        data = torch.from_numpy(train_dl)
        print('datashape is',data.shape)
        regularization_loss,t = 0,len(train_dl)//steps
        loss_res=[]
        loss0=0
        for epoch in range(40):
            for step in range(steps-135):
                print('.........................................')
                print('Now we strat to train epoch:',epoch,'step is',step)
                h_state = None  # 初始化隐藏层状态
                #h0 = model.init_hidden(DS.train_dl.shape[1])
                for i in range(t//gap-2):
                    model.zero_grad()
                    inputs = data[step*t+i*gap:step*t+i*gap+gap,:].reshape(-1,gap,self.Dataset.dim)
                    targets = data[step*t+i*gap+gap:step*t+(i+2)*gap,:].reshape(-1,gap,self.Dataset.dim)
                    #print(inputs)
                    #print('inputs.size is',inputs.shape)
                    yhat, h_state = model(inputs, h_state)
                    h_state = h_state.detach()
                    loss = L2_loss(yhat, targets)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    #for name in model.state_dict():
                    #    if ('weight_hh') in name:
                    #        print('weight',name,'less than 1')
                    #        model.state_dict()[name]=torch.div(model.state_dict()[name],torch.norm(model.state_dict()[name],p='fro'))
                    #        print(torch.norm(model.state_dict()[name], p='fro', dim=None, keepdim=False, out=None, dtype=None))
                print('yhat.shape is',yhat.shape)
                print('Now the loss=',loss)
            loss_res.append(float(loss.detach().numpy()))
            torch.save(model, save_path)
            print('Model has been saved')
            if  abs(loss0-loss)<0.0001:
               break
            else:
                loss0=loss

    def plot_time_series_model(self, model,gap):
        time=self.Dataset.time[0:len(self.Dataset.time):gap].reshape(-1,1)
        print(len(time))
        ori_data=self.Dataset.transform_back(self.Dataset.ori_data,'ori2back')
        #plt.show()
        x_0=self.Dataset.ori_data[0,:]
        path_all = []
        h_state=None
        for i in range(len(time)):
            x_0 = self.Dataset.ori_data[i:i+gap,:]
            x_t=torch.from_numpy(x_0.reshape(-1,gap,self.Dataset.dim))
            yhat,h_state = model(x_t, h_state)
            h_state = h_state.detach()
            yhat = yhat.detach().numpy()
            yhat=yhat.reshape(-1,self.Dataset.dim)
            #print('x_0 shape is',x_0.shape)
            #jacobian(torch.from_numpy(x_0),torch.from_numpy(yhat),gap)
            #path_all.append(list(self.Dataset.transform_back(yhat[0],'train2back')))
            path_all.append(yhat)
        for dim in range(45):
            print(dim)
            plt.plot(time, ori_data[:len(self.Dataset.ori_data)//10:gap, dim])
            plt.show()
            plt.plot(time,[i[dim] for i in path_all])
            plt.show()
        print('Over Now')
        return path_all


def load_model(save_path):
    model = torch.load(save_path)
    return model


def plot_train(Dataset,dim):
    time=Dataset.time[0:len(Dataset.time):gap].reshape(-1,1)
    print(len(time))
    train_data=Dataset.transform_back(Dataset.train_dl,'train2back')
    plt.plot(time, train_data[:len(time), dim])
    plt.show()

if __name__ == "__main__":
    steps,gap=140,10
    data_path='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    LC=LCrnn(DS,steps,gap)
    #save_path='C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/lcrnn_model_1.pkl'
    embedding_dim=DS.dim
    target_size=DS.dim
    num_layers = 3
    input_size=DS.dim
    hidden_size=DS.dim*3
    rnn = RNN(input_size, hidden_size, num_layers, target_size)
    LC.train_rnn(rnn,save_path)
    model = load_model(save_path)
    for name, param in model.named_parameters():
        print(name,param.shape)
    print('OK')
    for name in model.state_dict():
        print(name)
        print(model.state_dict()[name].shape)
    dim=20
    plot_train(DS,dim)
    print('OKL')

