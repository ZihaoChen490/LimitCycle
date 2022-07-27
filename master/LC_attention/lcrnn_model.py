# pytorch LSTM for regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch import Tensor,nn
from torch.nn import Linear, Sigmoid, Module, LayerNorm,ReLU
from torchsummary import summary
import torch, gensim
torch.manual_seed(3)
from sklearn.model_selection import train_test_split
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
#LSTM(sequence_len,batchsize,dim)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers=num_layers
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.h1 = Linear(in_features=hidden_size, out_features=hidden_size*8)
        self.act1=ReLU()
        self.LN1=LayerNorm(hidden_size*8,eps=1e-05,elementwise_affine=True)
        self.h2 = Linear(in_features=hidden_size*8, out_features=output_size*8)
        self.act2=Sigmoid()
        self.output_layer = Linear(in_features=output_size*8, out_features=output_size)


    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)

        #    if torch.norm(self.rnn.weight_hh_l[i])>1:
        #        self.rnn.weight_hh_l[i]=torch.div(self.rnn.weight_hh_l[i],torch.norm(self.rnn.weight_hh_l[i]))
         #       print(self.rnn.weight_hh_l[i])
        #        gamma = torch.ones(m.weight.data.shape[1])
        #        beta = torch.zeros(m.weight.data.shape[1])
        #        m.weight.data = self.simple_batch_norm_1d(m.weight.data, gamma, beta)
                #print(torch.norm(m.weight.data))
                #m.weight.data.normal_(0, 1)
        #    elif isinstance(m, nn.BatchNorm3d):
        #        m.weight.data.fill_(1)
        rnn_out, (h_n, h_c) = self.rnn(x, h_state)   # h_state是之前的隐层状态
        #h_state=LayerNorm(3,h_state, 1e-05,  True)
            #print(idx,self.rnn._flat_weights[idx])
        X=self.h1(rnn_out.view(rnn_out.shape[0],gap,hidden_size))
        X=self.act1(X)
        #print(X.shape)
        X=self.LN1(X)
        X=self.h2(X)
        #print(X.shape)
        X=self.act2(X)
        out=self.output_layer(X)
        #print(out.shape)
        return out, h_state       # torch.stack扩成[1, output_size, 1]

def cal_c(W):
    with torch.no_grad():
        res=torch.sqrt(1-torch.div(1,W))
        c=torch.atanh(res)-W*res
    return c

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

    def train_rnn(self,model,save_path,batch_size):
        gap=self.gap
        steps=self.steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001 ,weight_decay=0.001)
        data = torch.from_numpy(self.Dataset.ori_data)
        print('datashape is',data.shape)
        regularization_loss,t = 0,len(data)//steps
        loss0=0
        for epoch in range(self.epoch):
            for step in range(steps):
                loss_res = []
                train_epoch=data[step * t :(step+1) * t , :]
                print('.........................................')
                print('Now we strat to train epoch:',epoch,'step is',step)
                h_n=torch.zeros(3, batch_size, hidden_size)
                h_c=torch.zeros(3, batch_size, hidden_size)
                h_state = (h_n,h_c)  # 初始化隐藏层状态
                #h0 = model.init_hidden(DS.train_dl.shape[1])
                for i in range((t-gap)//batch_size):
                    #print(i)
                    model.zero_grad()
                    inputs = torch.cat([train_epoch[i*batch_size+j:i*batch_size+gap+j,:] for j in range(batch_size)]).reshape(batch_size,gap,self.Dataset.dim).to(torch.float32)
                    #print('data is',data[step*t+i,:])
                    #print('inputs is',inputs[0])
                    #targets=torch.cat([train_epoch[i * batch_size + j+1:i * batch_size + gap + j+1, :] for j in  range(batch_size)]).reshape(batch_size, gap, self.Dataset.dim).to(torch.float32)
                    targets=torch.cat([train_epoch[i * batch_size + j+1:i * batch_size + gap + j+1, :] for j in  range(batch_size)]).reshape(batch_size, gap, self.Dataset.dim).to(torch.float32)
                    #print('data targets is',data[step*t+i+1,:])
                    #print('targets inputs is',targets.shape)
                    yhat, (h_n, h_c) = model(inputs, h_state)
                    print(yhat[0,0,:])
                    h_n = h_n.detach()
                    h_c= h_c.detach()
                    h_state=(h_n,h_c)
                    loss = L2_loss(yhat, targets)
                    loss.backward(retain_graph=True)
                    loss_res.append(float(loss.detach().numpy()))
                    optimizer.step()
                print('yhat.shape is',yhat.shape)
                print('Now the loss=',sum(loss_res))
            torch.save(model, save_path)
            #torch.save(model.state_dict(), save_path)
            #model.eval()
            print('Model has been saved')
            if  abs(loss0-sum(loss_res))<0.0001:
               break
            else:
                loss0=sum(loss_res)

    def plot_time_series_model(self, model,gap):
        time=self.Dataset.time[0:len(self.Dataset.time):gap].reshape(-1,1)
        print(len(time))
        #ori_data=self.Dataset.transform_back(self.Dataset.ori_data,'ori2back')
        #plt.show()
        x_0=self.Dataset.ori_data[1000:1000+gap,:]
        path_all = []
        h_n=torch.zeros(3, 1, hidden_size)
        h_c=torch.zeros(3, 1, hidden_size)
        h_state=(h_n,h_c)
        for i in range(len(time)):
            x_t=torch.from_numpy(x_0.reshape(-1,gap,self.Dataset.dim))
            yhat, (h_n, h_c) = model(x_t, h_state)
            h_n = h_n.detach()
            h_c = h_c.detach()
            h_state = (h_n, h_c)
            #jacobian(torch.from_numpy(x_0),torch.from_numpy(yhat),gap)
            #path_all.append(list(self.Dataset.transform_back(yhat[0],'train2back')))
            yhat=yhat.squeeze()
            path_all.append(yhat[5])
 #       path_all=np.vastack()
        for dim in range(44):
            print(dim)
            plt.plot(time, self.Dataset.ori_data[:len(self.Dataset.ori_data)//10:gap, dim])
            plt.show()
            plt.plot(time,[i[dim] for i in path_all])
            plt.show()
        print('Over Now')
        return path_all


def load_model(save_path):
    model = torch.load(save_path)
    return model


def plot_train(Dataset):
    for dim in range(Dataset.dim):
        time=Dataset.time[0:len(Dataset.time):gap].reshape(-1,1)
        print(len(time))
        train_data=Dataset.train_dl#transform_back(Dataset.train_dl,'train2back')
        plt.plot(time, train_data[:len(time), dim])
        plt.show()


def plot_rnn(steps,test,predict,dim):
    print(len(test))
    print(len(predict))
    #print(len([print(i[:,10]) for i in test]))
    test_n=np.array(test).astype(np.float32)
    predict_n=np.array(predict).astype(np.float32)
    test_n[:, :, dim].reshape(-1, 1).tolist()
    predict_n[:, :, dim].reshape(-1, 1).tolist()
    print(test_n.shape)
    print(test_n[:,:,dim].reshape(-1,1).shape)
    print(len(test_n[:,:,dim].reshape(-1,1).tolist()))
    print(steps)
    test=test_n[:, :, dim].reshape(-1, 1).tolist()
    predict=predict_n[:, :, dim].reshape(-1, 1).tolist()
    time=np.array([i for i in range (steps)]).tolist()
    plt.plot(time[:1000], test[:1000], c='r')
    plt.show()
    plt.plot(time[:1000], predict[:1000] ,c='b')
    plt.show()
    plt.draw()
    plt.pause(0.05)
    plt.ioff()


if __name__ == "__main__":
    steps,gap=10,280
    data_path='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    LC=LCrnn(DS,steps,gap)
    #save_path='C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/lcrnn_model_x0.pkl'
    batch_size=64
    embedding_dim=DS.dim
    target_size=DS.dim
    num_layers = 3
    input_size=DS.dim
    hidden_size=DS.dim*3
    #plot_train(DS)
    model = RNN(input_size, hidden_size, num_layers, target_size)
    #print(model)
    model = load_model(save_path)
    #LC.train_rnn(model,save_path,batch_size)
    LC.plot_time_series_model(model,gap)
    for name, param in model.named_parameters():
        print(name,param.shape)
    print('OK')
    for name in model.state_dict():
        print(name)
        print(model.state_dict()[name].shape)
    #dim=20
    #print('OKL')

