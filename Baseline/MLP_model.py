# pytorch mlp for regression
import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack, sqrt
import pandas as pd
import networkx as nx
from torch.autograd import Variable
import torch, time, sys,math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_tweedie_deviance
from torch import Tensor,nn
from torch.nn import Linear,Sigmoid,ReLU, Module
from torch.optim import SGD
from torch.nn import MSELoss,CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
from sklearn.preprocessing import normalize
#data_sorce='C:/Aczh work place/3_paper/algo_new/data/'
#argv='C:/Aczh work place/3_paper/algonew/experiment-pend/'
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, n_inputs)
        kaiming_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(n_inputs, n_inputs*4)
        kaiming_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(n_inputs*4, n_inputs*8)
        kaiming_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        # third hidden layer and output
        self.hidden4 = Linear(n_inputs*8, n_inputs)
        kaiming_uniform_(self.hidden4.weight)
        self.act4 = ReLU()
        self.hidden5 = Linear(n_inputs, n_inputs)
        self.act5 = ReLU()
    # forward propagate input
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X=self.act3(X)
        X = self.hidden4(X)
        X=self.act4(X)
        X = self.hidden5(X)
        X = self.act5(X)
        return X

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)
        rnn_out, h_state = self.rnn(x, h_state)   # h_state是之前的隐层状态
        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]       # 相当于获取每个时间点上的输出，然后过输出层
            out.append(self.output_layer(every_time_out))
        return torch.stack(out, dim=1), h_state       # torch.stack扩成[1, output_size, 1]


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N*C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x N(*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check  B*N*N
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class SCNRL:
    def __init__(self,Dataset,steps,gap):
        self.Dataset=Dataset
        self.steps=steps
        self.gap=gap
        self.lam=0.5
        self.lr=0.0001
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
        model = [MLP(DS.dim+1) for i in range(num_model)]
        train_dl=self.Dataset.train_dl
        time=np.vstack([self.Dataset.time[0:len(self.Dataset.time)] for i in range(self.Dataset.train_dl.shape[0]//self.Dataset.time.shape[0])])
        print(train_dl.shape)
        print(time.shape)
        print(time)
        data_all,loss_res = [], []
        torch.manual_seed(8)
        np.random.seed(5)
        optimizer = [torch.optim.Adam(model[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) for i in range(num_model)]
        data=torch.from_numpy(np.hstack([time,train_dl]).astype(np.float32))
        print(data.shape)
        regularization_loss,t = 0,len(train_dl)//steps
        print('each length of the dataset is',t)
        for epoch in range(self.epoch):
            #for step in range(steps//num_model):
            for step in range(steps-70):
                print('......................................................................')
                print('Now we strat to train the',step,'th step in epoch:', epoch, 'with total num||step:', t-gap,'||',steps-70)
                for rd in range(t-gap):
                    inputs=[data[(step)*t+rd,:] for i in range(num_model)]
                    #NewIndex, TargetSet = ReIndex(train_dl, gap, num_model,(step)*t+rd)
                    targets=[torch.from_numpy(np.hstack([time[int(self.NewIndex[(step)*t+rd][i]+gap)],self.TargetSet[int(self.NewIndex[(step)*t+rd][i]),:]])) for i in range(num_model)]
                    #inputs = [data[(step*num_model+i)*t+rd,:] for i in range(num_model)]
                    #targets=[data[(step*num_model+i)*t+rd+gap,:] for i in range(num_model)]
                    [optimizer[i].zero_grad() for i in range(num_model)]
                    yhat = [model[i](inputs[i]) for i in range(num_model)]
                    loss = [abs(L1_loss(yhat[i], targets[i]) -self.lam * torch.norm(yhat[i] - sum(yhat) /len(model))) for i in range(len(model))]
                    #loss = [(L2_loss(yhat[i], targets[i])) for i in range(num_model)]
                    [loss[i].backward(retain_graph=True) for i in range(num_model)]
                    [optimizer[i].step() for i in range(num_model)]
                print('Now the loss=',loss)
            loss_res.append([loss[i].detach().numpy() for i in range(len(model))])
            for i in range(len(model)):
                torch.save(model[i],'C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/model_ito_l1_' + str(i) + '.pkl')
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
        time=self.Dataset.time[0:len(self.Dataset.time):gap]
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
        time=DS.time[0:len(DS.time):gap].reshape(-1,1)
        print(len(time))
        ori_data=self.Dataset.transform_back(self.Dataset.ori_data,'ori2back')
        plt.plot(time, ori_data[:len(self.Dataset.ori_data)//10, dim])
        plt.show()
        x_0=self.Dataset.ori_data[0,:]
        path_all = []
        for i in range(len(time)):
            path_all.append(list(self.Dataset.transform_back(x_0,'train2back')))
            yhat = predict(x_0, model)
            x_0 = yhat[0]
        plt.plot(time,[i[dim] for i in path_all])
        plt.show()
        return path_all


if __name__ == "__main__":
    num_model=4
    data_path='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    steps,gap=140,10
    print('train_dl.shape',DS.train_dl.shape)
    print('test_dl.shape',DS.test_dl.shape)
    print('ori_data.shape',DS.ori_data.shape)
    print('len(Ds)',DS.ori_data)
    print('len(Ds)',DS.ori_data[:len(DS.ori_data),13:20].shape)
    SC=SCNRL(DS,steps,gap)
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    SC.train_sto_mlp_model(num_model,steps,gap)
    #model = SC.load_model(save_path)
    dim=1
    #data_all=SC.plot_model(1000,model,True)
    SC.plot_time_series_model(model,steps,gap,dim)
    mse = SC.evaluate_model_on_test(model,steps,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    mse = SC.evaluate_model_ori_data(save_path,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    #get_scalar_fields(data_all, model)
    #GET_VECTOR_FIELDS(data_all, base_model,hnn_model, save_path)
    train_dl = SC.Dataset.train_dl
    data = train_dl
    regularization_loss, t = 0, len(train_dl) // steps
    print('each length of the dataset is', t)
    for epoch in range(SC.epoch):
        # for step in range(steps//num_model):
        for step in range(steps - 70):
            print('......................................................................')
            print('Now we strat to train the', step, 'th step in epoch:', epoch, 'with total step:', steps)
            for rd in range(t - gap):
                if rd % 5 == 0:
                    inputs = data[(step) * t + rd, :]
                    in_point = np.vstack([data[int(SC.NewIndex[(step) * t + rd][i]), :] for i in range(20)])
                    ori_targets = data[(step) * t + rd + gap, :]
                    targets = np.vstack([SC.TargetSet[int(SC.NewIndex[(step) * t + rd][i]), :] for i in range(20)])
                    plt.scatter(in_point[:, 10], in_point[:, 14], c='y')
                    plt.scatter(targets[:, 10], targets[:, 14], c='g')
                    plt.scatter(inputs[10], inputs[14], c='r')
                    plt.scatter(ori_targets[10], ori_targets[14], c='m')
                    plt.show()