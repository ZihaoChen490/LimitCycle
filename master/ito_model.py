# pytorch mlp for regression
import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack, sqrt
import pandas as pd
from torch.autograd import Variable
import torch, time, sys,math
from sklearn.metrics import mean_squared_error
from torch import Tensor
from torch.nn import Linear,Sigmoid,ReLU, Module
from torch.optim import SGD
from torch.nn import MSELoss,CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
from sklearn.preprocessing import normalize

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 60)
        kaiming_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(60, 30)
        kaiming_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(30, 10)
        kaiming_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        # third hidden layer and output
        self.hidden4 = Linear(10, n_inputs)
        kaiming_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
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
        return X

class SCNRL:
    def __init__(self,Dataset,steps,gap):
        self.Dataset=Dataset
        self.steps=steps
        self.gap=gap

    def train_sto_mlp_model(self,train_dl,model,steps=200,gap=10):
        print(train_dl.shape)
        data_all = []
        loss_res = []
        num_model=len(model)
        NewIndex,TargetSet=ReIndex(train_dl[:len(train_dl)//10],gap,num_model)
        torch.manual_seed(9)
        np.random.seed(5)
        lam=0.5
        optimizer = [torch.optim.Adam(model[i].parameters(), lr=0.000001, weight_decay=1e-5) for i in range(num_model)]
        data=torch.from_numpy(train_dl)
        regularization_loss,t = 0,len(train_dl)//steps
        for epoch in range(8):
            #for step in range(steps//num_model):
            for step in range(steps-180):
                print('......................................................................')
                print('Now we strat to train the',step,'th step in epoch:', epoch, 'with total step:', steps)
                for rd in range(t-gap):
                    inputs=[data[(step)*t+rd,:] for i in range(num_model)]
                    #NewIndex, TargetSet = ReIndex(train_dl, gap, num_model,(step)*t+rd)
                    targets=[torch.from_numpy(TargetSet[NewIndex[(step)*t+rd][i],:]) for i in range(num_model)]
                    #inputs = [data[(step*num_model+i)*t+rd,:] for i in range(num_model)]
                    #targets=[data[(step*num_model+i)*t+rd+gap,:] for i in range(num_model)]
                    [optimizer[i].zero_grad() for i in range(num_model)]
                    yhat = [model[i](inputs[i]) for i in range(num_model)]
                    loss = [abs(L2_loss(yhat[i], targets[i]) -lam * torch.norm(yhat[i] - sum(yhat) /len(model))) for i in range(len(model))]
                    #loss = [(L2_loss(yhat[i], targets[i])) for i in range(num_model)]
                    [loss[i].backward(retain_graph=True) for i in range(num_model)]
                    [optimizer[i].step() for i in range(num_model)]
                print('Now the loss=',loss)
            loss_res.append([loss[i].detach().numpy() for i in range(len(model))])
            if epoch>4 and (sum([(loss_res[epoch-1][i]-loss_res[epoch][i]) for i in range(len(model))]))<1e-10:
                break
        for i in range(len(model)):
            torch.save(model[i], 'C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/model_noise_' + str(i) + '.pkl')
        try:
            data_a=self.plot_model(1000,model,True)
            data_all.append(data_a[-1][-1])
        except:
            print('Unsave model')
        return data_all
    def plot_train(self,train_dl):
        plt.plot(train_dl[:,0],train_dl[:,1])
        print(train_dl[:,0])
        plt.show()
        plot_vector_fields(train_dl)


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
        return mse
    def load_model(self,save_path):
        model = [MLP(DS.dim) for i in range(num_model)]
        for i in range(num_model):
            model[i] = torch.load(save_path+'model_noise_' + str(i) + '.pkl')
        return model

    def evaluate_model_ori_data(self, path,gap=1):
        model=self.load_model(path)
        predictions, actuals = list(), list()
        data= torch.from_numpy(self.Dataset.ori_data)
        t = 1
        gap=1
        print('ori_data is', self.Dataset.ori_data.shape)
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
def fx(a,b):
    alpha=0.001
    return (a+alpha*b)/(1+alpha)

dn=np.random.randn(1)+3
for i in range(100000):
    n=np.random.randn(1)+3
    dn=fx(dn,n)

print(dn)

if __name__ == "__main__":
    num_model=8
    steps,gap=200,10
    data_path='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    print('train_dl.shape',DS.train_dl.shape)
    print('test_dl.shape',DS.test_dl.shape)
    print('ori_data.shape',DS.ori_data.shape)
    plt.plot(DS.train_dl[:len(DS.ori_data)//1,0],DS.train_dl[:len(DS.ori_data)//1,1])
    plt.show()
    plt.plot(DS.train_dl[:len(DS.train_dl)//10,0],DS.train_dl[:len(DS.train_dl)//10,1])
    plt.show()
    SC=SCNRL(DS,steps,gap)
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    model = SC.load_model(save_path)
    mse = SC.evaluate_model_on_test(model,steps,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    mse = SC.evaluate_model_ori_data(save_path,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    #data_all=SC.plot_model(1000,model,True)
    #get_scalar_fields(data_all, model)
    #GET_VECTOR_FIELDS(data_all, base_model,hnn_model, save_path)