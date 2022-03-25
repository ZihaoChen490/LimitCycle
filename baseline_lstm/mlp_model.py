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
data_sorce='C:/Aczh work place/3_paper/algo_new/data/'
argv='C:/Aczh work place/3_paper/algonew/experiment-pend/'
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

def train_mlp_model(train_dl, model,step,gap=1):
    torch.manual_seed(9)
    np.random.seed(5)
    #criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    data=torch.from_numpy(train_dl)
    regularization_loss,t = 0,int(len(train_dl) //step)
    data_all=[]
    for i in range(step):
        loss_res = []
        for epoch in range(300):
            print('.........................................')
            print('Now we strat to train the',i,'th epoch:',epoch,'with total step:', step)
            inputs = data[i*t,:]
            targets=(data[i*t+gap, :]-data[i*t,:])/gap
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = L2_loss(yhat, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('Now the loss=',loss)
            loss_res.append(loss.detach().numpy())
            if epoch>200 and (loss_res[epoch-1]-loss_res[epoch])<1e-7:
                try:
                    data_a=plot(2,100,gap,model)
                    data_all.append(data_a[-1][-1])
                    torch.save(model, 'C:/Aczh work place/3_paper/algonew/experiment-pend/mlp_new_8_model.pkl')
                    break
                except:
                    print('Unsave model')
                    break
    return data_all


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    data = torch.from_numpy(test_dl)
    t=1
    for i in range(len(test_dl)//t - 1):
        inputs = data[i*t, :]
        targets = data[(i+1)* t, :]
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
#        actual = actual.reshape((len(actual), 2))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse





def train_simple_mlp_model(train_dl, model,step=1,gap=1,shape=(1,20000),save_path='C:/A'):
    print(shape[0])
    torch.manual_seed(12)
    np.random.seed(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    data=torch.from_numpy(train_dl)
    regularization_loss,t = 0,int(len(train_dl) //shape[0])
    data_all=[]
    loss_res = []
    for epoch in range(300):
        # enumerate mini batches
        print('.........................................')
        #print('Now we strat to train the',1,'th epoch:',epoch,'with total step:', step)
        for batch in range(shape[0]//20):
            print('Now we strat to train the epoch:', epoch, 'with total step:', batch)
            start_p=batch*shape[1]
            for i in range(t-gap):
                inputs = data[start_p+gap*i, :]
                targets=np.gradient(data[start_p+gap*(i+1), :])
                optimizer.zero_grad()
                yhat = model(inputs)
                loss = L2_loss(targets,gap* yhat)
                loss.backward(retain_graph=True)
                optimizer.step()
        print('Now the loss=',loss)
        loss_res.append(loss.detach().numpy())
        if epoch>10 and (loss_res[epoch-1]-loss_res[epoch])<1e-6:
            torch.save(model, save_path)
            try:
               data_a=plot(50,1000,gap,model,2,True)
               data_all.append(data_a[-1][-1])
               break
            except:
                print('Unsave model')
                break
    return data_all


def predict(row, model):
    # convert row to data
    try:
        row = torch.tensor([row],requires_grad=True,dtype=torch.float32)
        #row = torch.Tensor([row])
    except:
        print('row is Tensor already')
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    if isinstance(yhat, tuple):
        ghat=np.array([yhat[0].detach().numpy(),yhat[1].detach().numpy()]).flatten().reshape(len(yhat[0]),2)
        return ghat
    yhat = yhat.detach().numpy()
    return yhat


def plot_time_series_model( Dataset,model,steps,gap,dim, flag=False):
    ax1 = plt.subplot(1, 1, 1)
    time=DS.time[0:len(DS.time):gap].reshape(-1,1)
    print(len(time))
    plt.plot(time, Dataset.ori_data[:len(Dataset.ori_data)//10, dim])
    plt.show()
    x_0=Dataset.ori_data[0,:]
    path_all = []
    for i in range(len(time)):
        path_all.append(list(x_0))
        yhat = predict(x_0, model)
        x_0 = yhat[0]
    print(type(path_all))
    print(len(path_all))
    plt.plot(time,[i[dim] for i in path_all])
    plt.show()
    return path_all

def evaluate_model_on_test(Dataset,model,step=60,gap=1):
    predictions, actuals = list(), list()
    data = torch.from_numpy(Dataset.test_dl)
    t=1
    print('test_dl is',len(Dataset.test_dl))
    for i in range((len(Dataset.test_dl)-gap)//10):
        inputs = data[i * t, :]
        targets = data[i * t + gap, :]
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
#        actual = actual.reshape((len(actual), 2))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse

def evaluate_model_ori_data(Dataset,path,gap=1):
    model=load_model(path)
    predictions, actuals = list(), list()
    data= torch.from_numpy(Dataset.ori_data)
    t = 1
    gap=1
    print('ori_data is', Dataset.ori_data.shape)
    for i in range(Dataset.ori_data.shape[1]):
        inputs = data[i * t, :]
        targets = data[i * t + gap, :]
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #actual = actual.reshape((len(actual), 2))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse


def load_model(save_path):
     model= torch.load(save_path)
     return model

if __name__ == "__main__":
    steps,gap=1,1
    data_path = 'C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS = Dataset(data_path, 1)
    #print('train_dl.shape',train_dl.shape)
    #plt.plot(train_dl[:,0],train_dl[:,1])
    #plt.show()
    #train_d=np.random.rand(train_dl.shape[0],train_dl.shape[1])
    #for i in range(train_dl.shape[1]):
    #    train_d[:,i]=wavelet_denoising(train_dl[:,i])
    #plt.plot(train_d[:,0],train_d[:,1])
    #val_dl, val_test_dl, max_min2 = prepare_data(data_path, 2)
    model = MLP(DS.dim)
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/model/mlp_model_0'
    #data_all = train_simple_mlp_model(DS.train_dl, model, steps, gap,DS.shape,save_path)
    dim=10
    plot_time_series_model(DS,model,steps,gap,dim, flag=False)
    model = load_model(save_path)
    mse = evaluate_model_on_test(DS,model,steps,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    mse = evaluate_model_ori_data(DS,save_path,gap)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))