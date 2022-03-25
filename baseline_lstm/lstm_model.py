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



class LSTM(Module):
    def __init__(self,embedding_dim,batch_size,hidden_dim,voacb_size,target_size,num_layers):
        super(LSTM,self).__init__()
        self.embedding_dim=embedding_dim
        #self.embedding = torch.nn.Embedding(self.voacb_size, self.embedding_dim)
        self.lstm=torch.nn.LSTM( input_size = embedding_dim,
                                 hidden_size = hidden_dim,
                                 num_layers=num_layers,
                                 batch_first = True,)
        #print(self.embedding_dim)
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        #print('hidden',self.hidden_dim)
        self.voacb_size=voacb_size
        #print('voacb', self.voacb_size)
        self.target_size=target_size
        self.num_layers=num_layers
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=target_size)
        self.out = torch.nn.Linear(self.embedding_dim, self.target_size*(1+num_layers))
        self.hidden = (torch.autograd.Variable(torch.zeros(self.embedding_dim, batch_size, self.hidden_dim)),
                       torch.autograd.Variable(torch.zeros(self.embedding_dim, batch_size, self.hidden_dim)))
        self.act4 = Sigmoid()
        self.linear1 = Linear(self.hidden_dim, 14)
        kaiming_uniform_(self.linear1.weight)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.act5 = Sigmoid()
        self.linear2 = Linear(14, self.target_size)
        kaiming_uniform_(self.linear2.weight)
        # third hidden layer and output
        self.dropout2 = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,inputs,h_state):
        #embeddings_out=self.embedding((inputs))
        #print(self.embedding)
        #print(self.embedding_dim)
        #print('embedding_out res',embeddings_out.view(-1,1,self.embedding_dim).shape)
        #lstm_out,self.hidden=self.lstm(embeddings_out.view(-1,1,self.embedding_dim))
        lstm_out, self.hidden = self.lstm(inputs,h_state)
        lstm_out=self.act4(lstm_out)
        linear_out1 = self.linear1(lstm_out)#.view(-1,self.hidden_dim))
        dropout_out1 = self.dropout1(linear_out1)
        linear_out2 = self.linear2(dropout_out1)
        dropout_out2 = self.dropout2(linear_out2)
        out = []
        for time in range(dropout_out2.size(1)):
            every_time_out = dropout_out2[:, time, :]       # 相当于获取每个时间点上的输出，然后过输出层
            out.append(self.output_layer(every_time_out))
        #a,b,c=lstm_out.shape
        #print('abc is',lstm_out.shape)
        #print('lstm_out res',lstm_out.view(-1,c).shape)
        #lstm_out=lstm_out.view(-1, c)
        #lstm_out = self.out(lstm_out)
        #print(softmax_out.shape)
        return torch.stack(out, dim=1), self.hidden # torch.stack扩成[1, output_size, 1]

    def init_hidden(self,batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        return hidden





def train_rnn(train_dl,model,steps,gap,shape,save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/rnn_model_2.pkl'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, weight_decay=0.001)
    data = torch.from_numpy(train_dl)
    print('datashape is',data.shape)
    regularization_loss,t = 0,len(train_dl)//steps
    loss_res=[]
    #(10,)->(1,10,1)
    for epoch in range(15):
        for step in range(steps):
            print('.........................................')
            print('Now we strat to train epoch:',epoch,'step is',step)
            h_state = None  # 初始化隐藏层状态
            #h0 = model.init_hidden(DS.train_dl.shape[1])
            for i in range(t//gap-2):
                model.zero_grad()
                inputs = data[step*t+i*gap:step*t+i*gap+gap,:].reshape(-1,gap,shape[1])
                targets = data[step*t+i*gap+gap:step*t+(i+2)*gap,:].reshape(-1,gap,shape[1])
                #print(inputs)
                #print('inputs.size is',inputs.shape)
                yhat, h_state = model(inputs, h_state)
                print(yhat.shape)
                h_state = h_state.detach()
                loss = L2_loss(yhat, targets)
                loss.backward(retain_graph=True)
                optimizer.step()
            print('yhat.shape is',yhat.shape)
            print('Now the loss=',loss)
        loss_res.append(float(loss.detach().numpy()))
        if epoch>3 and (loss_res[epoch-1]-loss_res[epoch-2])<0.0001:
            try:
                torch.save(model, save_path)
                break
            except:
                print('Unsave model')
                break


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

def plot_time_series_model(Dataset,model,gap=10,dim=0):
    data = torch.from_numpy(Dataset.ori_data)
    steps=1
    t=len(data)//steps
    print('ori_data is', data.shape)
    mses=[]
    predictions, actuals = list(), list()
    for step in range(steps):
        print(step)
        h_state = None  # 初始化隐藏层状态
        for rd in range((t - 2 * gap) // gap):
            inputs = data[step * t + rd * gap:step * t + (rd + 1) * gap, :].reshape(-1, gap, DS.shape[1])
            yhat, h_state = model(inputs, h_state)
            h_state = h_state.detach()
            yhat = yhat.detach().numpy()
            actual = data[step * t + (rd + 1) * gap:step * t + (rd + 2) * gap, :].numpy()
            #        actual = actual.reshape((len(actual), 2))
            predictions.append(np.squeeze(yhat))
            actuals.append(actual)
            #print(actual.shape)
        plot_rnn((t - 2 * gap) , actuals, predictions, dim)
    dim1, dim2 = 10, 14
    predict_n = np.array(predictions).astype(np.float32)
    predictions=predict_n[:, :, :].reshape(-1, 44)
    actual_n = np.array(actuals).astype(np.float32)
    actuals=actual_n[:, :, :].reshape(-1, 44)
    plot_vector_fields(actuals,dim1,dim2)
    plot_vector_fields(predictions,dim1,dim2)
    return predictions


def train_lstm(train_dl, model,steps,gap,shape,save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/lstm_model_1.pkl'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.6,weight_decay=0.01)
    data = torch.from_numpy(train_dl)
    print('datashape is',data.shape)
    regularization_loss,t = 0,len(train_dl)//steps
    loss_res=[]
    for epoch in range(100):
        for step in range(steps-120):
            print('.........................................')
            print('Now we strat to train epoch:',epoch,'step is',step)
            h_state = None  # 初始化隐藏层状态
            #h0 = model.init_hidden(DS.train_dl.shape[1])
            for i in range(t//gap-2):
                model.zero_grad()
                inputs = data[step*t+i*gap:step*t+i*gap+gap,:].reshape(-1,gap,shape[1])
                targets = data[step*t+i*gap+gap:step*t+(i+2)*gap,:].reshape(-1,gap,shape[1])
                print(inputs)
                #print('inputs.size is',inputs.shape)
                yhat, h_state = model(inputs, h_state)
                h_state = h_state.detach()
                loss = L2_loss(yhat, targets)
                loss.backward(retain_graph=True)
                optimizer.step()
            print('yhat.shape is',yhat.shape)
            print('Now the loss=',loss)
            loss_res.append(float(loss.detach().numpy()))
        if epoch>3 and (loss_res[epoch-1]-loss_res[epoch-2])<0.0001:
            try:
                torch.save(model, save_path)
                break
            except:
                print('Unsave model')
                break

def evaluate_model_on_test(Dataset,model,steps=60,gap=1):
    data = torch.from_numpy(Dataset.test_dl)
    t=len(Dataset.test_dl)//steps
    mses=[]
    for step in range(steps):
        h_state = None  # 初始化隐藏层状态
        predictions, actuals = list(), list()
        for rd in range((t-2*gap)//gap):
            inputs = data[step * t + rd*gap:step * t + (rd+1)*gap, :].reshape(-1, gap, DS.shape[1])
            yhat, h_state = model(inputs, h_state)
            h_state = h_state.detach()
            yhat = yhat.detach().numpy()
            actual = data[step * t + (rd+1)* gap:step * t + (rd + 2) * gap, :].numpy()
#        actual = actual.reshape((len(actual), 2))
            predictions.append(np.squeeze(yhat))
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
        mses.append(mse)
    mse=sum(mses)/len(mses)
    return mse

def evaluate_model_ori_data(Dataset,model,gap=1):
    data = torch.from_numpy(Dataset.transform_back(Dataset.ori_data,'ori2train').astype(np.float32))
    steps=5
    t=len(Dataset.ori_data)//steps
    print('ori_data is', Dataset.ori_data.shape)
    mses=[]
    for step in range(steps):
        h_state = None  # 初始化隐藏层状态
        predictions, actuals = list(), list()
        for rd in range((t - 2 * gap) // gap):
            inputs = data[step * t + rd * gap:step * t + (rd + 1) * gap, :].reshape(-1, gap, DS.shape[1])
            yhat, h_state = model(inputs, h_state)
            h_state = h_state.detach()
            yhat = yhat.detach().numpy()
            actual = data[step * t + (rd + 1) * gap:step * t + (rd + 2) * gap, :].numpy()
            #        actual = actual.reshape((len(actual), 2))
            predictions.append(np.squeeze(yhat))
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
        mses.append(mse)
    mse = sum(mses) / len(mses)
    return mse


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    data = torch.from_numpy(test_dl)
    t=1
    h_state = None  # 初始化隐藏层状态
    for i in range(test_dl.shape[2]//t - 1):
        inputs = data[:, :, i * t].type(torch.LongTensor)
        targets = data[:, :, (i + 1) * t].reshape(test_dl.shape[1]*test_dl.shape[0], ).type(torch.LongTensor)
        yhat = model(inputs)
        yhat = yhat.detach().numpy().reshape(2,14,1)
        actual = targets.numpy()
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse
def get_all(path):
    dic=process_data(path)
    data=dic['data']
    cpath=dic['pathdata']
    samples_num=data.shape[0]
    input_dim=cpath.shape[1]
    Time=dic['Time'][0]
    train,test,cnt,dnt={},{},0,0
    for i in data[0:int(samples_num*0.7)]:
        train[cnt]=i
        cnt=cnt+1
    for j in data[int(samples_num*0.7):]:
        test[dnt]=j
        dnt=dnt+1
    total_steps=train[1].shape[0]
    return dic,train,test,Time,cpath
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    model.hidden = model.init_hidden(train_dl.shape[1])
    row = Tensor([row]).type(torch.LongTensor).reshape(2,14)
    yhat = model(row)
    yhat = yhat.detach().numpy().reshape(2,28)
    #print(yhat.shape)
    return yhat[:,:14]
def plot(num,step,model):
    ax1 = plt.subplot(1, 1, 1)
    path_all = []
    print('step:',step)
    for i in range(num):
        x_0=np.random.rand(2,14,1)
        print('第',i,'个x_0 is :',x_0.shape)
        pre_path=[]
        for i in range(step):
            pre_path.append(x_0.reshape(2,14))
            yhat = predict(x_0, model)
            x_0=yhat.reshape(2,14,1)
        path_all.append(pre_path)
    for path in path_all:
        for j in range(path[0].shape[1]):
            plt.plot([path[i][0][j]*1500 for i in range(step)],[path[i][1][j]*1500 for i in range(step)])
            plt.show()
  #  res=[cpath for cpath in path_all]
  #  path = 'C:/Aczh work place/3_paper/algonew/experiment-pend'
  #  np.savetxt(path+'/pre_path.csv',res , delimiter = ',')
def prepare_data(path):
    dic,train,test,Time,cpath=get_all(path)
    train_d=np.array([[[0 for i in range(train[0].shape[0])] for j in range(len(train))]for k in range(train[0].shape[1])])
    test_d=np.array([[[0 for i in range(test[0].shape[0])] for j in range(len(test))]for k in range(test[0].shape[1])])
    #elnum=train[0].shape[0]
    for i in range(len(train)):
        train_d[:,i,:] = train[i][:,:].reshape((train[0].shape[1],train[0].shape[0]))
    for i in range(len(test)):
        train_d[:,i,:] = test[i][:,:].reshape((train[0].shape[1],train[0].shape[0]))
    train_dl, test_dl = train_d[:2,:,:],test_d[:2,:,:]
    for i in range(train_dl.shape[0]):
        train_gd=train_dl[i,:,:]
        train_dl[i,:,:]=minmax_scale(train_gd).astype(np.float32)
        test_gd=test_dl[i,:,:]
        test_dl[i,:,:]=minmax_scale(test_gd).astype(np.float32)
    return train_dl,test_dl


def load_model(save_path):
    model = torch.load(save_path)
    return model


class lstm(torch.nn.Module):
    def __init__(self, output_size, hidden_size, embed_dim, sequence_length):
        super(lstm, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        #对应特征维度
        self.embed_dim = embed_dim
        self.dropout = 0.8
        #对应时间步长
        self.sequence_length = sequence_length
        #1层lstm
        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            )

        self.layer_size = self.layer_size
        self.attention_size = 30
        #（4，30）
        self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
        #（30）
        self.u_omega = Variable(torch.zeros(self.attention_size))
        #将隐层输入全连接
        self.label = nn.Linear(hidden_size * self.layer_size, output_size)
    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        # tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        # 张量相乘
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input):
        # input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)
        # print('input.size():',input.size())
        s, b, f = input.size()
        h_0 = Variable(torch.zeros(self.layer_size, b, self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, b, self.hidden_size))
        print('input.size(),h_0.size(),c_0.size()', input.size(), h_0.size(), c_0.size())
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits

def plot_train(Dataset,dim):
    time=Dataset.time[0:len(Dataset.time):gap].reshape(-1,1)
    print(len(time))
    train_data=Dataset.transform_back(Dataset.train_dl,'train2back')
    plt.plot(time, train_data[:len(time), dim])
    plt.show()
    #plot_vector_fields(train_dl,dim)
#model = torch.load('C:/Aczh work place/3_paper/algonew/experiment-pend/lstm_model_1.pkl')
#mse = evaluate_model(test_dl, model)
#print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
if __name__ == "__main__":
    num_model=8
    steps,gap=200,10
    data_path='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    DS.find_tracks(DS.train_dl,1e-3)
    print('train_dl.shape',DS.train_dl.shape)
    print('test_dl.shape',DS.test_dl.shape)
    print('ori_data.shape',DS.ori_data.shape)
    #print('len(Ds)',DS.ori_data)
    print('len(Ds)',DS.ori_data[:len(DS.ori_data),13:20].shape)
    print('ori_bound is',DS.ori_upbound,DS.ori_lowbound)
    #save_path='C:/Aczh work place/3_paper/SCNRL_Master/model/CellCycle/'
    save_path='C:/Aczh work place/3_paper/SCNRL_Master/baseline/rnn_model_2.pkl'
    embedding_dim=DS.dim
    hidden_dim=DS.dim*3
    voacb_size=1
    batch_size=1
    target_size=DS.dim
    num_layers = 2
    input_size=DS.dim
    hidden_size=DS.dim*3
    #model=LSTM(embedding_dim,batch_size,hidden_dim,voacb_size,target_size,num_layers)
    #rnn = RNN(input_size, hidden_size, num_layers, target_size)
    #train_rnn(DS.train_dl,rnn,steps,gap,DS.shape)
    #train_lstm(DS.train_dl, model,steps,gap,DS.shape)
    model = load_model(save_path)
    for name, param in model.named_parameters():
        print(name,param.shape)
    print('OK')
    for name in model.state_dict():
        print(name)
        print(model.state_dict()[name].shape)
    dim=20
    plot_train(DS,dim)
    predict = plot_time_series_model(DS,model,gap,dim)
    #predict=DS.transform_back(predict,'train2back')
    #print('OK')
    #np.savetxt('predict_ori.txt',predict)
    print('OKL')
    #print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    #mse = evaluate_model_on_test(DS,model,int(steps*0.3),gap)
    #print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    #mse = evaluate_model_ori_data(DS,model,gap)
    #print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
