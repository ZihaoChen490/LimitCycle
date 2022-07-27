from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_Utils import Dataset
from Plot_Utils import *
from Math_Utils import *
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA,ARIMA
from sklearn.metrics import mean_squared_error
def feature_root_detect(datasets,dim):
    temp = datasets.train_dl[:,dim]
    print(temp.ndim)
    t = adfuller(temp[::100])  # For ADF test
    output = pd.DataFrame(
        index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    #print(output)
    return output

def ARMA_rank(datasets,dim):
    data=datasets.train_dl[::100,dim]
    lag_acf = acf(data, nlags=20)
    lag_pacf = pacf(data, nlags=20, method='ols')
    print(lag_acf,lag_pacf)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    plot_acf(data, lags=100, ax=axes[0])
    plot_pacf(data, lags=100, ax=axes[1])
    #plt.show()


def ARMA_model(datasets,dim,order = (3, 3)):
    train = datasets.train_dl[::100,dim]
    test = datasets.ori_data[::10,dim]
    tempModel = ARMA(train, order).fit()
    delta = tempModel.fittedvalues - train  # Residual
    score = 1 - delta.var() / train.var()
    print('score is',score)
    # Scores is in [0,1], closer to 1 corresponding to better efficient
    predicts = tempModel.predict(301,300+len(test), dynamic=True)
    print(len(predicts))
    mse=mean_squared_error(test, predicts)
    comp = pd.DataFrame()
    comp['original'] = test
    comp['predict'] = predicts
    comp.plot()
    print(comp)
    #plt.show()
    return mse
def ARIMA_model(datasets,dim,oreder=(2, 1, 2)):
    data=datasets.train_dl[::100,dim]
    test=datasets.ori_data[::10,dim]
    model = ARIMA(data, order=oreder).fit()  
    delta = model.fittedvalues - data[1:]
    score = 1 - delta.var() / data[1:].var()
    print(score)
    start_index = 301
    end_index =300+len(test)
    forecast = model.predict(start=start_index, end=end_index)
    print(forecast)
    comp = pd.DataFrame()
    comp['original'] = test
    comp['predict'] = forecast
    comp.plot()
    print(comp)
    #plt.show()
    mse=mean_squared_error(test, forecast)
    return mse

if __name__ == "__main__":
    steps,gap=200,10
    data_path='../SNCRL_Dataset/CellCycle/'
    DS=Dataset(data_path,1)
    print('train_dl.shape',DS.train_dl.shape)
    print('test_dl.shape',DS.test_dl.shape)
    print('ori_data.shape',DS.ori_data.shape)
    print('len(Ds)',DS.ori_data)
    print('len(Ds)',DS.ori_data[:len(DS.ori_data),13:20].shape)
    mse_arma=[]
    mse_arima=[]
    try:
        for i in range(DS.dim):
            dim=i
        #output=feature_root_detect(DS,dim)
        #ARMA_rank(DS,dim)
            r1=ARMA_model(DS,dim,(3,3))
            mse_arma.append(r1)
            r2=ARIMA_model(DS, dim, (2, 1,2))
            mse_arima.append(r2)
    except:
        print(dim)
        print('ARMA MSE: %.3f, RMSE: %.3f' % (sum(mse_arma), sqrt(sum(mse_arma))))
        print('ARIMA MSE: %.3f, RMSE: %.3f' % (sum(mse_arima), sqrt(sum(mse_arima))))
