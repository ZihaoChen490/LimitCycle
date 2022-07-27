import numpy as np
from numpy import argpartition
import pandas as pd
#np.set_printoptions(threshold=np.inf)
import multiprocessing as mp
import time,heapq,datetime,seaborn,numba,os
import scipy.io as scio
import hdf5storage as hdf5
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize,minmax_scale,scale
from tsmoothie.utils_func import sim_seasonal_data
from tsmoothie.smoother import DecomposeSmoother
import matplotlib.pylab as pyl
from pywt import wavedec
from scipy import signal
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.signal as signal
from Plot_Utils import *
from Math_Utils import *

class Dataset:
    def __init__(self,path,index):
        self.path=path
        self.list_name=[]
        self.mat_name=[]
        self.train_dl,self.test_dl,self.max_min1,self.ori_data=self.prepare_data(self.path,index)
        print('self.train_dl',self.train_dl.shape)

    def prepare_data(self,path, index):
        dic, train, test, self.time, ori, self.shape = self.get_all(path, index)
        self.dim=self.shape[1]
        self.time=self.time.reshape(-1,1)
        self.time=minmax_scale(self.time)
        train_d = np.array([[0 for i in range(train[0].shape[0])] for j in range(train[0].shape[1] * len(train))]).astype(np.float32)
        test_d = np.array([[0 for i in range(test[0].shape[0])] for j in range(test[0].shape[1] * len(test))]).astype(np.float32)
        ori_d = np.array([[0 for i in range(ori[0].shape[0])] for j in range(ori[0].shape[1] * len(ori))]).astype(np.float32)
        elnum = train[0].shape[1]
        print(train_d.shape,test_d.shape,ori_d.shape)
        for i in range(len(train)):
            train_d[i * elnum:(i + 1) * elnum, :] = train[i].T
        for i in range(len(test)):
            test_d[i * elnum:(i + 1) * elnum, :] = test[i].T
        for i in range(len(ori)):
            ori_d[i * elnum:(i + 1) * elnum, :] = ori[i].T
        self.train_upbound=self.get_bound(train_d)
        self.train_lowbound=np.min(train_d,axis=0).astype(np.float32)
        self.ori_upbound=np.max(ori_d,axis=0).astype(np.float32)
        self.ori_lowbound=np.min(ori_d,axis=0).astype(np.float32)
        #print(self.train_lowbound.shape)
        max_ori, min_ori = [0 for i in range(train[0].shape[0])], [self.train_upbound for i in range(train[0].shape[0])]
        max_now, min_now = [0 for i in range(train[0].shape[0])], [self.train_upbound for i in range(train[0].shape[0])]
        #print('max_ori',max_ori)
        for i in range(self.dim):
            train_d_row=train_d[:,i]
            train_d_row[train_d_row>self.train_upbound[i]]=self.train_upbound[i]
            train_d[:,i]=train_d_row
            test_d_row = test_d[:, i]
            test_d_row[test_d_row > self.train_upbound[i]] = self.train_upbound[i]
            test_d_row[test_d_row < self.train_lowbound[i]] = self.train_lowbound[i]
            test_d[:, i] = test_d_row
        # timeseries, fwbest=fftTransfer(train_d[...,:2])
        # print(timeseries)
        # print(fwbest)
        #plt.scatter(ori_d[:len(ori_d) // 1, 13], ori_d[:len(ori_d)  // 1, 20])
        #plt.show()
        #together = np.vstack((train_d, test_d, ori_d)).astype(np.float32)
        #train_dl = atan(train_d)*2/pi
        #test_dl = atan(test_d)*2/pi
        #ori_dl = atan(ori_d)*2/pi
        #minmaxtogether = minmax_scale(together[..., :self.dim]).astype(np.float32)
        #train_dl = minmaxtogether[:len(train_d), :]
        #test_dl = minmaxtogether[len(train_d):len(train_d) + len(test_d), :]
        #ori_dl = minmaxtogether[len(train_d) + len(test_d):len(minmaxtogether), :]
        train_dl, test_dl,ori_dl = minmax_scale(train_d[..., :self.dim]).astype(np.float32), minmax_scale(test_d[..., :self.dim]).astype(
            np.float32),minmax_scale(ori_d[..., :self.dim]).astype(np.float32)

        #plt.scatter(train_dl[:len(train_dl)  // 1, 13], train_dl[:len(train_dl)  // 1, 20])
        #plt.show()
        for i in range(self.dim):
            max_now[i] = train_dl[:, i].max()
            min_now[i] = train_dl[:, i].min()
        return train_dl, test_dl, [np.array(max_ori), np.array(max_now), np.array(min_ori),
                                   np.array(min_now)],ori_dl
    def get_bound(self,datasets):
        upbound= np.array([0 for i in range(datasets.shape[1])]).astype(np.float32)
        for i in range(datasets.shape[1]):
            dataset=datasets[:,i].astype(np.float32)
            where_are_inf = np.isinf(dataset)
            dataset[where_are_inf] = dataset.min()
            res = dataset.max()
            #print('res is',res)
            #print(len(dataset))
            cnt=0
            while res>0:
                res=res//10
                cnt+=1
            while len(dataset[dataset>pow(10,cnt-1)])/(len(dataset))<0.01:
                #print(len(dataset[dataset > pow(10, cnt-1)]))
                dataset[dataset > pow(10, cnt-1)]=pow(10,cnt-2)
                cnt=cnt-1
            gt=10
            if cnt>0:
                for j in range(10,0,-1):
                    if len(dataset[dataset > pow(10,cnt-2)*j]) / (len(dataset)) > 0.05:
                        gt=j
                        break
            dataset[where_are_inf]=pow(10,cnt)*gt
            upbound[i]=pow(10,cnt)*gt
        #print('upbound is', upbound)
        return upbound

    def get_all(self,path, index,split_num=0.7):
        dic = self.load_data(path,index)
        data = dic['data']
        # print('data.shape',data[0].shape)
        ori_data = dic['pathdata']
        samples_num = data.shape[0]
        time = dic['time'][0]
        train, test,ori,cnt, dnt,ent = {},{}, {}, 0, 0,0
        for i in data[0:int(samples_num*split_num)]:
            train[cnt] = i
            cnt = cnt + 1
        for j in data[int(samples_num * split_num):]:
            test[dnt] = j
            dnt = dnt + 1
        for k in ori_data[0:ori_data.shape[0]]:
            ori[ent] = k
            ent = ent + 1
        return dic, train, test, time, ori, (data.shape[0], data[0].shape[0], data[0].shape[1])
    def load_data(self,path,index):
        listdir(path,self.list_name)
        lis=[0 for i in range(3)]
        for i in self.list_name:
            res=i.split('_')
            if res[-1]=='v6.mat':
                if 'D=0' in res and 'YY' in res:
                    lis[0]=i
                if str(index) in res and 'YY' in res:
                    lis[1]=i
                if str(index) in res and 'TT' in res:
                    lis[2]=i
        dic=self.process_data(lis)
        return dic

    def process_data(self,lis):
        name = ['pathdata', 'data', 'time']
        dic = dict({})
        for i in range(len(lis)):
            hg = load_mat(lis[i])
            if len(hg) > 1:
                tg = hg
                dic[name[i]] = tg
            else:
                tg = hg[0]
                dic[name[i]] = tg
        print('successly load', name[i])
        return dic

    def primary_data(self,step=10, gap=1, data_path='C:/Aczh work place/3_paper/algo_new/data/'):
        # fig1, ax1 = plt.subplots()
        Cycle_nos = [[0.11307394, 0.07597382], [0.07850351, 0.04150944], [0.10505617, 0.06679342],
                     [0.11571446, 0.079058774], [0.08464647, 0.0497617], [0.08367057, 0.044767097],
                     [0.11733942, 0.078370884], [0.07839489, 0.042922024], [0.2918319, 0.34125295],
                     [0.12101538, 0.08073301], [0.122714326, 0.08166276], [0.092909835, 0.05384412],
                     [0.12800306, 0.09206281], [0.13034421, 0.08995868], [0.09400665, 0.059143055],
                     [0.11177004, 0.07367713], [0.17821412, 0.15235686], [0.17299022, 0.14054206],
                     [0.121608645, 0.08303627], [0.12327456, 0.08418817], [0.1343672, 0.09648457],
                     [0.110850334, 0.07265582], [0.117687784, 0.079576045], [0.11105673, 0.07321814],
                     [0.10991231, 0.07000575], [0.13567936, 0.09603847], [0.12058076, 0.084363654],
                     [0.09591067, 0.058704406], [0.12387468, 0.08607804], [0.08925655, 0.051861666],
                     [0.1847061, 0.15064327], [0.119924866, 0.08374668], [0.24202512, 0.27488664],
                     [0.10983366, 0.072221965], [0.11692483, 0.07746354], [0.14326113, 0.10386492],
                     [0.10959328, 0.07138749], [0.14483738, 0.10656789], [0.13136466, 0.09423474],
                     [0.122188956, 0.081091784], [0.3636335, 0.17975372]]
        Cycle_nos = np.array(Cycle_nos)
        raw_data, test_dl, max_min2, shape2 = self.prepare_data(data_path, 4)
        raw_data = change_range(raw_data, max_min2[2][:2], max_min2[0][:2])
        # ax1.scatter(raw_data[:, 0], raw_data[:, 1], c='g', marker='o', s=0.1)
        train_dl, test_dl, max_min1, shape1 = self.prepare_data(data_path, 0)
        Cycle_std = change_range(train_dl, max_min1[2][:2], max_min1[0][:2])
        # ax1.scatter(Cycle_std[:, 0], Cycle_std[:, 1], c='b', marker='o', s=8)
        Cycle_nos = change_range(Cycle_nos, (max_min2[0][:2] - max_min2[2][:2]) * Cycle_nos.min() + max_min2[2][:2],
                                 (max_min2[0][:2] - max_min2[2][:2]) * Cycle_nos.max() + max_min2[2][:2])
        # ax1.scatter(Cycle_nos[:, 0], Cycle_nos[:, 1], c='r', marker='o', s=8)
        peak_point = np.array([0.50691175, 0.4524363]) * (max_min2[0][:2] - max_min2[2][:2]) + max_min2[2][:2]
        return Cycle_nos, Cycle_std, raw_data, peak_point, shape2

    def transform_back(self,data,type):
        if type=='train':
            train_data=self.train_lowbound+data*(self.train_upbound-self.train_lowbound)
            return train_data
        if type=='ori':
            ori_data=self.ori_lowbound+data*(self.ori_upbound-self.ori_lowbound)
            return ori_data
        if type=='ori2train':
            ori_data = self.ori_lowbound + data * (self.ori_upbound - self.ori_lowbound)
            train_data = (ori_data-self.train_lowbound ) / (self.train_upbound - self.train_lowbound)
            return train_data
        if type=='train2ori':
            train_data=self.train_lowbound+data*(self.train_upbound-self.train_lowbound)
            ori_data = (train_data-self.ori_lowbound ) / (self.ori_upbound - self.ori_lowbound)
            return ori_data
        if type=='ori2back':
            ori_data=data*(self.ori_upbound-self.ori_lowbound)+self.ori_lowbound
            return ori_data
        if type=='train2back':
            train_data=data*(self.train_upbound-self.train_lowbound)+self.train_lowbound
            return train_data

    def find_tracks(self,IS,threshold=1e-2):
        NewIndex=[]
        bat=200
        m_IS=IS
        for i in range(len(IS)//bat-1,-1,-1):
            mark = np.ones(len(m_IS))
            print(i)
            res = np.vstack([[np.linalg.norm(m_IS - m_IS[i*bat+j],ord=1,axis=1) < threshold] for j in range(bat)])
            print(res.shape)
    #        res = np.vstack([[np.sum(np.abs(m_IS - m_IS[i*bat+j]), axis=1) < threshold] for j in range(bat)])
            gres=np.dot(res,mark)-np.ones(bat)
            print('what gres.shape is',gres.shape)
            if np.sum(gres)>1:
                print('now is k',i)
                Idx = np.array([k for k in range(i * bat, (i + 1) * bat)])
                NewIndex.append(np.where(np.sum((IS - IS[Idx[gres]]) ** 2, axis=1) < threshold))
            else:
                m_IS=np.delete(m_IS, np.s_[i*bat:(i+1)*bat:1],axis=0)
                print('m_IS alteration is',m_IS.shape)
            if i % 2000 == 0:
                print('Now the NewIndex forward to', i)
        return NewIndex


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)





def argv_isnparray(argv):
    if argv.ndim == 2:
        print('数组长度', len(argv[0]),'数组宽度', len(argv))
    elif argv.ndim == 3:
        print('数组长度', len(argv[0][0]),'数组宽度', len(argv[0]),'数组高度', len(argv))
    print('数组形状', argv.shape,'最大值', np.max(argv),'最小值', np.min(argv))
def argv_isdict(argv):
    print('字典的元素个数', len(argv))
    if isinstance(argv,dict):
        print('字典的键包括', argv.keys())
        cnt=0
        for i in argv.keys():
            if isinstance(argv[i],dict):
                argv_isdict(argv[i].values())
            elif isinstance(argv[i],np.ndarray):
                cnt=cnt+1
                print('字典键的键',i,'是字典第',str(cnt),'个数组')
                argv_isnparray(argv[i])
            else:
                print('字典的键',i ,'的键值是', argv[i])
    else:
        print('元素的值', argv)
def get_mes(argv,str=None):
    if str:
        print('输入的是',str,type(argv))
    else:
        print('输入的是一个', type(argv))
    if isinstance(argv, list):
        print('列表长度', len(argv[0]),'列表宽度:', len(argv))
    elif isinstance(argv, np.ndarray):
        return argv_isnparray(argv)
    elif isinstance(argv,dict):
        return argv_isdict(argv)
    else:
        print('既不是列表也不是数组也不是字典')
def load_mat(argv):
    try:
        argv1 = scio.loadmat(argv)
    except:
        argv1 = hdf5.loadmat(argv)
    hg=list(argv1.values())
    tg=hg[-1]
    return tg

def smooth(data):
    is_cycle,cycles=fftTransfer(data,10,0.2)
    print(is_cycle)
    print(cycles)
    smoother = DecomposeSmoother(smooth_type='lowess', periods=cycles[0],smooth_fraction=0.3)
    smoother.smooth(data)
    low, up = smoother.get_intervals('sigma_interval')
    print('aweaweae',low,up)
    # plot the smoothed timeseries with intervals
    plt.figure(figsize=(5, 4))
    for i in range(data.shape[1]):
        plt.subplot(1, 3, i + 1)
        plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
        plt.plot(smoother.data[i], '.k')
        plt.title(f"timeseries {i + 1}")
        plt.xlabel('time')
        plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)
    #plt.show()
    print(smoother.smooth_data)
    return smoother.smooth_data
def ReIndex(dataset,gap,num_model):
    InputSet,TargetSet=Id2Id(dataset,gap)
    IS=np.array(InputSet)
    print(InputSet.shape,TargetSet.shape)
    #NewIndex=[[0 for i in range(num_model)] for j in range(len(InputSet))]
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    num_cores=num_cores*2
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(FindNearPoint, args=(IS,num_model,num_cores, param)) for param in range(num_cores)]
    NewIndex = sum([p.get() for p in results]).tolist()
    #print(len(results[0]))
    #for i in range(len(InputSet)):
    #for i in range(len(InputSet)):
        #na=np.sum((IS-IS[i])**2,axis=1,dtype='float16')
    #print('na,ok')
    #idx=[np.argpartition(na[i], num_model+1) for i in range(len(InputSet))]
    print('NewIndex is',NewIndex.shape)
    return NewIndex,TargetSet

#@numba.jit
def FindNearPoint(IS,num_model,num_cores,param):
    NewIndex=[[0] for j in range(len(IS))]
    IS=minmax_scale(IS[..., :IS.shape[1]]).astype(np.float32)
    WeiIndex=np.array([[0 for i in range(num_model)] for j in range(len(IS))])
    for i in range(len(IS)//num_cores*param,len(IS)//num_cores*(param+1)):
        #NewIndex[i] = np.argsort(np.sum((IS-IS[i])**2,axis=1))[0:num_model]
        NewIndex[i]=np.where(np.sum((IS-IS[i])**2,axis=1)<1e-2)
        #NewIndex[i]=argpartition(np.sum((IS-IS[i])**2,axis=1), num_model)[:num_model]
        #idx=np.delete(idx,np.where(idx==i),axis=0)
        #NewIndex[i]=idx.tolist()
        print(NewIndex[i])
        if i%2000==0:
            print('Now the NewIndex forward to',i)
    return NewIndex


cal_threshold
def largest_within_delta(X, k, delta):
    right_idx = X.searchsorted(k,'right')-1
    if (k - X[right_idx]) <= delta:
        return right_idx
    else:
        return None


def get_range(data_all):
    xmin, xmax = 0, 1
    dxmin, dxmax = 0, 1
    for data in data_all:
        for idata in data:
            xmin, xmax = min(idata[0], xmin), max(idata[0], xmax)
            dxmin, dxmax = min(idata[1], dxmin), max(idata[1], dxmax)
    return (xmin, xmax, dxmin, dxmax)

def change_range(values, vmin=0, vmax=1):
    start_zero = values - np.min(values)
    return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin

def Id2Id(Dataset,Gap):
    Input=Dataset[0:len(Dataset)-Gap,:]
    Target=Dataset[Gap:len(Dataset),:]
    return Input,Target







def save_file(file,path):
    if isinstance(file,dict):
        save_dict_to_file(file, path)
    elif isinstance(file,list):
        fileObject = open(path+'.txt', 'w')
        for ip in file:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
def save_dict_to_file(_dict, filepath):
    try:
        with open(filepath, 'w') as dict_file:
            for (key,value) in _dict.items():
                dict_file.write('%s:%s\n' % (key, value))
    except IOError as ioerr:
        print("文件 %s 无法创建" % (filepath))
def data_classification_tree(data):
    boolen_is_stable = is_stable(data)
    boolen_is_periodicity = is_periodicity(data)
    if boolen_is_stable:
        # 平稳数据
        return 0
    else:
        if boolen_is_periodicity[0]:
            return 1
        else:
            return 2
def is_periodicity(data, show_pic=False):
    f, Pxx_den = signal.periodogram(data[:,0])
    if show_pic is True:
        pyl.semilogy(1/f, Pxx_den)
        pyl.show()
    result = pd.DataFrame(columns=['freq', 'spec'])
    result['freq'] = f
    result['spec'] = Pxx_den
    result = result.sort_values(by='spec', ascending=False)
    cycle_list = 1 / result.head(12)['freq'].values
    is_cycle = False
    cycles = [int(sum(cycle_list)//len(cycle_list))]
    for cycle in cycle_list:
        if cycle % 1 == 0:
            is_cycle = True
            cycles.append(cycle)
    return is_cycle, cycles
def is_stable(data, n_threshold=1.1, show_pic=False):
    raw_data_std = np.std(data, ddof=1)
    coeffs = wavedec(data, 'db4', level=2)
    cA2, cD2, cD1 = coeffs
    cD2_std = np.std(cD2, ddof=1)
    if show_pic is True:
        plt.subplot(311)
        plt.title('original')
        plt.plot(data)
        plt.subplot(312)
        plt.title('ca2')
        plt.plot(cA2)
        plt.subplot(313)
        plt.title('cd2')
        plt.plot(cD2)
        plt.show()
    # 全局波动指标与局部波动指标的比值来描述两者的差异
    n = raw_data_std / cD2_std
    # 比值小于阈值为平稳数据，大于为波动数据
    if n < n_threshold:
        return True
    else:
        return False