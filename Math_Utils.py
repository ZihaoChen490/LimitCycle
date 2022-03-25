import numpy as np
import matplotlib.pyplot as plt
import os, torch, pickle, zipfile,pywt,types
from math import *
from Data_Utils import *
from Plot_Utils import *
from torch import tensor
import scipy, scipy.misc, scipy.integrate,imageio, shutil,math
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics import roc_curve, roc_auc_score
solve_ivp = scipy.integrate.solve_ivp
import networkx as nx
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import scipy.signal as signal

EPSILON=0
f = lambda x,a : a*x
import math
from contextlib import suppress
import warnings
import torch as tc
class Quick_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.01, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]
        print('self.kkkkk', len(self.centers_))
        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            print("质点:",self.clf_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
            print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


def fftTransfer(timeseries, n=10, fmin=0.2):
    #print(len(timeseries))
    yf = abs(np.fft.fftn(timeseries))#,axes=1))  # 取绝对值
    #print('yfffffffffffffff',yf.shape)
    yfnormlize = yf / len(timeseries)  # 归一化处理
    print('yfnormlize is',yfnormlize.shape)
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
    print('xorder = ', xorder)
    print(len(xwbest[0]))
    print(xwbest[1])
    print(type(xorder))
    xworder=[]
    fworder=[]
    if len(fwbest) <= n:
        xworder = [xwbest[0][xorder[x]] for x in range(len(xorder))] # 返回频率从大到小的极值顺序
        fworder = [fwbest[xorder[x]] for x in range(len(xorder))] # 返回幅度
        #fwbest = fwbest[fwbest >= fmin].copy()
        return xworder,fworder#len(timeseries)/xwbest[0][:n], fwbest[:n]    #x转化为周期输出,f是振幅
    else:
        xworder = [xwbest[0][xorder[x]] for x in range(len(xorder))]  # 返回频率从大到小的极值顺序
        fworder = [fwbest[xorder[x]] for x in range(len(xorder))]  # 返回幅度
        return xworder[:n],fworder[:n]
        #fwbest = fwbest[fwbest >= fmin].copy()
        #print('len fwbest is',len(fwbest))
        #print('xwbest is also',xwbest)
        #return len(timeseries)/xwbest[0][:n], fwbest[:n]  # 只返回前n个数   #转化为周期输出


def eucliDist(A,B):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(A,B)]))

def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    retData = (dataset - minVals) / ranges
    return retData, minVals, ranges
'''
# integrate along those fields starting from point (1,0)
def get_model(args, baseline):
    output_dim = args.input_dim if baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)
    model_name = 'baseline' if baseline else 'hnn'
    path = "{}/pend{}-{}.tar".format(args.save_dir, RK4, model_name)
    model.load_state_dict(torch.load(path))
    return model
'''

def rk4(fun, y0, t, dt, *args, **kwargs):
  dt2 = dt / 2.0
  k1 = fun(y0, t, *args, **kwargs)
  k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
  k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
  k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy


def wavelet_denoising(data):
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(data, db4)
    print(coeffs)
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    meta = pywt.waverec(coeffs, db4)
    return meta
def phase_flow(E, mass, k, theta=0):
    a = np.sqrt(2 * mass * E)
    b = np.sqrt(2 * E / k)
    q = b * np.cos(theta)
    p = a * np.sin(theta)
    return q, p


def CentralityMeasures(G):
    # Betweenness centrality
    bet_cen = nx.betweenness_centrality(G)
    # Closeness centrality
    clo_cen = nx.closeness_centrality(G)
    # Eigenvector centrality
    eig_cen = nx.eigenvector_centrality(G)
    # Degree centrality
    deg_cen = nx.degree_centrality(G)
    #print bet_cen, clo_cen, eig_cen
    print("# Betweenness centrality:" + str(bet_cen))
    print("# Closeness centrality:" + str(clo_cen))
    print("# Eigenvector centrality:" + str(eig_cen))
    print("# Degree centrality:" + str(deg_cen))
    return deg_cen
def quantile_p(data, p):
    pos = (len(data) + 1)*p
    #pos = 1 + (len(data)-1)*p
    pos_integer = int(math.modf(pos)[1])
    pos_decimal = pos - pos_integer
    Q = data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1])*pos_decimal
    return Q
def arsinh(x=0):
    return np.log(x+np.sqrt(x*x+1))
def transform(raw_matrix,transformtype):
    if transformtype == "arsinh":
        transformed_matrix=arsinh(raw_matrix)
        return transformed_matrix
    elif transformtype == "none":
        transformed_matrix=raw_matrix
        return transformed_matrix
    else:
        print("This kind of 'transformtype' is unavailable right now!")
        return None
def dist(raw_data,type='cosine'):
    if type=='minkowski':
        distA = pdist(raw_data, metric=type,p=2)
        distB = squareform(distA)
        return distB
    distA=pdist(squareform(raw_data,force=True),metric=type)
    distB=squareform(distA)
    return distB
def integrate_model(model, t_span, y0, fun=None, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  fun = default_fun if fun is None else fun
  return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
def rk4(fun, y0, t, dt, *args, **kwargs):
  dt2 = dt / 2.0
  k1 = fun(y0, t, *args, **kwargs)
  k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
  k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
  k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy



def L2_loss(u, v):
    if isinstance(u, tuple):
          ls=[(u[i]-v[i]).pow(2) for i in range(len(u))]
          return sum(ls).mean()
    if isinstance(u,list):
        u=tensor(u)
    if isinstance(v,list):
        v=tensor(v)
    if type(u) is np.ndarray:
        u=tensor(u)
    #print(torch.norm((u-v), p='fro', dim=None, keepdim=False, out=None, dtype=None))
    return torch.norm((u-v), p='fro', dim=None, keepdim=False, out=None, dtype=None)

def cal_threshold(dataset):
    res=len(dataset)
    idx=np.random.randint(0, res, res//100)
    Exp=np.mean(dataset[idx],axis=0)
    #Esig=np.cov(dataset[idx],rowvar=0)
    #Esig.diagonal()
    return Exp/100



def L1_loss(u, v):
    if isinstance(u, tuple):
          ls=[(u[i]-v[i]).pow(1) for i in range(len(u))]
          return sum(ls).mean()
    if isinstance(u,list):
        u=tensor(u)
    if isinstance(v,list):
        v=tensor(v)
    if type(u) is np.ndarray:
        u=tensor(u)
    #print(torch.norm((u-v), p='fro', dim=None, keepdim=False, out=None, dtype=None))
    return torch.norm((u-v), p=2, dim=None, keepdim=False, out=None, dtype=None)

def Ito_loss(u, v):
    if isinstance(u, tuple):
          ls=[(u[i]-v[i]).pow(2) for i in range(len(u))]
          return sum(ls).mean()
    if isinstance(u,list):
        u=tensor(u)
    if isinstance(v,list):
        v=tensor(v)
    if type(u) is np.ndarray:
        u=tensor(u)
    return pow((u-v),2).mean()



def read_lipson(experiment_name, save_dir):
  desired_file = experiment_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)
def str2array(string):
    lines = string.split('\\n')
    names = lines[0].strip("b'% \\r").split(' ')
    dnames = ['d' + n for n in names]
    names = ['trial', 't'] + names + dnames
    data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]
    return np.asarray(data), names
def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)
def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing
def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl
def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
    '''Given a three dimensional array [frames, height, width], make
    a gif and save it.'''
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    for i in range(len(frames)):
        im = (frames[i].clip(-.5,.5) + .5)*255
        im[divider,:] = 0
        im[divider + 1,:] = 255
        if pixels is not None:
          im = scipy.misc.imresize(im, pixels)
        scipy.misc.imsave(temp_dir + '/f_{:04d}.png'.format(i), im)
    images = []
    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(temp_dir, file_name)
            images.append(imageio.imread(file_path))
    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)
    shutil.rmtree(temp_dir) # remove all the images
    return png_save_path
def ss(output, label):
    threthod,tp,fp = 0.5,0,0
    for i in range(len(output)):
        if output[i] > threthod:
            if label[i] == 1:
                tp += 1
            else:
                fp += 1
    sens = tp / sum(label)
    spec = 1 - fp / (len(label) - sum(label))
    return sens, spec
def Spiralmatrix(Dimension):
    array_lists = [[0 for j in range(Dimension)] for i in range(Dimension)]
    num = 1
    for i in range(Dimension//2+1):  # 0 1 2
        for j in range(i, Dimension-i):
            array_lists[i][j] = num
            num += 1
        for j in range(i+1, Dimension-i):
            array_lists[j][Dimension-i-1] = num
            num += 1
        for j in range(Dimension-i-2, i, -1):
            array_lists[Dimension-i-1][j] = num
            num += 1
        for j in range(Dimension-i-1, i, -1):
            array_lists[j][i] = num
            num += 1
    for i in array_lists:
        for j in i:
            print(j, end='\t')
        print()
def chase(A,B):
    C=(A+B)/2
    return C
def vel(A,B,t):
#    va=Vector(A)
#    vb=Vector(B)
    return (A-B)/t
def acc_vel(A,B,C,t):
    return (vel(A,B,t)-vel(B,C,t))/t
def hamilton_solve(tmax=100, tsteps=100, method='',y0=[0, 0]):
    t = np.linspace(0, tmax, tsteps)
    sol = odeint(f, y0, t)
    return sol

def f(y, t):
    '''return [p, -q]
    本问题中 f 不显含时间'''
    return y[1], -y[0]
class Vector:
    def __init__(self,lst):
        if isinstance(lst,np.ndarray):
            self._values=lst.tolist()
        elif isinstance(lst,np.float64):
            self._values=lst.tolist()
        else:
            self._values =list(lst)
    @classmethod
    def zero(cls,dim):
        '返回一个dim维的零向量'
        return cls([0] * dim)
    def normalize(self):
        '返回向量的单位向量'
        if self.norm() < EPSILON:
            raise ZeroDivisionError('Normalize error! norm is zero.')
        return Vector(self._values) / self.norm()
    def norm(self):
        '返回向量的模'
        return math.sqrt(sum(e**2 for e in self))
    def __add__(self, another):
        get_mes(another)
        get_mes(self)
        self.__str__()
        assert len(self) == len(another), \
            'Error in adding. length of vectors must be same.'
        return Vector([a + b for a, b in zip(self, another)])
    def __sub__(self, another):
        assert len(self) == len(another), \
            'Error in adding. length of vectors must be same.'
        return Vector([a - b for a, b in zip(self, another)])
    def __mul__(self, k):
        '返回数量乘法的结果向量：self * k'
        return Vector([k * e for e in self])
    def __truediv__(self, k):
        '返回数量除法的结果向量：self / k'
        return (1 / k) * self
    def __rmul__(self, k):
        '返回数量乘法的结果向量：k * self'
        return self * k
    def __pos__(self):
        '返回向量取正的结果'
        return 1 * self
    def __neg__(self):
        return -1 * self
    def __iter__(self):
        '返回向量的迭代器'
        return self._values.__iter__()
    def __getitem__(self, index):
        '取向量的第index个元素'
        return self._values[index]
    def __len__(self):
        '返回向量长度，有多少元素'
        return len(self._values)
    def __repr__(self):
        return 'Vector({})'.format(self._values)
    def __str__(self):
        return '({})'.format(', '.join(str(e) for e in self._values))
    def __inpro__(self):
        return 0



__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

###### Basic tools ######

NoneVars = ["None", "none", None, "Null", "null", "No", "no", "False", "false", False, "0", 0]

def solve_vec(b, A):
    return tc.solve(b.unsqueeze(-1), A)[0].squeeze(-1)

def smart_grad(outputs, inputs, grad_outputs = None, retain_graph = None, create_graph = None,
        only_inputs = True, allow_unused = False):
    if create_graph is None: create_graph = tc.is_grad_enabled()
    if retain_graph is None: retain_graph = create_graph
    gradients = tc.autograd.grad(outputs, inputs, grad_outputs, retain_graph, create_graph, only_inputs, allow_unused)
    if isinstance(inputs, tc.Tensor): return gradients[0]
    else: return gradients

def track_var(var: tc.Tensor, track: bool):
    if track: return var if var.requires_grad else var.detach().requires_grad_(True)
    else: return var.detach() if var.requires_grad else var

###### Jacobian and Laplacian ######

def jacobian_normF2(y, x, n_mc = 0):
    """
    `y` is vector-valued. `x` requires grad.
    If `n_mc` == 0, use exact calculation. If `n_mc` > 0, use the Hutchinson's estimator.
    """
    if n_mc > 0: # Hutchinson's estimator
        ls_gradx_yproj = [smart_grad(y, x, grad_outputs=tc.randn_like(y), retain_graph=True)
                for _ in range(n_mc)]
        ls_quad = [(gradx_yproj**2).sum() for gradx_yproj in ls_gradx_yproj]
        return tc.stack(ls_quad).mean()
    elif n_mc == 0: # exact calculation
        with tc.enable_grad(): ls_y = y.flatten().unbind()
        return tc.stack([(smart_grad(yi, x, retain_graph=True)**2).sum() for yi in ls_y]).sum()

def jacobian(y, x, ndim_batch = 0):
    """
    `y` is vector-valued. `x` requires grad.
    `y.shape` = shape_batch + shape_y, `x.shape` = shape_batch + shape_x,
    where len(shape_batch) = ndim_batch.
    Output shape: shape_batch + shape_x + shape_y.
    This is the 'transpose' to the functional form `autograd.functional.jacobian`.
    """
    shape_batch = y.shape[:ndim_batch]
    print(y.shape)
    print(y.shape[:ndim_batch])
    assert shape_batch == x.shape[:ndim_batch]
    shape_y = y.shape[ndim_batch:]
    shape_x = x.shape[ndim_batch:]
    print('what we get is that',shape_x,shape_y)
    with tc.enable_grad():
        y_sum = y.sum(dim=tuple(range(ndim_batch))) if ndim_batch else y
        ls_y = y_sum.flatten().unbind()
    jac_flat = tc.stack([smart_grad(yi, x, retain_graph=True) for yi in ls_y], dim=-1) # shape_batch + shape_x + (shape_y.prod(),)
    return jac_flat.reshape(shape_batch + shape_x + shape_y)

def directional_jacobian(y, x, v):
    """
    Vector-Jacobian product `( v(x)^T (grad_x y(x)^T) )^T`,
    or the tensor form of `tc.autograd.functional.jvp` (their Jacobian is transposed).
    Based on `grad_eta (v(x)^T grad_x (eta^T y(x))) = grad_eta (v(x)^T (grad_x y(x)^T) eta) = ( v(x)^T (grad_x y(x)^T) )^T`.
    """
    eta = tc.zeros_like(y, requires_grad=True) # The value of `eta` does not matter
    with tc.enable_grad():
        gradx_yproj = smart_grad(y, x, grad_outputs=eta)
    return smart_grad(gradx_yproj, eta, grad_outputs=v)

def laplacian(y, x, n_mc = 0, gradx_y = None):
    """
    `y` is a scalar, or is summed up first. `x` requires grad.
    `gradx_y` overwrites `y`. It should have the same shape as `x`.
    If `n_mc` == 0, use exact calculation. If `n_mc` > 0, use the Hutchinson's estimator.
    """
    if gradx_y is None:
        with tc.enable_grad(): gradx_y = smart_grad(y.sum(), x)
    if n_mc > 0: # Hutchinson's estimator
        ls_eta = [tc.randn_like(gradx_y) for _ in range(n_mc)]
        ls_quad = [(smart_grad(gradx_y, x, grad_outputs=eta, retain_graph=True) * eta).sum() for eta in ls_eta]
        return tc.stack(ls_quad).mean()
    elif n_mc == 0: # exact calculation
        with tc.enable_grad(): ls_gradx_y = gradx_y.flatten().unbind()
        return tc.stack([smart_grad(gradxi_y, x, retain_graph=True).flatten()[i]
                for i, gradxi_y in enumerate(ls_gradx_y)]).sum()

###### Losses in standalone form ######

def compatibility_jacnorm(logp, logq, x, z, n_mc = 1):
    """ `x` and `z` require grad. All inputs have a matching batch size. """
    with tc.enable_grad():
        gradx_logratio = smart_grad(logp.sum() - logq.sum(), x)
    return jacobian_normF2(gradx_logratio, z, n_mc) / logp.numel()

###### Losses assembled: avoid repeated gradient evaluation ######