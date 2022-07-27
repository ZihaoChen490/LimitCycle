import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch,time,scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from sklearn.metrics import mean_squared_error
from torch import Tensor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import networkx as nx
from Data_Utils import *
from Math_Utils import *
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
plt.style.available



def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 2)
        dx = model(x).data.numpy().reshape(-1)
        return dx
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

def predict(row, model):
    # convert row to data
    try:
        row = torch.tensor([row],requires_grad=True,dtype=torch.float32)
        #row = torch.Tensor([row])
    except:
        print('row is Tensor already')
    # make prediction
    #yhat = sum([model[i](row) for i in range(len(model))])/len(model)
    yhat=model(row)
    # retrieve numpy array
    if isinstance(yhat, tuple):
        ghat=np.array([yhat[i].detach().numpy() for i in range(len(yhat))]).flatten().reshape(len(yhat[0]),2)
        return ghat
    yhat = yhat.detach().numpy()
    return yhat


def get_mesh_inputs(data, gridsize=30):
    xmin, xmax, dxmin, dxmax = get_range(data)
    mesh_x, mesh_dx = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(dxmin, dxmax, gridsize))
    np_mesh_inputs = np.stack([mesh_x.flatten(), mesh_dx.flatten()]).T
    mesh_inputs = torch.tensor(np_mesh_inputs, requires_grad=True, dtype=torch.float32)
    return mesh_inputs
def time_derivative( x, model,t=None, separate_fields=False):
    yhat = predict(x, model)
    F1, F2 = (x)  # traditional forward pass
    conservative_field = torch.zeros_like(x)  # start out with both components set to 0
    solenoidal_field = torch.zeros_like(x)
    if separate_fields:
        return [conservative_field, solenoidal_field]
    return conservative_field + solenoidal_field
def get_scalar_fields(data_all, model):
    gridsize = 30
    mesh_inputs = get_mesh_inputs(data_all, gridsize)
    print(mesh_inputs.shape)
    X = mesh_inputs.reshape(gridsize, gridsize, 2)[..., 0].detach().numpy()
    dX = mesh_inputs.reshape(gridsize, gridsize, 2)[..., 1].detach().numpy()
    yhat = predict(mesh_inputs, model)
    F_2 = yhat - mesh_inputs.detach().numpy()
    F2_norm = np.linalg.norm(F_2, ord=2, axis=1, keepdims=False)
    np_F2 = F2_norm.reshape(gridsize, gridsize)
    fig = plt.figure(figsize=(4.2, 4), facecolor='white', dpi=100)
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    plt.contourf(X, dX, np_F2, cmap='gray_r', levels=20)
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$\dot \\theta$")
    plt.title("Phase space")
    plt.show()
def GET_VECTOR_FIELDS(data_all, base_model,hnn_model, save_path):
    t_span = [0, 1]
    y0 = np.asarray([0.75, 0])
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000), 'rtol': 1e-12}
    base_ivp = integrate_model(base_model, t_span, y0, **kwargs)
    hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)
    DPI = 300
    FORMAT = 'pdf'
    LINE_SEGMENTS = 10
    ARROW_SCALE = 30  # 100 for pend-sim, 30 for pend-Pend
    ARROW_WIDTH = 9e-3
    LINE_WIDTH = 2
    RK4 = ''
    mesh_inputs = get_mesh_inputs(data_all, gridsize=10)
    input_x = mesh_inputs.detach().numpy()
    base_dx = base_model(mesh_inputs).detach().numpy()
    hnn_dx = hnn_model(mesh_inputs).detach().numpy()
    fig = plt.figure(figsize=(11.3, 3.2), facecolor='white', dpi=DPI)
    ax = fig.add_subplot(1, 3, 1, frameon=True)
    for data in data_all:
        for i in range(len(data)-1):
            ax.quiver(data[i][0], data[i][1], data[i+1][0], data[i+1][1],cmap='gray_r',color='red', scale=ARROW_SCALE, width=ARROW_WIDTH)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$", rotation=0)
    ax.set_title("Pend pendulum data")
    plt.legend()
    ax = fig.add_subplot(1, 3, 2, frameon=True)
    ax.quiver(input_x[:, 0], input_x[:, 1], base_dx[:, 0], base_dx[:, 1],
              cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)
    for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i) / LINE_SEGMENTS, 0, 1 - float(i) / LINE_SEGMENTS)
        ax.plot(l[:, 0], l[:, 1], color=color, linewidth=LINE_WIDTH)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$", rotation=0)
    ax.set_title("Baseline NN")
    ax = fig.add_subplot(1, 3, 3, frameon=True)
    ax.quiver(input_x[:, 0], input_x[:, 1], hnn_dx[:, 0], hnn_dx[:, 1],
              cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH)
    for i, l in enumerate(np.split(hnn_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i) / LINE_SEGMENTS, 0, 1 - float(i) / LINE_SEGMENTS)
        ax.plot(l[:, 0], l[:, 1], color=color, linewidth=LINE_WIDTH)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$", rotation=0)
    ax.set_title("Hamiltonian NN")
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/HNN{}.{}'.format(save_path, RK4, FORMAT))

def quick_landscape(grid,raw_data):
    land=[[0 for i in range(grid.shape[0])]for j in range(grid.shape[1])]
    get_mes(grid)
    print(grid[:,:,0])
    print(grid.shape)
    for i in range(len(grid.shape[0])):
        for j in range(len(grid.shape[1])):
            for k in range(len(raw_data)):
                if raw_data[k,0]:
                    land[i][j]=land[i][j]+1
    print(land)



def plot_vector_fields_landscape(data):
    ax = plt.subplot(1, 1, 1)
    ARROW_SCALE = 20
    ARROW_WIDTH = 2e-3
    gap=10
    xgrid = np.linspace(0, 1, num=data.shape[0]//gap, endpoint=True, retstep=False, dtype=float)
    ygrid = np.linspace(0, 1, num=data.shape[0]//gap, endpoint=True, retstep=False, dtype=float)
    X, Y = np.meshgrid(xgrid, ygrid)
    print(X.shape)
    X=np.array(X).reshape(-1,1)
    Y = np.array(Y).reshape(-1, 1)
    U=np.array(np.gradient(data[::gap,::gap])[0]).reshape(-1,1)
    V=np.array(np.gradient(data[::gap,::gap].T)[0]).reshape(-1,1)
    print(X.shape)
    #print(Y.shape())
    print(U.shape)
    ax.quiver(X, Y, U, V, cmap='gray_r', color='red', scale=ARROW_SCALE,width=ARROW_WIDTH)
    ax.set_xlabel("$PCA1$")
    ax.set_ylabel("$PCA2$", rotation=0)
    ax.set_title("Phase flow")
    plt.legend()
    plt.show()
    return data
def plot_vector_fields(data,dim1,dim2):
    ax = plt.subplot(1, 1, 1)
    ARROW_SCALE = 30
    ARROW_WIDTH = 9e-3
    step=1000
    print(data.shape)
    X=data[:,dim1]
    Y=data[:,dim2]
    U=np.append(X[1:],X[0])-X
    V=np.append(Y[1:],Y[0])-Y
    print(U.min())
    print(U.max())
    ax.quiver(X, Y, U, V, cmap='gray_r', color='red', scale=ARROW_SCALE,width=ARROW_WIDTH)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$", rotation=0)
    ax.set_title("Pend pendulum data")
    plt.legend()
    plt.show()
    return data

def aligns(Cycle_nos,Cycle_std,raw_data,peak_point):
    scaler=raw_data.max()
    Cycle_nos=(np.array(Cycle_nos))/(scaler)
    Cycle_std=(Cycle_std)/(scaler)
    raw_data=(raw_data)/(scaler)
    peak_point=peak_point/scaler
    return Cycle_nos,Cycle_std,raw_data,peak_point
# print('comp',db.components_.shape)
# print('lab',db.labels_)
# print('core_sap',db.core_sample_indices_)
# print('uni_lab',uni_labels)
# plot_3D(self.raw_data[:200, 0],self.raw_data[:200, 1],C)
def plot_3D(x,y,z):
    # xx, yy = np.meshgrid(x, y)
    print(z)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.arange(x.min(), x.max(), 1/len(x))
    ynew = np.arange(y.min(), y.max(), 1/len(y))
    znew = f(xnew, ynew)
    xx1, yy1 = np.meshgrid(xnew, ynew)
    print(xx1.shape)
    newshape = (xx1.shape[0]) * (xx1.shape[1])
    y_input = xx1.reshape(newshape)
    x_input = yy1.reshape(newshape)
    z_input = znew.reshape(newshape)
    sns.set(style='white')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_input, y_input, z_input, cmap=cm.coolwarm)
    plt.show()

def snn_sim_matrix(X, k=5):
    try:
        X = np.array(X)
    except:
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    sim_matrix = 0.5 + np.zeros((samples_size, samples_size))  # snn相似度矩阵
    for i in range(samples_size):
        t = np.where(knn_matrix == i)[0]
        c = list(combinations(t, 2))
        for j in c:
            if j[0] not in knn_matrix[j[1]]:
                continue
            sim_matrix[j[0]][j[1]] += 1
    sim_matrix = 1 / sim_matrix  # 将相似度矩阵转化为距离矩阵
    sim_matrix = np.triu(sim_matrix)
    sim_matrix += sim_matrix.T - np.diag(sim_matrix.diagonal())
    return sim_matrix
def snn(raw_data,k=8):
    #figsnn= plt.figure()
    X=raw_data
    t1 = time.time()
    sim_matrix = snn_sim_matrix(X, k)
    t2 = time.time()
    print( "the time of creating sim matrix is %.5fs" % (t2 - t1))
    t1 = time.time()
    db = DBSCAN(eps=0.5, min_samples=5, metric='precomputed').fit(sim_matrix)
    t2 = time.time()
    print( "the time of clustering is %.5fs" % (t2 - t1))
    core_samples_mask=np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=3)
    plt.title('SNN')
    plt.show()
    return db,unique_labels,n_clusters_,sim_matrix

def plot_graph(GN):
    G=nx.Graph()
    point=[i for i in range(len(GN))]
    N=[[i] for i in range(len(GN))]
    for i in range(len(GN)):
        for j in range(len(GN)):
            if i!=j and GN[i][j]:
                N[i].append(j)
    print(N)
    G.add_nodes_from(point)
    edglist=[]
    for i in range(len(GN)):
        for j in range(1,len(N[i])):
            edglist.append((N[i][0],N[i][j]))
    print(edglist)
    G.add_edges_from(edglist)
    deg_cen=CentralityMeasures(G)
#    G=nx.Graph(edglist)
    position = nx.circular_layout(G)
    nx.draw_networkx_nodes(G,position, nodelist=point, node_color="r")
    nx.draw_networkx_edges(G,position)
    nx.draw_networkx_labels(G,position)
    #plt.show()
    return G,deg_cen

def plot_phase():
    theta = np.linspace(0, 2 * np.pi, 100)
    q1, p1 = phase_flow(E=20, mass=1, k=1, theta=theta)
    q2, p2 = phase_flow(E=20, mass=1, k=2, theta=theta)
    plt.plot(q1, p1, label="k=1")
    plt.plot(q2, p2, label="k=2")
    plt.xlabel('q')
    plt.ylabel('p')
    plt.legend(loc='best')
    #plt.style.use(['science', 'ieee', 'no-latex'])
    plt.gca().set_aspect('equal')
    plt.show()