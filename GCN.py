import torch
import networkx as nx
import dgl
import matplotlib

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
# https://zhuanlan.zhihu.com/p/93828551
def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
G.ndata['feat'] = torch.eye(34)


# 主要定义message方法和reduce方法
# NOTE: 为了易于理解，整个教程忽略了归一化的步骤
def gcn_message(edges):
    # 参数：batch of edges
    # 得到计算后的batch of edges的信息，这里直接返回边的源节点的feature.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # 参数：batch of nodes.
    # 得到计算后batch of nodes的信息，这里返回每个节点mailbox里的msg的和
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g 为图对象； inputs 为节点特征矩阵
        # 设置图的节点特征
        g.ndata['h'] = inputs
        # 触发边的信息传递
        g.send(g.edges(), gcn_message)
        # 触发节点的聚合函数
        g.recv(g.nodes(), gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        # 线性变换
        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


# 以空手道俱乐部为例
# 第一层将34层的输入转化为隐层为8
# 第二层将隐层转化为最终的分类数2
# net = GCN(34, 8, 2)
# inputs = torch.eye(34)
# labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
# labels = torch.tensor([0, 1])  # their labels are different

net = GCN(34, 8, 3)
inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 2, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1, 2])  # their labels are different

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
import matplotlib.animation as animation
import matplotlib.pyplot as plt
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)

def draw(i):
    # fig = plt.figure(dpi=150)
    plt.clf()#    此时不能调用此函数，不然之前的点将被清空。
    color = [ 'green', 'b', 'r', '#7FFFD4', '#FFC0CB', '#00022e','#F0F8FF']
    # cls1color = '#00FFFF'
    # cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(color[cls])
        # colors.append(cls1color if cls else cls2color)
    # ax = fig.subplots()
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)

    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=200)


for epoch in range(40):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

# 动态显示
# fig = plt.figure(dpi=150)
# fig.clf()
# ax = fig.subplots()
# # draw(0)  # draw the prediction of the first epoch
# ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
# plt.pause(30)
# # plt.close()

# 降维二维显示
# import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
# from sklearn.decomposition import PCA           #加载PCA算法包
#
# x, y= [], []
# for i in range(34):
#     x.append(all_logits[39][i].numpy())
#     y.append(all_logits[39][i].numpy().argmax())
#     print("{} {}".format(all_logits[39][i].numpy(), all_logits[39][i].numpy().argmax()))
#
#
# pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
# reduced_x=pca.fit_transform(x)#对样本进行降维
# print(reduced_x)
#
# # #可视化
# color = ['b', 'r', '#7FFFD4', '#FFC0CB', '#00022e','#F0F8FF', 'green']
# for index, item in enumerate(reduced_x):
#     plt.scatter(item[0], item[1], c= color[y[index]])
# plt.show()


# 三维显示
import math


x, c= [], []
color = ['b', 'r','#7FFFD4','#FFC0CB', '#00022e','#F0F8FF', 'green']
for i in range(34):
    x.append(all_logits[39][i].numpy())
    c.append(all_logits[39][i].numpy().argmax())
    print("{} {}".format(all_logits[39][i].numpy(), all_logits[39][i].numpy().argmax()))
x1 = [i[0] for i in x]
x2 = [i[1] for i in x]
y = [i[2] for i in x]
d = []
for i in range(len(x1)):
    d.append(color[c[i]])
fig1=plt.figure()#创建一个绘图对象
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)
ax.scatter(x1, x2, y, c = d)#用取样点(x,y,z)去构建曲面
plt.show()#显示模块中的所有绘图对象
