from torch import nn
import torch
from torch.nn import functional as F


"""
this implementation depends highly on torch.einsum 
"""



# todo
"""
figure out  whether the projection weight matrices for q,v and k are different for every head 
or it's just a big weight matrix before splitting into heads(am using the latter for now)

"""


class MultiHeadAttention(nn.Module):
    def __int__(self,d_model:int,n_heads:int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.queries = nn.Linear(self.d_model,self.d_model,bias=False)
        self.keys = nn.Linear(self.d_model,self.d_model,bias=False)
        self.values = nn.Linear(self.d_model,self.d_model,bias=False)
        self.fc_out = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(0.1)
        self.layerNorm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,queries,keys,values,mask)->torch.Tensor:

        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        q = queries.reshape(queries.shape[0],queries.shape[1],self.n_heads,self.d_head)
        k = keys.reshape(keys.shape[0],keys.shape[1],self.n_heads,self.d_head)
        v = values.reshape(values.shape[0],values.shape[1],self.n_heads,self.d_head)

        attn = torch.einsum("nqhd,nkhd->nhqk",q,k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0,float('-inf'))
        attn = torch.softmax(attn/(self.d_model**0.5),dim=3)

        # out = torch.einsum('nhqk,nvhd->nvhd')
        out = torch.einsum('nhql,nlhd->nqhd',attn,v).reshape(values.shape[0],values.shape[1],self.d_model)

        out = self.dropout(F.relu(self.fc_out(out)))
        out += q

        out = self.layerNorm(out)
        return out


class Intermediate(nn.Module):
    def __int__(self,d_model:int,activation,intermediateSize:int=2048):

        self.act1 = activation()
        self.linear1 = nn.Linear(d_model,intermediateSize)
        self.linear2 = nn.Linear(intermediateSize,d_model)
        self.dropout = nn.Dropout(0.1)
        self.layerNorm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,x):
        res = x
        out = self.dropout(self.act1(self.linear2(self.linear1(x))))
        out += res
        out = self.layerNorm(out)
        return out


class EncoderLayer(nn.Module):

    def __int__(self,d_model:int,n_heads):
        super(EncoderLayer, self).__int__()
        self.multiHeadAttention = MultiHeadAttention(d_model,n_heads)
        self.feedForwardNetwork = Intermediate(d_model,nn.ReLU)

    def forward(self,x,mask):
        x = self.multiHeadAttention(x,mask)
        return self.feedForwardNetwork(x)
