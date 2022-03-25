import torch
from torch import nn

"""
this implementation of the lexicon encoder is based on the one from BERT https://arxiv.org/abs/1810.04805
the positionalEncoder formula can be found in the Attention is all you need paper https://arxiv.org/abs/1706.03762
PE(pos,2i)   =  sin(pos/10000.exp(2i/d_model))
PE(pos,2i+1) =  cos(pos/10000.exp(2i/d_model))
"""


class PositionalEncoder(nn.Module):
    def __init__(self,d_model:int,max_len:int=512):
        super().__init__()
        pe = torch.zeros((max_len,d_model)).float()

        div_term = torch.pow(10000,2*(torch.arange(0,d_model)//2)/d_model).unsqueeze(0).float()
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        pe[:,0::2] = torch.sin((pos/div_term)[:,0::2])
        pe[:,1::2] = torch.cos((pos/div_term)[:,1::2])

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.pe[:,:x.size[1]].clone().detach()


class LexiconEncoder(nn.Module):

    def __init__(self,d_model:int, vocab_size:int, max_seq_length: int=512):
        super().__init__()
        self.positionalEncoder = PositionalEncoder(d_model,max_seq_length)
        self.segmentEmbedding = nn.Embedding(2,d_model)
        self.tokenEmbeddings = nn.Embedding(vocab_size,d_model)

    def forward(self,x,token_types)->torch.Tensor:
        x = self.tokenEmbeddings(x)
        x = x + self.positionalEncoder(x)
        x = x + self.segmentEmbedding(token_types)

        return x










