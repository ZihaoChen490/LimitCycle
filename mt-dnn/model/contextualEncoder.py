from torch import nn
import torch
from .myEncoder import  EncoderLayer
# todo
"""
write the transformerEncoder implementation  and replace the pytorch one 

"""


class TransformerEncoder(nn.Module):

    def __int__(self,d_model:int,n_heads:int,n_layers:int,implementation:str):
        assert (d_model % n_heads == 0) , "the embedding dimension should be divisible by the number of heads"
        # encoderLayer = nn.TransformerEncoderLayer(d_model,n_heads)
        encoderLayer = EncoderLayer(d_model,n_heads)
        self.contextualEncoder = nn.TransformerEncoder(encoderLayer,n_layers)

    def forward(self,x,padding_mask)->torch.Tensor:

        return self.contextualEncoder(x,padding_mask)
