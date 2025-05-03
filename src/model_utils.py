import torch
import torch.nn as nn
import math
from einops import rearrange
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = rearrange(x, "b f d -> f b d")
        
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        x = self.dropout(x)
        
        x = rearrange(x, "f b d -> b f d")
        
        return x

class FeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_size, ff_size)
        self.linear2 = nn.Linear(ff_size, input_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out =  self.dropout2(out)
        return out