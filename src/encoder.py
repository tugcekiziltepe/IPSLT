import torch.nn as nn
from model_utils import PositionalEncoding, PositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, ff_size = 2048, num_heads = 8, hidden_size = 400, beta = 0.5, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads,batch_first=True)
        self.feed_forward = PositionWiseFeedForward(hidden_size, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.beta = beta

    def forward(self, e_l_minus_1, e_k_minus_1, src_mask): # (1, 4, 400)
        self_att_output, _ = self.self_attention(e_l_minus_1, e_l_minus_1, e_l_minus_1, src_mask) # (1, 4, 400)
        # self_att_output = self.norm1(self.dropout(self_att_output) + e_l_minus_1) # (1, 4, 400)
        
        cross_att_output, _ = self.cross_attention(e_l_minus_1, e_k_minus_1, e_k_minus_1) # (1, 4, 400)
        attention_output = self.beta * self_att_output + (1 - self.beta) * cross_att_output # (1, 4, 400)
        cross_att_output = self.norm2(self.dropout(attention_output) + e_k_minus_1) # (1, 4, 400)
        
        ff_output = self.feed_forward(cross_att_output) # (1, 4, 400)
        ff_output = self.norm3(self.dropout(ff_output) +  cross_att_output) # (1, 4, 400)
        
        return ff_output
    
class TransformerEncoder(nn.Module):

    def __init__(self,
                hidden_size = 400,
                ff_size = 2048,
                num_layers = 3,
                num_heads = 8,
                dropout= 0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer()
            for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size, 31102)
        self.emb_dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask, encoded_mem = None, initial = False):
        x = self.pe(x) # (1, 1, 400)
        x = self.emb_dropout(x)
        
        for i, layer in enumerate(self.layers):
            if initial or i == 0:
                e_k_minus_1 = x
            elif encoded_mem is not None:
                e_k_minus_1 = encoded_mem
            
            x = layer(x, e_k_minus_1, src_mask)
            e_k_minus_1 = x
        return x