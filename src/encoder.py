import torch.nn as nn
from src.model_utils import PositionalEncoding, FeedForward

    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, ff_size=2048, num_heads=8, hidden_size=400, beta=0.5, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.feed_forward = FeedForward(hidden_size, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.beta = beta

    def forward(self, e_k_l_minus_1, e_k_minus_1, src_mask):  # (B, S, H)
        # src_mask: (B, S) where True = valid token, False = padding

        # Self-attention with padding mask
        self_att_output, _ = self.self_attention(
            e_k_l_minus_1, e_k_l_minus_1, e_k_l_minus_1, key_padding_mask=~src_mask
        )

        # Cross-attention with same padding mask
        cross_att_output, _ = self.cross_attention(
            e_k_l_minus_1, e_k_minus_1, e_k_minus_1, key_padding_mask=~src_mask
        )

        attention_output = self.beta * self_att_output + (1 - self.beta) * cross_att_output
        cross_att_output = self.norm1(self.dropout(attention_output) + e_k_l_minus_1)

        ff_output = self.feed_forward(cross_att_output)
        ff_output = self.norm2(ff_output + cross_att_output)

        return ff_output
   
class TransformerEncoder(nn.Module):
    def __init__(self,
                 hidden_size=400,
                 ff_size=2048,
                 num_layers=3,
                 num_heads=8,
                 dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                ff_size=ff_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout) # Buna bir bak!

    def forward(self, x, src_mask, e_k_minus_1=None):
        out = x.contiguous() 

        for i, layer in enumerate(self.layers):

            out = layer(e_k_l_minus_1= out, e_k_minus_1=e_k_minus_1, src_mask=src_mask)

        return out