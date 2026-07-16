import math

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def attention (Q,K,V, mask=None, attn_bias=None, return_weights=False):
    dk = Q.size(-1)
    heads = Q.size(1)
    
    T = (Q @ K.transpose(-2, -1))/math.sqrt(dk)
    
    if attn_bias is not None:
        if len(attn_bias.shape) == 3:
            attn_bias = attn_bias.unsqueeze(1).repeat(1, heads, 1, 1)
        T = T + attn_bias

    if mask is not None:
        T = T.masked_fill_(mask.unsqueeze(1)==0, -1e9)
    T = F.softmax(T, dim=-1)
    
    if return_weights:
        return T @ V, T
    else:
        return T @ V
    

class MultiHeadAttention (nn.Module):
    def __init__ (self, num_heads, embed_dim, input_dim=None, bias=True):
        super(MultiHeadAttention, self).__init__()
        if not input_dim:
            input_dim = embed_dim
        self.num_heads = num_heads
        self.dk = embed_dim//num_heads
        self.WQ = nn.Linear(input_dim, embed_dim, bias=bias)
        self.WK = nn.Linear(input_dim, embed_dim, bias=bias)
        self.WV = nn.Linear(input_dim, embed_dim, bias=bias)
        self.WO = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward (self, x, kv, mask, attn_bias=None, return_weights=False):
        batch_size = x.size(0)
        # Q.shape: (bs, heads, seq_len, dk)
        Q = self.WQ(x ).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)
        K = self.WK(kv).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)
        V = self.WV(kv).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)

        if len(mask.shape) == 2:
            mask = torch.einsum('bi,bj->bij', mask, mask)
        attn_result = attention(Q, K, V, mask=mask, attn_bias=attn_bias, return_weights=return_weights)
        if return_weights:
            x, weights = attn_result
        else:
            x = attn_result
            
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.dk)
        result = self.WO(x)
        
        if return_weights:
            return result, weights
        else:
            return result
    




