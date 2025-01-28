import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

class Attention(nn.Module):
    """
    Implements a single-head attention mechanism. This class supports both self-attention
    and cross-attention depending on the context provided.

    Args:
        embed_dim (int): The dimensionality of the embedding space.
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int, optional): The dimensionality of the context for cross-attention. 
                                     If None, self-attention is performed.
        num_heads (int, optional): The number of attention heads. Default is 1.
    """
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, tokens, context=None):
        if self.self_attn:
            Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
        else:
            Q, K, V = self.query(tokens), self.key(context), self.value(context)
        
        scoremats = torch.einsum('bth,bsh->bts', Q, K)
        attnmats = F.softmax(scoremats, dim=1)
        ctx_vecs = torch.einsum("bts,bsh->bth", attnmats, V)
        return ctx_vecs

class TransformerBlock(nn.Module):
    """
    Implements a Transformer block that includes self-attention, cross-attention, 
    and a feed-forward network with normalization layers.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
    """
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.attn_self = Attention(hidden_dim, hidden_dim)
        self.attn_cross = Attention(hidden_dim, hidden_dim, context_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x, context=None):
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x

class SpatialCrossAttention(nn.Module):
    """
    Implements a Spatial Cross Attention that applies a Transformer block to spatial data, 
    typically images. This allows spatial interactions within the Transformer architecture.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
    """
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in