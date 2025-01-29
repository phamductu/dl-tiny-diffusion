import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

from positional_embeddings import PositionalEmbedding

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.dense(x).unsqueeze(2).unsqueeze(3)

# class Attention(nn.Module):
#     """
#     Implements a single-head attention mechanism. This class supports both self-attention
#     and cross-attention depending on the context provided.

#     Args:
#         embed_dim (int): The dimensionality of the embedding space.
#         hidden_dim (int): The dimensionality of the hidden states.
#         context_dim (int, optional): The dimensionality of the context for cross-attention. 
#                                      If None, self-attention is performed.
#         num_heads (int, optional): The number of attention heads. Default is 1.
#     """
#     def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
#         super().__init__()
#         self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
#         if context_dim is None:
#             self.self_attn = True
#             self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
#             self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         else:
#             self.self_attn = False
#             self.key = nn.Linear(context_dim, embed_dim, bias=False)
#             self.value = nn.Linear(context_dim, hidden_dim, bias=False)

#     def forward(self, tokens, context=None):
#         if self.self_attn:
#             Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
#         else:
#             Q, K, V = self.query(tokens), self.key(context), self.value(context)
        
#         scoremats = torch.einsum('bth,bsh->bts', Q, K)
#         attnmats = F.softmax(scoremats, dim=1)
#         ctx_vecs = torch.einsum("bts,bsh->bth", attnmats, V)
#         return ctx_vecs

# class TransformerBlock(nn.Module):
#     """
#     Implements a Transformer block that includes self-attention, cross-attention, 
#     and a feed-forward network with normalization layers.

#     Args:
#         hidden_dim (int): The dimensionality of the hidden states.
#         context_dim (int): The dimensionality of the context for cross-attention.
#     """
#     def __init__(self, hidden_dim, context_dim):
#         super().__init__()
#         self.attn_self = Attention(hidden_dim, hidden_dim)
#         self.attn_cross = Attention(hidden_dim, hidden_dim, context_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, 4*hidden_dim),
#             nn.GELU(),
#             nn.Linear(4*hidden_dim, hidden_dim),
#             nn.GELU()
#         )

#     def forward(self, x, context=None):
#         x = self.attn_self(self.norm1(x)) + x
#         x = self.attn_cross(self.norm2(x), context=context) + x
#         x = self.ffn(self.norm3(x)) + x
#         return x

# class SpatialCrossAttention(nn.Module):
#     """
#     Implements a Spatial Cross Attention that applies a Transformer block to spatial data, 
#     typically images. This allows spatial interactions within the Transformer architecture.

#     Args:
#         hidden_dim (int): The dimensionality of the hidden states.
#         context_dim (int): The dimensionality of the context for cross-attention.
#     """
#     def __init__(self, hidden_dim, context_dim):
#         super().__init__()
#         self.transformer = TransformerBlock(hidden_dim, context_dim)

#     def forward(self, x, context=None):
#         b, c, h, w = x.shape
#         x_in = x
#         x = rearrange(x, "b c h w -> b (h w) c")
#         x = self.transformer(x, context)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
#         return x + x_in

class Attention(nn.Module):
    """
    Implements multi-head attention mechanism. Supports both self-attention and cross-attention.

    Args:
        embed_dim (int): The dimensionality of the embedding space.
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int, optional): The dimensionality of the context for cross-attention. 
                                     If None, self-attention is performed.
        num_heads (int): The number of attention heads.
    """
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, tokens, context=None):
        batch_size = tokens.size(0)
        if self.self_attn:
            Q, K, V = self.query(tokens), self.key(tokens), self.value(tokens)
        else:
            Q, K, V = self.query(tokens), self.key(context), self.value(context)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scoremats = torch.einsum('bqhd,bkhd->bqhk', Q, K)
        attnmats = F.softmax(scoremats / (self.head_dim ** 0.5), dim=-1)  # Scale dot-product attention
        ctx_vecs = torch.einsum('bqhk,bvhd->bqhd', attnmats, V)
        ctx_vecs = ctx_vecs.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(ctx_vecs)
        return output
    
class TransformerBlock(nn.Module):
    """
    Implements a Transformer block with multi-head self-attention, cross-attention, 
    and a feed-forward network with normalization layers.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
        num_heads (int): The number of attention heads.
    """
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super().__init__()
        self.attn_self = Attention(hidden_dim, hidden_dim, num_heads=num_heads)
        self.attn_cross = Attention(hidden_dim, hidden_dim, context_dim=context_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x, context=None):
        x = self.attn_self(self.norm1(x)) + x
        x = self.attn_cross(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x
    
class SpatialCrossAttention(nn.Module):
    """
    Implements a Spatial Cross Attention with multi-head attention applied to spatial data, 
    typically images.

    Args:
        hidden_dim (int): The dimensionality of the hidden states.
        context_dim (int): The dimensionality of the context for cross-attention.
        num_heads (int): The number of attention heads.
    """
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads=num_heads)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x + x_in

class UNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512], embed_dim=256, num_class=10, context_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(embed_dim, "sinusoidal"),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.act = nn.GELU()
        
        # Considtional embedding
        self.cond_embedding = nn.Embedding(num_class, context_dim)
        
        # Shared dropout
        self.dropout = nn.Dropout(p=0.1)

        # Encoding blocks
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.atten2 = SpatialCrossAttention(channels[1], context_dim)

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.atten3 = SpatialCrossAttention(channels[2], context_dim)
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.atten4 = SpatialCrossAttention(channels[3], context_dim)

        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=1, padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[4])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4])
        self.atten5 = SpatialCrossAttention(channels[4], context_dim)

        # Decoding blocks
        self.tconv5 = nn.ConvTranspose2d(channels[4], channels[3], 3, stride=1, padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.dense8 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.dense9 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1, padding=1)


    def forward(self, x, t, y=None):
        # Embed time
        t_emb = self.act(self.time_embedding(t))
        y_emb = self.cond_embedding(y).unsqueeze(1)

        # Encoding
        h1 = self.dropout(self.act(self.gnorm1(self.conv1(x) + self.dense1(t_emb))))
        h2 = self.dropout(self.act(self.gnorm2(self.conv2(h1) + self.dense2(t_emb))))
        h2 = self.atten2(h2, y_emb)
        h3 = self.dropout(self.act(self.gnorm3(self.conv3(h2) + self.dense3(t_emb))))
        h3 = self.atten3(h3, y_emb)
        h4 = self.dropout(self.act(self.gnorm4(self.conv4(h3) + self.dense4(t_emb))))
        h4 = self.atten4(h4, y_emb)
        h5 = self.dropout(self.act(self.gnorm5(self.conv5(h4) + self.dense5(t_emb))))
        h5 = self.atten5(h5, y_emb)

        # Decoding
        h = self.dropout(self.act(self.tgnorm5(self.tconv5(h5) + self.dense6(t_emb))))
        h = self.dropout(self.act(self.tgnorm4(self.tconv4(h + h4) + self.dense7(t_emb))))
        h = self.dropout(self.act(self.tgnorm3(self.tconv3(h + h3) + self.dense8(t_emb))))
        h = self.dropout(self.act(self.tgnorm2(self.tconv2(h + h2) + self.dense9(t_emb))))
        h = self.tconv1(h+h1)

        return h