import torch
import torch.nn as nn
from .mlp import MLP
from .ssa import SSA

class Block(nn.Module):
    """
    Transformer block with spiking neurons.
    
    This module combines a Spike Self-Attention (SSA) layer and a Multi-Layer Perceptron (MLP)
    with spiking neurons. The block processes input through normalization, attention, and MLP layers.
    The output is a residual connection of the input with the processed features.
    The module processes input tensors with shape (T, B, N, C) where T is time steps,
    B is batch size, N is sequence length, and C is feature dimension.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x 