# Reference : https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
# Reference : https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    """
    Examples :
        >>> attention = MultiheadAttention(embed_dim, heads)
        >>> output, weights = attention(query, key, value)
    """
    def __init__(self, num_heads, model_dim, key_dim = None, value_dim = None):
        self.num_heads = num_heads
        self.model_dim = model_dim
        assert self.model_dim%self.num_heads == 0, "model_dim must be divisible by num_heads"
        self.head_dim = self.model_dim//self.num_heads

        self.key_dim = model_dim if key_dim is None else key_dim
        self.value_dim = model_dim if key_dim is None else value_dim

        self.query = nn.Parameter(torch.randn(self.model_dim, self.model_dim))
        self.key = nn.Parameter(torch.randn(self.model_dim, self.key_dim))
        self.value = nn.Parameter(torch.randn(self.model_dim, self.value_dim))

    def forward(self, query, key, value):
        """
        Inputs
            - query : (target sequence length, batch size, embedding dimension
            - key : (source sequence length, batch size, embedding dimension)
            - value : (source sequence length, batch size, embedding dimension)

        Outputs
            - outputs : (target sequence length, batch size, embedding dimension
            - weights : (batch size, target sequence length, source sequence length)
        """







