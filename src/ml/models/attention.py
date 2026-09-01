import torch.nn as nn
import torch

import math
from typing import Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the input tensor through the Multi-Head Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            mask: torch.Tensor = None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Context vector of shape [batch_size, seq_len, d_model].
                - Attention weights of shape [batch_size, num_heads, seq_len, seq_len].
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1) == 0.,value =-1e9)

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context_vector = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out_context_vector = self.out_proj(context_vector)
        return out_context_vector, attn_weights