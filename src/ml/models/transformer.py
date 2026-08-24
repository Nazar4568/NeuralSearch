import torch
import torch.nn as nn
from src.ml.models.attention import MultiHeadAttention
from typing import Tuple
class FeedForward(nn.Module):
    """
    Implements the Feed-Forward Network (FFN) component of a Transformer block.

    This network expands the input dimension to a higher space, applies a
    non-linear activation (GELU), and then projects it back to the original dimension.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the feed-forward layers.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self,d_model: int,num_heads: int, d_ff: int):
        super().__init__()

        self.attention = MultiHeadAttention(d_model,num_heads)
        self.FeedForward = FeedForward(d_model,d_ff)

        self.norm1 =nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the input tensor through a single Transformer block using Pre-LN architecture.

        This block applies Layer Normalization, Multi-Head Attention, and a Feed-Forward
        Network, connected via residual (skip) connections.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The updated hidden states of shape [batch_size, seq_len, d_model].
                - The attention weights from the Multi-Head Attention layer.
        """
        x =  self.norm1(x)
        context_vector, attn_weights =  self.attention(x)
        x = x + context_vector

        x = self.norm2(x)
        ffn_output = self.FeedForward(x)
        x = x + ffn_output

        return x,attn_weights
