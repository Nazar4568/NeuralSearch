from src.ml.models.transformer import TransformerBlock
import torch
import torch.nn as nn
from src.ml.models.embeddings import TextEmbeddings


class TextRetriever(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, d_ff, num_layers,max_seq_len):
        super().__init__()
        self.embedder = TextEmbeddings(vocab_size, d_model, max_seq_len)
        self.module_list = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, input_ids: torch.Tensor,attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Passes the tokenized input through the Embedding layer and multiple Transformer blocks.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape [batch_size, seq_len].
            attention_mask: torch.Tensor

        Returns:
            torch.Tensor: The final hidden states of the tokens of shape [batch_size, seq_len, d_model].
        """
        x = self.embedder(input_ids)
        for layer in self.module_list:
            x, _ = layer(x, mask=attention_mask)
        x = self.mean_pooling(x,attention_mask)
        return x

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        final_embeddings = sum_embeddings / sum_mask

        return final_embeddings
