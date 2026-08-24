import torch.nn as nn
from transformers import AutoModel
import torch

class TextEmbeddings(nn.Module):
    def __init__(self,vocab_size: int, d_model: int, max_seq_len: int = 512,
                 dropout_rate: float = 0.1, padding_idx: int = 0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

        self.LayerNorm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,input_ids: torch.Tensor):
        """
        Converts input token IDs into dense vectors, combining word embeddings
        and learned positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor of token indices of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Embedded and normalized tensor of shape [batch_size, seq_len, d_model].
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)

        position_ids = self.position_embeddings(position_ids)
        position_ids = position_ids.unsqueeze(0)
        position_ids_embedded = self.word_embeddings(input_ids)
        ids = position_ids_embedded + position_ids
        ids_normalized = self.LayerNorm(ids)
        ids_final = self.dropout(ids_normalized)

        return ids_final