from src.ml.models.transformer import TransformerBlock
import torch
import torch.nn as nn
from src.ml.models.embeddings import TextEmbeddings
class TextRetriever(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, d_ff, num_layers,max_seq_len):
        super().__init__()
        self.embedder = TextEmbeddings(vocab_size, d_model, max_seq_len)
        self.module_list = nn.ModuleList()

