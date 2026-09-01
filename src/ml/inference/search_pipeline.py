import torch
from typing import List
from src.ml.models.retriever import TextRetriever
from src.data.tokenizer import TextTokenizer


class NeuralSearchPipeline:
    """
    Pipeline for text embedding inference.
    Initializes tokenizer and model, loads weights, and provides a vectorized interface.
    """

    def __init__(self, save_path: str, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 max_seq_len: int, device: str = "cpu"):
        self.device = device
        self.tokenizer = TextTokenizer()

        self.model = TextRetriever(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

        self.model.load_state_dict(torch.load(save_path, map_location=self.device,weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Transforms raw text strings into dense vector embeddings using mini-batches.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded = self.tokenizer.batch_encode(batch_texts)

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                embeddings = self.model(input_ids, attention_mask)
                all_embeddings.append(embeddings)

        final_embeddings = torch.cat(all_embeddings, dim=0)

        return final_embeddings.cpu().tolist()
