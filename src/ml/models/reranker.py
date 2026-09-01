import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List


class CrossEncoderReranker(nn.Module):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, queries: List[str], documents: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        outputs = self.model(**encoded)
        return outputs.logits.squeeze(-1)

    def rank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        self.eval()
        queries = [query] * len(documents)

        with torch.no_grad():
            scores = self.forward(queries, documents)

        scored_docs = list(zip(scores.tolist(), documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored_docs[:top_k]]