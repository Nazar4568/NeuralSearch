import json
import linecache
import subprocess
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Dataset
from src.data.tokenizer import TextTokenizer


class RetrievalDataset(Dataset):
    def __init__(self, filepath: str):
        self.filepath = filepath
        result = subprocess.run(['wc', '-l', filepath], capture_output=True, text=True)
        self.total_lines = int(result.stdout.split()[0])

    def __len__(self) -> int:
        return self.total_lines

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        line_number = idx + 1

        raw_line = linecache.getline(self.filepath, line_number)
        if not raw_line:
            raise IndexError(f"Line {line_number} is empty or out of bounds.")

        data = json.loads(raw_line)

        query = data.get("query", "")
        positive = data.get("positive_doc", "")
        negative = data.get("negative_doc", "")

        if not negative:
            negative = "dummy negative sequence"

        return query, positive, negative


class CollateFunction:
    def __init__(self, tokenizer: TextTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries, pos, neg = zip(*batch)

        query_tensors = self.tokenizer.batch_encode(list(queries))
        pos_tensors = self.tokenizer.batch_encode(list(pos))
        neg_tensors = self.tokenizer.batch_encode(list(neg))

        return {
            "query": query_tensors,
            "positive": pos_tensors,
            "negative": neg_tensors
        }