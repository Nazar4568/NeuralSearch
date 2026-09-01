import sys
import os
import json
import torch
import time
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ml.inference.search_pipeline import NeuralSearchPipeline
from src.database.vector_db import QdrantManager
from src.retrieval.dense import AdvancedRetriever


def load_test_data(filepath: str, max_samples: int = 1000):
    queries = []
    expected_docs = []
    corpus_dict = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            queries.append(data["query"])
            expected_docs.append(data["positive_doc"])

            corpus_dict[data["positive_doc"]] = True
            if data.get("negative_doc"):
                corpus_dict[data["negative_doc"]] = True

    return queries, expected_docs, list(corpus_dict.keys())


def main():
    print("Initializing Evaluation Pipeline...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pipeline = NeuralSearchPipeline(
        vocab_size=30522, d_model=384, num_heads=12, d_ff=1536,
        num_layers=2, max_seq_len=512, save_path="models/retriever_weights.pth", device=device
    )

    db = QdrantManager()
    collection_name = "eval_collection"

    db.create_collection(collection_name, vector_size=384)
    retriever = AdvancedRetriever(search_pipeline=pipeline, vector_db=db)

    data_path = os.path.join(project_root, "data", "processed", "code_triplets_test.jsonl")
    queries, expected_docs, corpus = load_test_data(data_path)

    print(f"Indexing {len(corpus)} documents...")
    corpus_vectors = pipeline.embed_batch(corpus, batch_size=32)

    db.upsert_documents(
        collection_name=collection_name,
        texts=corpus,
        vectors=corpus_vectors,
        ids=list(range(len(corpus)))
    )

    hits = {1: 0, 5: 0, 10: 0}
    start_time = time.time()

    for query, expected_doc in tqdm(zip(queries, expected_docs), total=len(queries)):
        retrieved = retriever.retrieve(
            query=query, collection_name=collection_name, top_k_retrieve=10, top_k_rerank=10
        )
        if expected_doc in retrieved[:1]: hits[1] += 1
        if expected_doc in retrieved[:5]: hits[5] += 1
        if expected_doc in retrieved[:10]: hits[10] += 1

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / len(queries)) * 1000

    n = len(queries)
    print("\n--- Evaluation Results ---")
    print(f"Recall@1:  {hits[1] / n:.4f}")
    print(f"Recall@5:  {hits[5] / n:.4f}")
    print(f"Recall@10: {hits[10] / n:.4f}")
    print(f"Average Latency per query: {avg_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()