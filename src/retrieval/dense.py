from typing import List
from src.ml.inference.search_pipeline import NeuralSearchPipeline
from src.database.vector_db import QdrantManager
from src.ml.models.reranker import CrossEncoderReranker


class AdvancedRetriever:
    """
    Orchestrates the Two-Stage Retrieval pipeline:
    Stage 1: Fast ANN search using Bi-Encoder and Qdrant.
    Stage 2: Precision re-ranking using Cross-Encoder.
    """

    def __init__(
            self,
            search_pipeline: NeuralSearchPipeline,
            vector_db: QdrantManager,
            reranker: CrossEncoderReranker = None
    ):
        self.pipeline = search_pipeline
        self.db = vector_db
        self.reranker = reranker

    def retrieve(
            self,
            query: str,
            collection_name: str,
            top_k_retrieve: int = 50,
            top_k_rerank: int = 5
    ) -> List[str]:
        """
        Executes end-to-end search with optional reranking.
        """
        query_vectors = self.pipeline.embed_batch([query])
        target_vector = query_vectors[0]

        initial_candidates = self.db.search(
            collection_name=collection_name,
            query_vector=target_vector,
            limit=top_k_retrieve
        )

        if not initial_candidates or not self.reranker:
            return initial_candidates[:top_k_rerank]

        final_results = self.reranker.rank(
            query=query,
            documents=initial_candidates,
            top_k=top_k_rerank
        )

        return final_results