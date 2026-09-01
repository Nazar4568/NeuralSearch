import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List

class QdrantManager:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)

    def create_collection(self, collection_name: str, vector_size: int = 384) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )


    def upsert_documents(self, collection_name: str, texts: List[str], vectors: List[List[float]], ids: List[int]):
        points = [
            PointStruct(id=idx, vector=vector, payload={"text": text})
            for idx, text, vector in zip(ids, texts, vectors)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[str]:
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )
        return [hit.payload["text"] for hit in response.points if hit.payload]