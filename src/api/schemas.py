from pydantic import BaseModel, Field
from typing import List

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User search query string")
    top_k_retrieve: int = Field(default=50, ge=10, le=100, description="Number of documents to retrieve from vector database (Recall phase)")
    top_k_rerank: int = Field(default=5, ge=1, le=20, description="Number of top documents to keep after reranking (Precision phase)")
    collection_name: str = Field(default="documents", description="Target collection name in Qdrant")

class SearchResponse(BaseModel):
    query: str
    retrieved_documents: List[str]
    llm_answer: str

class IndexRequest(BaseModel):
    documents: List[str] = Field(..., min_items=1, description="List of text documents to be indexed")
    collection_name: str = Field(default="documents", description="Target Qdrant collection name for indexing")
