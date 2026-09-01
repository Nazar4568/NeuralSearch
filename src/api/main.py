from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from src.ml.inference.search_pipeline import NeuralSearchPipeline
from src.database.vector_db import QdrantManager
from src.retrieval.dense import AdvancedRetriever
from src.ml.models.reranker import CrossEncoderReranker
from src.api.routes import search, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML models and connecting to DB...")

    app.state.pipeline = NeuralSearchPipeline(
        vocab_size=30522, d_model=384, num_heads=12, d_ff=1536,
        num_layers=2, max_seq_len=512, save_path="models/retriever_weights.pth"
    )

    app.state.db = QdrantManager()
    app.state.db.create_collection(collection_name="documents")

    print("Loading Cross-Encoder model...")
    app.state.reranker = CrossEncoderReranker()

    app.state.retriever = AdvancedRetriever(
        search_pipeline=app.state.pipeline,
        vector_db=app.state.db,
        reranker=app.state.reranker
    )
    print("System is ready.")

    yield

    print("Shutting down, freeing ML resources...")
    app.state.pipeline = None
    app.state.db = None
    app.state.reranker = None
    app.state.retriever = None


app = FastAPI(title="Neural RAG API", lifespan=lifespan)
app.include_router(search.router, tags=["Search"])
app.include_router(documents.router, tags=["Indexing"])