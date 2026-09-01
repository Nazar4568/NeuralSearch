import hashlib
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import IndexRequest

router = APIRouter()


@router.post("/index")
async def index_documents(request: IndexRequest, req: Request):
    pipeline = req.app.state.pipeline
    db = req.app.state.db

    try:
        vectors = pipeline.embed_batch(request.documents)
        ids = [
            int(hashlib.md5(text.encode('utf-8')).hexdigest()[:15], 16)
            for text in request.documents
        ]
        db.upsert_documents(
            collection_name=request.collection_name,
            texts=request.documents,
            vectors=vectors,
            ids=ids
        )
        return {"status": "success", "indexed_count": len(request.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))