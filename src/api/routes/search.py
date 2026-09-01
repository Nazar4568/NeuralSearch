from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import SearchRequest, SearchResponse
from src.generation.llm import LLMGenerator

router = APIRouter()
llm_generator = LLMGenerator()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, req: Request):
    retriever = req.app.state.retriever

    try:
        documents = retriever.retrieve(
            query=request.query,
            collection_name=request.collection_name,
            top_k_retrieve=request.top_k_retrieve,
            top_k_rerank=request.top_k_rerank
        )

        llm_answer = await llm_generator.generate_answer(
            query=request.query,
            context_documents=documents
        )

        return SearchResponse(
            query=request.query,
            retrieved_documents=documents,
            llm_answer=llm_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))