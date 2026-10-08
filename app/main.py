import asyncio
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from common.config import SERVICE
from .search.qdrant import generate_embedding, load_documents, QdrantDense
from .search.split import chunk_law
from .search.loader import fetch_law_root
from .search.factory import embedding_client, structured_client, CACHE_DIR
from .search.retriever import Retriever
from .search.reranker import LLMReranker
from .search.pipeline import RAGPipeline
from common.qdrant import get_qdrant_client
from common.config import COLLECTION_NAME

app = FastAPI()
_index_lock = asyncio.Lock()


@app.get("/")
def root():
    return {"message": "RAG API"}


@app.get("/law/chunks")
async def preview_chunks(limit: int = Query(20, ge=1, le=500)):
    """인덱싱 전에 청킹 결과만 확인한다."""
    docs = chunk_law(await fetch_law_root())
    return {
        "total": len(docs),
        "chunks": [
            {"content": d.page_content, "metadata": d.metadata} for d in docs[:limit]
        ],
    }


@app.post("/index")
async def index_law():
    async with _index_lock:
        docs = chunk_law(await fetch_law_root(CACHE_DIR / 'law.xml'))
        indexed = await generate_embedding(docs, embeddings=embedding_client())
        return {"chunks": len(docs), "indexed": indexed}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('질문을 입력하세요.')
        return value.strip()


@app.post('/ask')
async def ask(request: AskRequest):
    # One client per request; no shared mutable retriever/query state.
    client = get_qdrant_client()
    try:
        async with _index_lock:
            docs = load_documents(client)
        if not docs:
            raise HTTPException(409, '먼저 POST /index로 법령을 저장하세요.')
        llm = structured_client()
        retriever = Retriever(docs,
            QdrantDense(client, embedding_client(), COLLECTION_NAME, str(SERVICE.mst)),
            LLMReranker(llm), law_version=str(SERVICE.mst))
        response, _ = await RAGPipeline(retriever, llm).ask(request.question)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        # External exceptions can contain authenticated URLs. Do not expose their text.
        raise HTTPException(502, f'검색 또는 생성 호출 실패 ({type(exc).__name__})') from None
    finally:
        client.close()
