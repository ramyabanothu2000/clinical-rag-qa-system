"""
main.py
-------
FastAPI application entry point for the Clinical RAG Q&A System.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.chain.rag_chain import ClinicalRAGChain
from src.embeddings.embedder import get_embeddings
from src.vectorstore.store import VectorStoreManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
rag_chain: Optional[ClinicalRAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global rag_chain
    logger.info("Initializing RAG chain...")
    try:
        embeddings = get_embeddings()
        store = VectorStoreManager(
            embeddings=embeddings,
            backend=os.getenv("VECTOR_STORE", "faiss"),
            index_name=os.getenv("PINECONE_INDEX", "clinical-notes"),
        )
        store.load()
        retriever = store.as_retriever(k=5)
        rag_chain = ClinicalRAGChain(
            retriever=retriever,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
        )
        logger.info("RAG chain ready.")
    except Exception as e:
        logger.error("Failed to initialize RAG chain: %s", e)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinical RAG Q&A API",
    description="HIPAA-conscious retrieval-augmented generation for clinical notes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="What medications is the patient currently taking?")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")


class SourceDoc(BaseModel):
    content: str
    source: str
    note_type: str
    encounter_date: str


class QueryResponse(BaseModel):
    answer: str
    source_docs: List[SourceDoc]
    num_sources: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok" if rag_chain else "degraded",
        "model": os.getenv("LLM_MODEL", "gpt-4o"),
        "vector_store": os.getenv("VECTOR_STORE", "faiss"),
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not initialized. Check server logs.",
        )
    start = time.perf_counter()
    try:
        result = rag_chain.query(request.question)
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info("Query answered in %.1fms | sources=%d", latency_ms, result["num_sources"])

    return QueryResponse(
        answer=result["answer"],
        source_docs=result["source_docs"],
        num_sources=result["num_sources"],
        latency_ms=latency_ms,
    )


@app.post("/api/v1/reset")
async def reset_memory():
    if rag_chain:
        rag_chain.reset_memory()
    return {"status": "memory cleared"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
