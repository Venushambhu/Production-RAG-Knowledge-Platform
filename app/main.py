"""FastAPI service.

Endpoints
  GET  /health   status, number of indexed chunks, active configuration
  POST /ingest   upload PDF/MD/TXT files -> chunk, embed, store, refresh keyword index
  POST /query    single-shot RAG: one question, no memory (simple integrations, evaluation)
  POST /chat     multi-turn chat through the LangGraph agent, memory per session_id
Interactive docs (Swagger UI) are generated automatically at /docs.
"""
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agent import build_agent, run_turn
from app.config import get_settings
from app.ingest import SUPPORTED, ingest_paths
from app.providers import get_llm, get_reranker, get_vectorstore
from app.rag import answer
from app.retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")
state: dict = {}  # objects created once at startup and shared by all requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (before `yield`) and once at shutdown (after it).
    Loading models here, not per request, is what keeps requests fast."""
    s = get_settings()
    state["settings"] = s
    state["vectorstore"] = get_vectorstore(s)
    state["retriever"] = HybridRetriever(state["vectorstore"], s, get_reranker(s))
    state["llm"] = get_llm(s)
    state["agent"] = build_agent(state["retriever"], state["llm"], s)
    logger.info("Ready: llm=%s retrieval=%s reranker=%s", s.llm_provider, s.retrieval_mode, s.use_reranker)
    yield
    state.clear()


app = FastAPI(title="RAG Document Assistant", version="0.3.0", lifespan=lifespan)


# ---- request / response schemas: FastAPI validates input and documents the API from these ----
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    id: int
    source: str | None
    page: int | None
    snippet: str
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class ChatResponse(QueryResponse):
    route: str
    search_query: str


def _llm_unavailable(exc: Exception) -> HTTPException:
    """Turn a crash in the LLM call into a clear 503 instead of a raw 500."""
    logger.exception("LLM call failed")
    return HTTPException(503, f"LLM backend unavailable ({state['settings'].llm_provider}): "
                              f"{type(exc).__name__}: {str(exc)[:200]}")


@app.get("/health")
def health():
    s = state["settings"]
    return {"status": "ok", "chunks_indexed": state["vectorstore"]._collection.count(),
            "llm_provider": s.llm_provider, "retrieval_mode": s.retrieval_mode,
            "reranker": s.use_reranker}


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...)):
    s = state["settings"]
    # Uploaded files are written to a temporary folder that is deleted afterwards.
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in files:
            if Path(f.filename).suffix.lower() not in SUPPORTED:
                raise HTTPException(400, f"Unsupported file: {f.filename}. Allowed: {sorted(SUPPORTED)}")
            dest = Path(tmp) / Path(f.filename).name   # .name strips any "../" path tricks
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append(dest)
        result = ingest_paths(paths, state["vectorstore"], s.chunk_size, s.chunk_overlap)
    state["retriever"].refresh()   # new chunks must also enter the BM25 keyword index
    return result


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return answer(state["retriever"], state["llm"], req.question, req.top_k)
    except Exception as exc:
        raise _llm_unavailable(exc)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return run_turn(state["agent"], req.session_id, req.message)
    except Exception as exc:
        raise _llm_unavailable(exc)
