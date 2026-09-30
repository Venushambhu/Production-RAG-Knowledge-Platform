"""Central configuration.

Every setting can be overridden by an environment variable of the same name
(upper-case) or by a line in the .env file, e.g.  LLM_PROVIDER=openai
pydantic-settings reads them, converts types (str -> int/float/bool) and validates them.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM ------------------------------------------------------------------------
    llm_provider: str = "ollama"            # "ollama" (free, local) or "openai" (needs OPENAI_API_KEY)
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0                # 0 = deterministic, best for factual Q&A

    # ---- Embeddings -----------------------------------------------------------------
    embedding_provider: str = "huggingface" # "huggingface" or "fake" (tests only)
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Vector store + chunking ----------------------------------------------------
    chroma_dir: str = "./chroma_db"
    collection_name: str = "documents"
    chunk_size: int = 300                   # characters per chunk (300 beat 800 in our eval)
    chunk_overlap: int = 50                 # characters shared by neighbouring chunks

    # ---- Retrieval (Phase 3) --------------------------------------------------------
    top_k: int = 4                          # chunks given to the LLM
    retrieval_mode: str = "hybrid"          # "vector" or "hybrid" (vector + BM25 keyword search)
    fetch_k: int = 20                       # candidates pulled from each search before fusion
    rrf_k: int = 60                         # Reciprocal Rank Fusion constant (60 is the standard value)
    relevance_threshold: float = 0.2        # min cosine similarity for a question to count as "in the documents"
    use_reranker: bool = False              # cross-encoder re-ranking (downloads ~90 MB model on first use)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = 0.1           # min re-ranker probability (0-1); tuned with eval/run_eval.py

    # ---- Agent (Phase 2) ------------------------------------------------------------
    max_retrieval_retries: int = 1          # query reformulations before giving up
    max_history_messages: int = 6           # past messages used to rewrite follow-up questions


@lru_cache  # read the environment once, then reuse the same Settings object
def get_settings() -> Settings:
    return Settings()
