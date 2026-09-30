"""Factories that create external components (embeddings, vector store, LLM, re-ranker).

Keeping all vendor-specific imports here means the rest of the code only depends on
LangChain's generic interfaces (Embeddings, BaseChatModel). Switching from Ollama to
OpenAI is a config change, not a code change. Imports happen inside the functions so a
package you don't use (e.g. langchain_openai) never has to load.
"""
import math

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_embeddings(s: Settings) -> Embeddings:
    if s.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=s.hf_embedding_model,
            # unit-length vectors: dot product == cosine similarity
            encode_kwargs={"normalize_embeddings": True},
        )
    if s.embedding_provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=384)
    raise ValueError(f"Unknown embedding_provider: {s.embedding_provider}")


def get_vectorstore(s: Settings) -> Chroma:
    # "cosine" space makes relevance score == cosine similarity (range -1..1),
    # so relevance_threshold is easy to reason about.
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=get_embeddings(s),
        persist_directory=s.chroma_dir,         # data survives restarts
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_llm(s: Settings) -> BaseChatModel:
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=s.openai_model, temperature=s.temperature)  # reads OPENAI_API_KEY
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=s.ollama_model, temperature=s.temperature, base_url=s.ollama_base_url)
    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")


class CrossEncoderReranker:
    """Scores (question, chunk) pairs with a cross-encoder.

    A bi-encoder (our embedding model) embeds question and chunk separately and compares
    vectors: fast, but approximate. A cross-encoder reads question and chunk TOGETHER in one
    transformer pass, so it judges relevance much more accurately, but it is too slow to run
    on every chunk. Standard pattern: cheap search finds ~20 candidates, cross-encoder re-orders them.

    The model outputs raw logits (about -12 .. +12). Whether predict() converts them to 0..1
    depends on the sentence-transformers version, so we always ask for RAW logits and apply
    the sigmoid ourselves. Then rerank_threshold means the same thing on every machine.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def _raw_logits(self, pairs):
        identity = lambda x: x                 # "no activation": return the logits unchanged
        try:                                   # newer sentence-transformers
            return self.model.predict(pairs, activation_fn=identity)
        except TypeError:                      # older versions use a different argument name
            return self.model.predict(pairs, activation_fct=identity)

    def score(self, query: str, docs: list[Document]) -> list[float]:
        if not docs:
            return []
        logits = self._raw_logits([(query, d.page_content) for d in docs])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]   # sigmoid -> 0..1


def get_reranker(s: Settings):
    return CrossEncoderReranker(s.reranker_model) if s.use_reranker else None
