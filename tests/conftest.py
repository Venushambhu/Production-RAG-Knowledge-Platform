import hashlib
import math
import re

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class KeywordEmbedding(Embeddings):
    """Hashed bag-of-words embedding: texts sharing words get high cosine similarity.
    Deterministic and offline, so relevance-threshold logic can be tested for real."""

    def __init__(self, size: int = 4096):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def vectorstore(tmp_path):
    return Chroma(collection_name="test", embedding_function=KeywordEmbedding(),
                  persist_directory=str(tmp_path / "chroma"),
                  collection_metadata={"hnsw:space": "cosine"})
@pytest.fixture(autouse=True)
def _isolate_from_local_env(monkeypatch):
    """Tests must not depend on the developer's .env: environment variables override
    .env values, so pin the settings that would load real models."""
    monkeypatch.setenv("USE_RERANKER", "false")