"""Phase 3 retrieval tests. KeywordEmbedding (conftest.py) makes vector scores meaningful offline."""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.ingest import ingest_paths
from app.retrieval import HybridRetriever, reciprocal_rank_fusion, tokenize


def doc(text, source="a.md"):
    return Document(page_content=text, metadata={"source": source, "page": 1})


@pytest.fixture
def store(vectorstore):
    ingest_paths(sorted(Path("data/sample_docs").glob("*.md")), vectorstore, 300, 50)
    return vectorstore


def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("What is the limit for claims above EUR 500?") == ["limit", "claims", "above", "eur", "500"]


def test_rrf_rewards_documents_found_by_both_lists():
    a, b, c = doc("alpha"), doc("beta"), doc("gamma")
    fused = reciprocal_rank_fusion([[a, b], [c, b]], k=60)
    # b is 2nd in both lists: 2/(62) beats a or c, which are 1st in only one list: 1/(61)
    assert fused[0].page_content == "beta"
    assert {d.page_content for d in fused} == {"alpha", "beta", "gamma"}


def test_keyword_search_finds_exact_terms(store):
    r = HybridRetriever(store, Settings())
    hits = r.keyword_search("Security Operations Centre", 3)
    assert hits[0].metadata["source"] == "it_security_policy.md"
    assert hits[0].metadata["bm25_score"] > 0


def test_hybrid_search_returns_relevant_chunks_with_scores(store):
    r = HybridRetriever(store, Settings(retrieval_mode="hybrid", relevance_threshold=0.1, top_k=2))
    hits = r.search("How long must passwords be?")
    assert hits and hits[0].metadata["source"] == "it_security_policy.md"
    assert "rrf_score" in hits[0].metadata


def test_out_of_scope_question_returns_nothing(store):
    r = HybridRetriever(store, Settings(relevance_threshold=0.2))
    assert r.search("zebra quantum chromodynamics") == []


def test_vector_mode_skips_bm25(store):
    r = HybridRetriever(store, Settings(retrieval_mode="vector", relevance_threshold=0.1))
    hits = r.search("annual leave days")
    assert hits and all("bm25_score" not in d.metadata for d in hits)


class FakeReranker:
    """Scores 1.0 if the chunk contains the word 'passwords', else 0.0."""
    def score(self, query, docs):
        return [1.0 if "passwords" in d.page_content.lower() else 0.0 for d in docs]


def test_reranker_reorders_and_filters(store):
    r = HybridRetriever(store, Settings(rerank_threshold=0.5, top_k=4), reranker=FakeReranker())
    hits = r.search("security rules for laptops and passwords")
    assert hits and all("passwords" in d.page_content.lower() for d in hits)
    assert hits[0].metadata["rerank_score"] == 1.0


def test_index_refreshes_after_new_ingest(store, tmp_path):
    r = HybridRetriever(store, Settings())
    assert r.keyword_search("kombucha", 3) == []
    f = tmp_path / "kitchen.md"
    f.write_text("The office kitchen offers free kombucha on Fridays.")
    ingest_paths([f], store, 300, 50)
    assert r.keyword_search("kombucha", 3)[0].metadata["source"] == "kitchen.md"
