"""Phase 3 retrieval: vector search + BM25 keyword search, fused with RRF, optionally re-ranked.

Why hybrid?
  - Vector (semantic) search matches MEANING: "time off" finds "annual leave".
  - BM25 (keyword) search matches exact TERMS: "EUR 500", policy codes, product names,
    rare words that embedding models blur together.
  Combining them recovers chunks that either one alone would miss.

Pipeline for search(query, k):
  1. vector_search  -> top fetch_k chunks with cosine similarity scores
  2. keyword_search -> top fetch_k chunks with BM25 scores          (hybrid mode only)
  3. reciprocal_rank_fusion -> one merged ranking                   (hybrid mode only)
  4a. with re-ranker:   cross-encoder scores candidates, keep score >= rerank_threshold, top k
  4b. without:          relevance gate on the best cosine score, then top k
Returned Documents carry their scores in metadata (useful for debugging and the UI).
"""
import re
import threading

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.ingest import chunk_id

# Very common words carry no signal for keyword search, so we drop them.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and",
    "or", "do", "does", "did", "i", "you", "we", "it", "my", "can", "what", "how", "when",
    "who", "which", "with", "at", "by", "from", "as", "this", "that", "much", "many", "get",
}


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, stopwords removed: 'Claims above EUR 500' -> ['claims','above','eur','500']"""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several ranked lists into one.

    Each document gets  sum over lists of 1 / (k + rank)   (rank starts at 1).
    Only ranks are used, never raw scores, which is why RRF can combine cosine similarity
    (0..1) and BM25 (0..~20) without any normalisation. k=60 damps the influence of the
    very top ranks so one list cannot dominate.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = chunk_id(doc)                       # same chunk from both lists -> same key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key in docs:
                docs[key].metadata.update(doc.metadata)   # keep both vector_score and bm25_score
            else:
                docs[key] = doc
    ordered = sorted(scores, key=scores.get, reverse=True)
    for key in ordered:
        docs[key].metadata["rrf_score"] = round(scores[key], 5)
    return [docs[key] for key in ordered]


class HybridRetriever:
    def __init__(self, vectorstore, settings: Settings, reranker=None):
        self.vs = vectorstore
        self.s = settings
        self.reranker = reranker
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []
        self._indexed_count = -1
        self._lock = threading.Lock()                 # FastAPI serves requests on several threads

    # ---- keyword index -------------------------------------------------------------------
    def refresh(self) -> None:
        """(Re)build the BM25 index from every chunk currently stored in Chroma.
        BM25 lives in memory, so it is rebuilt after ingestion and on first use."""
        with self._lock:
            data = self.vs.get(include=["documents", "metadatas"])
            self._bm25_docs = [Document(page_content=t, metadata=m or {})
                               for t, m in zip(data["documents"], data["metadatas"])]
            corpus = [tokenize(d.page_content) for d in self._bm25_docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._indexed_count = len(self._bm25_docs)

    def _ensure_index(self) -> None:
        # Cheap check: rebuild only if the number of chunks changed since the last build.
        if self._indexed_count != self.vs._collection.count():
            self.refresh()

    # ---- the two searches ----------------------------------------------------------------
    def vector_search(self, query: str, n: int) -> list[Document]:
        # Chroma's cosine space returns a DISTANCE (0 = identical, 2 = opposite).
        # cosine similarity = 1 - distance. Computing it ourselves avoids LangChain's
        # "relevance scores must be between 0 and 1" warning when a chunk is unrelated (< 0).
        results = self.vs.similarity_search_with_score(query, k=n)
        docs = []
        for doc, distance in results:
            doc.metadata["vector_score"] = round(1.0 - distance, 4)
            docs.append(doc)
        return docs

    def keyword_search(self, query: str, n: int) -> list[Document]:
        self._ensure_index()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)       # one score per stored chunk
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        out = []
        for i in best:
            if scores[i] <= 0:                        # no query term appears in this chunk
                break
            doc = Document(page_content=self._bm25_docs[i].page_content,
                           metadata={**self._bm25_docs[i].metadata, "bm25_score": round(float(scores[i]), 4)})
            out.append(doc)
        return out

    # ---- main entry point ----------------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.s.top_k
        vector_docs = self.vector_search(query, self.s.fetch_k)

        if self.s.retrieval_mode == "hybrid":
            candidates = reciprocal_rank_fusion(
                [vector_docs, self.keyword_search(query, self.s.fetch_k)], k=self.s.rrf_k)
        else:
            candidates = vector_docs

        if not candidates:
            return []

        if self.reranker is not None:
            # The cross-encoder is the relevance judge: it both re-orders and filters.
            scores = self.reranker.score(query, candidates)
            for doc, sc in zip(candidates, scores):
                doc.metadata["rerank_score"] = round(float(sc), 4)
            ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
            return [d for d, sc in ranked if sc >= self.s.rerank_threshold][:k]

        # No re-ranker: use the best cosine similarity to decide whether the question is
        # covered by the documents at all. BM25 alone can't decide this, because a common
        # word like "policy" matches something in almost any corpus.
        best_cosine = max((d.metadata.get("vector_score", -1.0) for d in vector_docs), default=-1.0)
        if best_cosine < self.s.relevance_threshold:
            return []
        if self.s.retrieval_mode == "vector":
            candidates = [d for d in candidates if d.metadata["vector_score"] >= self.s.relevance_threshold]
        return candidates[:k]
