"""The cross-encoder must return 0..1 scores regardless of sentence-transformers version.
A stub model replaces the real one, so no download is needed."""
import pytest
from langchain_core.documents import Document

from app.providers import CrossEncoderReranker

RAW = [6.93, -1.65, -11.25]   # real logits observed on the sample docs


class NewStyleModel:                     # accepts activation_fn
    def predict(self, pairs, activation_fn):
        return RAW[: len(pairs)]


class OldStyleModel:                     # only accepts activation_fct
    def predict(self, pairs, activation_fct=None, **kwargs):
        if kwargs:
            raise TypeError("unexpected keyword")
        return RAW[: len(pairs)]


@pytest.mark.parametrize("model", [NewStyleModel(), OldStyleModel()])
def test_scores_are_sigmoid_probabilities(model):
    reranker = object.__new__(CrossEncoderReranker)   # skip __init__ (no model download)
    reranker.model = model
    docs = [Document(page_content=t) for t in ("a", "b", "c")]
    scores = reranker.score("q", docs)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > 0.99 and 0.1 < scores[1] < 0.2 and scores[2] < 0.001


def test_empty_candidates():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.model = NewStyleModel()
    assert reranker.score("q", []) == []
