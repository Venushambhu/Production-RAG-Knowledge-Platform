from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "-1")  # fake embeddings: accept every chunk
    from app import config, main
    config.get_settings.cache_clear()
    monkeypatch.setattr(main, "get_llm", lambda s: FakeListChatModel(
        responses=["documents", "Employees get 24 days [1]."]))
    with TestClient(main.app) as c:
        path = Path("data/sample_docs/leave_policy.md")
        c.post("/ingest", files=[("files", (path.name, path.read_bytes(), "text/markdown"))])
        yield c
    config.get_settings.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] > 0


def test_query(client):
    r = client.post("/query", json={"question": "How many annual leave days?"})
    assert r.status_code == 200 and r.json()["sources"][0]["source"] == "leave_policy.md"


def test_chat(client):
    r = client.post("/chat", json={"session_id": "abc", "message": "How many annual leave days?"})
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "documents" and body["answer"] == "Employees get 24 days [1]."


def test_llm_failure_returns_503(client, monkeypatch):
    from app import main

    def boom(*a, **k):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(main, "run_turn", boom)
    r = client.post("/chat", json={"session_id": "abc", "message": "hello"})
    assert r.status_code == 503 and "ConnectionError" in r.json()["detail"]


def test_rejects_unsupported_upload(client):
    r = client.post("/ingest", files=[("files", ("a.exe", b"x", "application/octet-stream"))])
    assert r.status_code == 400


def test_validates_question(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
