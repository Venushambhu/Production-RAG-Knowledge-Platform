from pathlib import Path

from langchain_core.documents import Document

from app.ingest import chunk_documents, chunk_id, ingest_paths

DOCS = Path("data/sample_docs")


def test_chunks_respect_size_and_keep_metadata():
    doc = Document(page_content="word " * 1000, metadata={"source": "a.md", "page": 1})
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "a.md" for c in chunks)


def test_chunk_id_is_stable():
    c = Document(page_content="hello", metadata={"source": "a.md", "page": 1})
    assert chunk_id(c) == chunk_id(Document(page_content="hello", metadata={"source": "a.md", "page": 1}))


def test_reingest_does_not_duplicate(vectorstore):
    paths = sorted(DOCS.glob("*.md"))
    first = ingest_paths(paths, vectorstore, 300, 50)
    ingest_paths(paths, vectorstore, 300, 50)
    assert vectorstore._collection.count() == first["chunks"]


def test_unsupported_files_are_skipped(vectorstore, tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("x")
    result = ingest_paths([bad], vectorstore, 300, 50)
    assert result["skipped"] == ["notes.docx"] and result["chunks"] == 0
