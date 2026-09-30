"""Load documents, split into chunks, and upsert into Chroma with stable IDs."""
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    """One Document per PDF page (keeps page numbers for citations), one per text file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=text, metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [Document(page_content=text, metadata={"source": path.name, "page": 1})] if text else []
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    """Content-based ID: re-ingesting the same file updates chunks instead of duplicating them."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def ingest_paths(paths: list[Path], vectorstore, chunk_size: int, chunk_overlap: int,
                 batch_size: int = 500) -> dict:
    docs: list[Document] = []
    skipped: list[str] = []
    for p in paths:
        if p.suffix.lower() not in SUPPORTED:
            skipped.append(p.name)
            continue
        docs.extend(load_file(p))

    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    # dict keeps one chunk per ID, removing exact duplicates within one batch
    unique = {chunk_id(c): c for c in chunks}
    items = list(unique.items())
    # Chroma rejects very large single inserts, so add in batches (a 100-page PDF can give 1000+ chunks)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        vectorstore.add_documents([c for _, c in batch], ids=[i for i, _ in batch])
    return {"files": len(paths) - len(skipped), "pages": len(docs),
            "chunks": len(unique), "skipped": skipped}
