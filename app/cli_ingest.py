"""Bulk-ingest a folder from the command line (run this while the API is STOPPED):

    python -m app.cli_ingest data/sample_docs
"""
import sys
from pathlib import Path

from app.config import get_settings
from app.ingest import ingest_paths
from app.providers import get_vectorstore

if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs")
    s = get_settings()
    paths = [p for p in folder.rglob("*") if p.is_file()]
    print(ingest_paths(paths, get_vectorstore(s), s.chunk_size, s.chunk_overlap))
