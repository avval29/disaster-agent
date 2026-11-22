"""ChromaDB persistent collection holding guideline chunks."""
from __future__ import annotations

import chromadb

from app.config import settings

COLLECTION = "protocols"


def get_collection():
    """Embeddings are always supplied explicitly (see embeddings.py), so Chroma gets none."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(COLLECTION, embedding_function=None)