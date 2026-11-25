"""Ingest guideline documents into ChromaDB.

Usage:  python -m app.rag.ingest
File naming convention: <hazard>_<anything>.pdf|txt|md where hazard is
flood, heatwave or earthquake. Anything else is tagged 'general'.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
import logging
import sys

logging.getLogger("pypdf").setLevel(logging.CRITICAL)

from app.config import settings
from app.rag import embeddings
from app.rag.store import get_collection

SUPPORTED = {".pdf", ".txt", ".md"}
BATCH = 64


def infer_hazard(filename: str) -> str:
    name = filename.lower()
    if "earthquake" in name or "seismic" in name:
        return "earthquake"
    if "heat" in name:
        return "heatwave"
    if "flood" in name:
        return "flood"
    return "general"


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring to end on a sentence boundary."""
    if overlap >= size / 2:
        raise ValueError("overlap must be less than half of size")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(". ", start + int(size * 0.6), end)
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_document(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number_or_None, text), ...]."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def _chunk_id(source: str, page: int | None, index: int) -> str:
    return hashlib.sha1(f"{source}|{page}|{index}".encode()).hexdigest()[:24]


def ingest_all(docs_dir: Path | None = None) -> dict[str, Any]:
    docs_dir = docs_dir or settings.docs_dir
    report: dict[str, Any] = {"files": 0, "chunks": 0, "skipped": [], "details": []}
    if not docs_dir.exists():
        report["skipped"].append(f"{docs_dir}: folder not found")
        return report

    collection = get_collection()
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED:
            report["skipped"].append(f"{path.name}: unsupported file type")
            continue
        try:
            pages = load_document(path)
        except Exception as exc:  # corrupt or encrypted file
            report["skipped"].append(f"{path.name}: could not read ({type(exc).__name__})")
            continue

        hazard = infer_hazard(path.name)
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        for page_no, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(_chunk_id(path.name, page_no, i))
                docs.append(chunk)
                metas.append({"source": path.name, "page": page_no or 0, "hazard": hazard})
        if not docs:
            report["skipped"].append(f"{path.name}: no extractable text (scanned PDF?)")
            continue

        collection.delete(where={"source": path.name})   # replace, never duplicate
        for s in range(0, len(docs), BATCH):
            part = slice(s, s + BATCH)
            collection.add(
                ids=ids[part], documents=docs[part], metadatas=metas[part],
                embeddings=embeddings.embed(docs[part]),
            )
        report["files"] += 1
        report["chunks"] += len(docs)
        report["details"].append({"file": path.name, "hazard": hazard, "chunks": len(docs)})
        print(f"  ingested {path.name:<55} hazard={hazard:<10} chunks={len(docs)}")
    return report


def reset_index() -> None:
    """Delete the whole collection so stale chunks from old runs disappear."""
    import chromadb
    from app.rag.store import COLLECTION

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION)
        print("Existing index deleted.")
    except Exception:
        pass


def main() -> None:
    if "--reset" in sys.argv:
        reset_index()
    print(f"Ingesting from {settings.docs_dir}")
    report = ingest_all()
    print(f"\nFiles: {report['files']} | chunks: {report['chunks']}")
    for item in report["skipped"]:
        print("  skipped:", item)
    if report["files"] == 0:
        print("Nothing was indexed. Put PDF/TXT/MD files in data/docs/ (see the naming convention).")


if __name__ == "__main__":
    main()