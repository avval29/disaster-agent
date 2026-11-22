"""Semantic search over ingested guideline chunks."""
from __future__ import annotations

from typing import Any

from app.rag import embeddings
from app.rag.store import get_collection
from app.security.injection_guard import looks_like_injection, sanitize_text

HAZARDS = {"flood", "heatwave", "earthquake"}


def indexed_count() -> int:
    return get_collection().count()


def is_indexed() -> bool:
    return indexed_count() > 0


def search(query: str, hazard: str | None = None, k: int = 4) -> list[dict[str, Any]]:
    """Top-k chunks for a query. With a known hazard, searches that hazard plus 'general'.

    Each hit carries 'suspicious' = True if the chunk looks like a prompt injection.
    A smaller 'distance' means a closer match.
    """
    collection = get_collection()
    total = collection.count()
    if total == 0 or not query.strip():
        return []

    kwargs: dict[str, Any] = {}
    if hazard in HAZARDS:
        kwargs["where"] = {"hazard": {"$in": [hazard, "general"]}}
    res = collection.query(
        query_embeddings=embeddings.embed([query]),
        n_results=min(max(k, 1), total),
        include=["documents", "metadatas", "distances"],
        **kwargs,
    )
    hits: list[dict[str, Any]] = []
    for id_, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "id": id_,
            "text": sanitize_text(doc, 1200),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "hazard": meta.get("hazard"),
            "distance": round(float(dist), 4),
            "suspicious": looks_like_injection(doc),
        })
    return hits