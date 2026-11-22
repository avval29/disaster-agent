"""Embedding function used by ingestion and search (local model, no API key).

The first real call downloads a small ONNX model (about 80 MB) once. Tests
replace embed() with a fake, so they run offline.
"""
from __future__ import annotations

_default = None


def embed(texts: list[str]) -> list[list[float]]:
    global _default
    if _default is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _default = DefaultEmbeddingFunction()
    return [[float(x) for x in vec] for vec in _default(texts)]