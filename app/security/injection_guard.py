"""Prompt-injection defences for text that originates outside the system.

Phase 8 extends this module. For now it covers retrieved documents and any
LLM-generated text that is fed back into a later prompt.
"""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules)",
        r"\b(reveal|print|show|output|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions|api\s*key)",
        r"\bnew\s+instructions?\s*:",
        r"<\s*/?\s*(system|data|assistant)\s*>",
        r"\boverride\s+(the\s+)?(safety|approval|human)",
    )
]
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DELIMITER_TAGS = re.compile(r"<\s*(/?)\s*(data|system|assistant)\s*>", re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


def sanitize_text(text: str, max_len: int = 2000) -> str:
    """Strip control characters, neutralise our prompt delimiters, and cap the length."""
    text = _CONTROL_CHARS.sub("", text)
    text = _DELIMITER_TAGS.sub(r"[\1\2]", text)
    return text[:max_len]