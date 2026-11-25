"""Small reusable validators. Phase 8 extends this module."""
import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LINK = re.compile(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+", re.IGNORECASE)


def has_link(text: str) -> bool:
    """True if the text contains a URL or an email address."""
    return bool(LINK.search(text))


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text))