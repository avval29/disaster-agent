"""LLM provider wrapper.

Agents call ask_json() and never import a provider SDK directly, so switching
between Gemini and a local Ollama model is a .env change. Any failure surfaces
as LLMUnavailable so callers can fall back to rule-based behaviour.
"""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 3
BACKOFF_S = 2.0


class LLMUnavailable(Exception):
    """The LLM is not configured or failed after all retries."""


class Ping(BaseModel):
    message: str


def is_configured() -> bool:
    provider = settings.llm_provider
    if provider == "gemini":
        key = settings.google_api_key
        return bool(key) and key != "your_key_here"
    if provider == "ollama":
        return True
    return False


def _build_model():
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, temperature=0)
    raise LLMUnavailable(f"unsupported LLM provider '{settings.llm_provider}'")


def _structured_call(system: str, user: str, schema: type[T]) -> T:
    model = _build_model().with_structured_output(schema)
    result = model.invoke([("system", system), ("human", user)])
    if isinstance(result, dict):
        result = schema.model_validate(result)
    if not isinstance(result, schema):
        raise ValueError("model returned no valid structured output")
    return result


def ask_json(system: str, user: str, schema: type[T]) -> T:
    """Ask the LLM for output matching a Pydantic schema, with retries and backoff."""
    if not is_configured():
        raise LLMUnavailable("LLM provider is not configured")

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _structured_call(system, user, schema)
        except LLMUnavailable:
            raise
        except Exception as exc:  # network, quota, malformed output, etc.
            last_exc = exc
            logger.warning("LLM attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, type(exc).__name__)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * 2 ** (attempt - 1))
    detail = f"{type(last_exc).__name__}: {str(last_exc)[:200]}"
    raise LLMUnavailable(f"LLM call failed after {MAX_ATTEMPTS} attempts ({detail})") from last_exc