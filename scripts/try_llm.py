"""Sends one tiny structured request to the configured LLM."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> int:
    model = settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model
    print(f"Provider: {settings.llm_provider} | model: {model}")
    start = time.perf_counter()
    try:
        reply = llm.ask_json("You are a test assistant.", "Reply with the message 'pong'.", llm.Ping)
    except llm.LLMUnavailable as exc:
        print(f"[FAIL] {exc}")
        print("Tip: run scripts/list_models.py and check GEMINI_MODEL in .env")
        return 1
    print(f"[OK] reply={reply.message!r} in {int((time.perf_counter() - start) * 1000)} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())