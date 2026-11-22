import pytest

from app import llm
from app.config import settings


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(llm, "BACKOFF_S", 0.0)


def test_retries_then_succeeds(llm_on, monkeypatch):
    calls = {"n": 0}

    def flaky(system, user, schema):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 quota")
        return llm.Ping(message="ok")

    monkeypatch.setattr(llm, "_structured_call", flaky)
    assert llm.ask_json("s", "u", llm.Ping).message == "ok"
    assert calls["n"] == 3


def test_exhausted_retries_raise_unavailable(llm_on, monkeypatch):
    calls = {"n": 0}

    def broken(system, user, schema):
        calls["n"] += 1
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_structured_call", broken)
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_json("s", "u", llm.Ping)
    assert calls["n"] == llm.MAX_ATTEMPTS


def test_not_configured_makes_no_call(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(llm, "_structured_call", boom)
    with pytest.raises(llm.LLMUnavailable):
        llm.ask_json("s", "u", llm.Ping)