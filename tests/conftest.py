import hashlib
import math
import re

import pytest

from app import db
from app.config import settings
from app.rag import embeddings

FLOOD_TEXT = ("Flood evacuation guidelines. Move residents of low lying areas to higher shelters "
              "before the river crosses the danger level. ") * 3
HEAT_TEXT = ("Heatwave action plan. Open cooling centres, distribute drinking water and avoid "
             "outdoor work during afternoon hours. ") * 3
GENERAL_TEXT = ("General incident command. The district magistrate coordinates all response "
                "agencies and public communication. ") * 3


def fake_embed(texts):
    """Offline bag-of-words embedding: deterministic and good enough to test retrieval logic."""
    vectors = []
    for text in texts:
        vec = [0.0] * 128
        for token in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Own database, vector store and docs folder per test; no retry delays; no real LLM."""
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.sqlite")
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "docs_dir", tmp_path / "docs")
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(settings, "http_backoff_s", 0.0)
    monkeypatch.setattr(settings, "llm_provider", "none")
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.delenv("SIMULATE_API_FAILURE", raising=False)
    db.init_db()
    db.seed_db()
    yield


@pytest.fixture
def sample_docs():
    """Writes three small guideline files into the temporary docs folder."""
    def _write():
        d = settings.docs_dir
        (d / "flood_guide.txt").write_text(FLOOD_TEXT)
        (d / "heatwave_plan.md").write_text(HEAT_TEXT)
        (d / "general_command.txt").write_text(GENERAL_TEXT)
    return _write