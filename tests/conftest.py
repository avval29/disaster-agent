import pytest

from app import db
from app.config import settings


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own empty, seeded database and no retry delays."""
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.sqlite")
    monkeypatch.setattr(settings, "http_backoff_s", 0.0)
    monkeypatch.delenv("SIMULATE_API_FAILURE", raising=False)
    db.init_db()
    db.seed_db()
    yield