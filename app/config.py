"""Central settings. Loads .env once; no other module should read it directly."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    seed_dir: Path = BASE_DIR / "data" / "seed"
    docs_dir: Path = BASE_DIR / "data" / "docs"
    db_path: Path = BASE_DIR / "data" / "disaster.sqlite"
    chroma_dir: Path = BASE_DIR / "data" / "chroma_store"

    # LLM (repr=False keeps the key out of logs and printouts)
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    google_api_key: str = field(default=os.getenv("GOOGLE_API_KEY", ""), repr=False)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # External tool behaviour
    http_timeout_s: int = _env_int("HTTP_TIMEOUT_S", 15)
    http_retries: int = _env_int("HTTP_RETRIES", 3)
    http_backoff_s: float = 1.0
    cache_max_age_s: int = _env_int("CACHE_MAX_AGE_S", 6 * 3600)

    # Demo location
    demo_city: str = "Delhi"
    demo_lat: float = 28.6139
    demo_lon: float = 77.2090

    # Agent behaviour
    max_replans: int = 3
    # Planning assumptions (illustrative, NOT official data; state this in the report)
    at_risk_fraction: dict = field(default_factory=lambda: {
        "flood": {"High": 0.0003, "Critical": 0.001},
        "heatwave": {"High": 0.0001, "Critical": 0.0002},
        "earthquake": {"High": 0.0002, "Critical": 0.0004},
    })
    resource_share: dict = field(default_factory=lambda: {"High": 0.5, "Critical": 1.0})


settings = Settings()