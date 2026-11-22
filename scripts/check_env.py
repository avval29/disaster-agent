"""Verifies that dependencies, API key, and external data APIs are reachable."""
import os
import sys
import importlib

import requests
from dotenv import load_dotenv

load_dotenv()

REQUIRED = [
    "langgraph", "langchain_core", "langchain_google_genai",
    "chromadb", "streamlit", "pydantic", "requests", "pandas", "pypdf",
]


def check_imports() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            print(f"[OK]   import {name}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] import {name}: {exc}")
    return ok


def check_env() -> bool:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key or key == "your_key_here":
            print("[FAIL] GOOGLE_API_KEY is not set in .env")
            return False
        print("[OK]   GOOGLE_API_KEY is set (value not displayed)")
    else:
        print(f"[OK]   LLM_PROVIDER={provider} (no API key required)")
    return True


def check_url(label: str, url: str, params: dict | None = None) -> bool:
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        print(f"[OK]   {label} reachable (HTTP {r.status_code})")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}\n")
    results = [
        check_imports(),
        check_env(),
        check_url(
            "Open-Meteo weather API",
            "https://api.open-meteo.com/v1/forecast",
            {"latitude": 28.61, "longitude": 77.21, "daily": "precipitation_sum", "forecast_days": 1},
        ),
        check_url(
            "USGS earthquake feed",
            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
        ),
    ]
    print()
    if all(results):
        print("Environment ready.")
        return 0
    print("Some checks failed. Fix the items marked [FAIL] before continuing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())