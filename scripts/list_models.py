"""Lists Gemini models available to your API key."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402


def main() -> int:
    if not settings.google_api_key:
        print("GOOGLE_API_KEY is not set in .env")
        return 1
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": settings.google_api_key},
        params={"pageSize": 200},
        timeout=20,
    )
    if resp.status_code != 200:
        print(f"Request failed: HTTP {resp.status_code}")
        return 1
    names = sorted(
        m["name"].removeprefix("models/")
        for m in resp.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "gemini" in m["name"]
    )
    print("Models supporting generateContent:")
    for n in names:
        print("  ", n)
    print("\nPick a current 'flash' model and set GEMINI_MODEL=<name> in .env")
    return 0


if __name__ == "__main__":
    sys.exit(main())