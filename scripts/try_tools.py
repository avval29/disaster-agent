"""Manual demo: live calls, then a simulated API failure to show cache fallback."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.tools.quake import get_recent_earthquakes  # noqa: E402
from app.tools.weather import get_weather_forecast  # noqa: E402


def show(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"ok={result.ok} source={result.source} latency={result.latency_ms}ms "
          f"cache_age_s={result.cache_age_s}")
    if result.error:
        print("note:", result.error)
    if result.data:
        body = result.data.get("summary") or {k: v for k, v in result.data.items() if k != "events"}
        print(json.dumps(body, indent=2, default=str))


def main() -> None:
    db.init_db()
    db.seed_db()
    lat, lon = settings.demo_lat, settings.demo_lon

    os.environ.pop("SIMULATE_API_FAILURE", None)
    show("Weather (live)", get_weather_forecast(lat, lon))
    show("Earthquakes within 300 km, last 24h (live)", get_recent_earthquakes(lat, lon))

    print("\n--- Simulating a weather API outage ---")
    os.environ["SIMULATE_API_FAILURE"] = "weather"
    show("Weather (outage, should fall back to cache)", get_weather_forecast(lat, lon))
    os.environ.pop("SIMULATE_API_FAILURE", None)


if __name__ == "__main__":
    main()