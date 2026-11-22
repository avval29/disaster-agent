"""Open-Meteo weather tool (no API key required)."""
from __future__ import annotations

from typing import Any

from app.tools.base import (
    ToolError, ToolResult, call_with_fallback, fetch_json, invalid_input, validate_coords,
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = [
    "precipitation_sum",
    "precipitation_probability_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
]
PAST_DAYS = 2       # antecedent rainfall matters for flood risk
FORECAST_DAYS = 3   # today + next two days


def _sum(values: list[float | None]) -> float:
    return round(sum(v for v in values if v is not None), 1)


def _max(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _parse(payload: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    daily = payload["daily"]
    dates = daily["time"]
    n = len(dates)
    for var in DAILY_VARS:
        if len(daily[var]) != n:
            raise ToolError(f"malformed weather response: '{var}' length mismatch")
    if n < PAST_DAYS + 1:
        raise ToolError("malformed weather response: too few days")

    days = [
        {
            "date": dates[i],
            "is_forecast": i >= PAST_DAYS,
            "rain_mm": daily["precipitation_sum"][i],
            "rain_prob_pct": daily["precipitation_probability_max"][i],
            "tmax_c": daily["temperature_2m_max"][i],
            "tmin_c": daily["temperature_2m_min"][i],
            "feels_like_max_c": daily["apparent_temperature_max"][i],
        }
        for i in range(n)
    ]
    past, ahead = days[:PAST_DAYS], days[PAST_DAYS:][:FORECAST_DAYS]
    current = payload.get("current") or {}

    return {
        "location": {"lat": lat, "lon": lon},
        "timezone": payload.get("timezone"),
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
        },
        "summary": {
            "past_2d_rain_mm": _sum([d["rain_mm"] for d in past]),
            "today_rain_mm": ahead[0]["rain_mm"] or 0.0,
            "next_3d_rain_mm": _sum([d["rain_mm"] for d in ahead]),
            "max_temp_next_3d_c": _max([d["tmax_c"] for d in ahead]),
            "max_feels_like_next_3d_c": _max([d["feels_like_max_c"] for d in ahead]),
            "max_rain_prob_next_3d_pct": _max([d["rain_prob_pct"] for d in ahead]),
        },
        "daily": days,
    }


def get_weather_forecast(lat: float, lon: float) -> ToolResult:
    """Past 2 days + 3-day forecast summary for a location."""
    try:
        validate_coords(lat, lon)
    except ValueError as exc:
        return invalid_input("weather", str(exc))

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "auto",
    }
    cache_key = f"weather:{lat:.2f}:{lon:.2f}"
    return call_with_fallback(
        "weather", cache_key, lambda: _parse(fetch_json(OPEN_METEO_URL, params), lat, lon)
    )