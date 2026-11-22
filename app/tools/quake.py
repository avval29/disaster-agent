"""USGS earthquake tool (no API key required)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.tools.base import (
    ToolResult, call_with_fallback, fetch_json, haversine_km, invalid_input, validate_coords,
)

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MAX_EVENTS_RETURNED = 5


def _parse(payload: dict[str, Any], lat: float, lon: float, query: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for feature in payload["features"]:
        props = feature["properties"]
        mag = props.get("mag")
        if mag is None:
            continue
        elon, elat = feature["geometry"]["coordinates"][:2]
        depth = feature["geometry"]["coordinates"][2] if len(feature["geometry"]["coordinates"]) > 2 else None
        ms = props.get("time")
        events.append({
            "magnitude": float(mag),
            "place": props.get("place"),
            "time_utc": datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None,
            "depth_km": depth,
            "distance_km": round(haversine_km(lat, lon, elat, elon), 1),
            "tsunami": bool(props.get("tsunami")),
            "pager_alert": props.get("alert"),
            "url": props.get("url"),
        })
    events.sort(key=lambda e: e["magnitude"], reverse=True)
    return {
        "query": query,
        "count": len(events),
        "max_magnitude": events[0]["magnitude"] if events else None,
        "strongest": events[0] if events else None,
        "tsunami_flag": any(e["tsunami"] for e in events),
        "events": events[:MAX_EVENTS_RETURNED],
    }


def get_recent_earthquakes(
    lat: float, lon: float, radius_km: float = 300, hours: int = 24, min_magnitude: float = 2.5
) -> ToolResult:
    """Recent earthquakes near a point. Zero events is a valid, successful result."""
    try:
        validate_coords(lat, lon)
        if not 1 <= radius_km <= 2000:
            raise ValueError("radius_km must be between 1 and 2000")
        if not 1 <= hours <= 168:
            raise ValueError("hours must be between 1 and 168")
        if not 0 <= min_magnitude <= 9:
            raise ValueError("min_magnitude must be between 0 and 9")
    except ValueError as exc:
        return invalid_input("quake", str(exc))

    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": 50,
    }
    query = {"radius_km": radius_km, "hours": hours, "min_magnitude": min_magnitude}
    cache_key = f"quake:{lat:.2f}:{lon:.2f}:{radius_km}:{hours}:{min_magnitude}"
    return call_with_fallback(
        "quake", cache_key, lambda: _parse(fetch_json(USGS_URL, params), lat, lon, query)
    )