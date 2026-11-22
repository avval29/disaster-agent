"""Simulated hazard scenarios for demos and tests.

Observations are produced by feeding synthetic raw payloads through the *same
parsers* the live tools use, so their shape is identical to real data. The
quiet live conditions are not a limitation: scenarios let you replay events
reproducibly. Always label runs on this data as simulated.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from app.config import settings
from app.tools.quake import _parse as _parse_quake
from app.tools.weather import _parse as _parse_weather


def _weather(past, ahead, tmax, prob, feels_extra: float = 3.0, tmin: float = 22.0) -> dict[str, Any]:
    """past: 2 daily rain values (mm); ahead/tmax/prob: 3 values for the forecast days."""
    today = date.today()
    dates = [(today + timedelta(days=i - 2)).isoformat() for i in range(5)]
    tmaxs = [tmax[0]] * 2 + list(tmax)
    payload = {
        "timezone": "Asia/Kolkata",
        "current": {"temperature_2m": tmax[0] - 4, "relative_humidity_2m": 55, "precipitation": 0.0},
        "daily": {
            "time": dates,
            "precipitation_sum": list(past) + list(ahead),
            "precipitation_probability_max": [0, 0] + list(prob),
            "temperature_2m_max": tmaxs,
            "temperature_2m_min": [tmin] * 5,
            "apparent_temperature_max": [t + feels_extra for t in tmaxs],
        },
    }
    return _parse_weather(payload, settings.demo_lat, settings.demo_lon)


def _quake(events: list[tuple[float, float, float, float, str]]) -> dict[str, Any]:
    """events: (magnitude, lat, lon, depth_km, place)."""
    event_ms = int((datetime.now(timezone.utc) - timedelta(minutes=30)).timestamp() * 1000)
    features = [
        {
            "properties": {"mag": mag, "place": place, "time": event_ms,
                           "tsunami": 0, "alert": None, "url": None},
            "geometry": {"coordinates": [lon, lat, depth]},
        }
        for mag, lat, lon, depth, place in events
    ]
    query = {"radius_km": 300, "hours": 24, "min_magnitude": 2.5}
    return _parse_quake({"features": features}, settings.demo_lat, settings.demo_lon, query)


def _calm() -> dict[str, Any]:
    return _weather([0, 0], [0, 0, 0], [28, 29, 29], [0, 0, 5])


_DEFS: dict[str, tuple[str, Callable[[], dict[str, Any]]]] = {
    "calm": (
        "Normal day: no rain, mild temperatures, no earthquakes.",
        lambda: {"weather": _calm(), "quake": _quake([])},
    ),
    "moderate_rain": (
        "Moderate rain forecast (48 mm peak day), little antecedent rain.",
        lambda: {"weather": _weather([2, 5], [48, 20, 5], [31, 30, 30], [80, 60, 30]),
                 "quake": _quake([])},
    ),
    "flood_heavy_rain": (
        "Heavy monsoon rain forecast (95 mm peak day) on already-wet ground.",
        lambda: {"weather": _weather([10, 20], [95, 80, 30], [29, 28, 29], [92, 88, 60]),
                 "quake": _quake([])},
    ),
    "flood_extreme": (
        "Extremely heavy rain forecast (210 mm peak day) after 100 mm in two days.",
        lambda: {"weather": _weather([40, 60], [210, 150, 60], [27, 27, 28], [98, 95, 80]),
                 "quake": _quake([])},
    ),
    "heatwave_high": (
        "Heatwave: maximum temperature reaching 45 C.",
        lambda: {"weather": _weather([0, 0], [0, 0, 0], [40, 43, 45], [0, 0, 0], feels_extra=3.0),
                 "quake": _quake([])},
    ),
    "heatwave_severe": (
        "Severe heatwave: 47 C with a much higher heat index.",
        lambda: {"weather": _weather([0, 0], [0, 0, 0], [44, 46, 47], [0, 0, 0], feels_extra=5.0),
                 "quake": _quake([])},
    ),
    "earthquake_m6": (
        "M6.4 earthquake about 40 km from Delhi, 30 minutes ago (simulated).",
        lambda: {"weather": _calm(),
                 "quake": _quake([(6.4, 28.95, 77.05, 10.0, "Near Sonipat, Haryana (simulated)")])},
    ),
    "earthquake_distant_m4": (
        "M4.4 earthquake in the Himalayan foothills, about 270 km away (simulated).",
        lambda: {"weather": _calm(),
                 "quake": _quake([(4.4, 30.9, 78.2, 15.0, "Uttarakhand foothills (simulated)")])},
    ),
}

SCENARIOS: list[str] = list(_DEFS)


def describe_scenarios() -> dict[str, str]:
    return {name: desc for name, (desc, _) in _DEFS.items()}


def get_scenario(name: str) -> dict[str, Any]:
    """Return {'name','description','weather','quake'}. Raises KeyError if unknown."""
    if name not in _DEFS:
        raise KeyError(name)
    description, build = _DEFS[name]
    return {"name": name, "description": description, **build()}