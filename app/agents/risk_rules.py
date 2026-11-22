"""Deterministic, explainable risk scoring: observations -> 0..100 score per hazard.

THRESHOLDS ARE SIMPLIFIED AND ILLUSTRATIVE. Flood bands follow IMD rainfall
categories (heavy >= 64.5 mm/day, very heavy >= 115.6, extremely heavy >= 204.5);
heat bands are inspired by IMD heatwave criteria; earthquake bands are generic
magnitude/distance scaling. Verify against official documents before citing.
"""
from __future__ import annotations

from typing import Any

NOISE_FLOOR = 10.0   # below this, no meaningful hazard
SEVERITY_FLOOR = {"Low": 0.0, "Moderate": 25.0, "High": 50.0, "Critical": 75.0}

FLOOD_POINTS = [(0, 0), (15, 10), (64.5, 50), (115.6, 70), (204.5, 90), (300, 100)]   # mm/day
HEAT_POINTS = [(35, 0), (40, 35), (45, 65), (48, 90), (50, 100)]                       # deg C
QUAKE_POINTS = [(3.0, 0), (4.0, 15), (5.0, 40), (6.0, 70), (7.0, 95), (8.0, 100)]      # magnitude


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation, clamped at both ends."""
    if x <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(points[-1][1])


def severity_from_score(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Moderate"
    return "Low"


def flood_score(weather: dict[str, Any]) -> tuple[float, list[str]] | None:
    days = [d for d in weather.get("daily", []) if d.get("is_forecast")]
    if not days:
        return None
    peak_day = max(days, key=lambda d: d.get("rain_mm") or 0.0)
    peak = peak_day.get("rain_mm") or 0.0
    prob = peak_day.get("rain_prob_pct")

    score = _interp(peak, FLOOD_POINTS)
    factor = 1.0 if prob is None else 0.7 + 0.3 * min(max(prob, 0), 100) / 100
    score *= factor
    drivers = [f"peak forecast rainfall {peak:.0f} mm/day (probability {prob if prob is not None else 'n/a'}%)"]

    past = (weather.get("summary") or {}).get("past_2d_rain_mm") or 0.0
    if past >= 50:
        score += 10
        drivers.append(f"saturated ground: {past:.0f} mm in the past 2 days")
    elif past >= 25:
        score += 5
        drivers.append(f"wet ground: {past:.0f} mm in the past 2 days")
    return round(min(score, 100.0), 1), drivers


def heat_score(weather: dict[str, Any]) -> tuple[float, list[str]] | None:
    summary = weather.get("summary") or {}
    tmax = summary.get("max_temp_next_3d_c")
    if tmax is None:
        return None
    feels = summary.get("max_feels_like_next_3d_c")
    score = _interp(tmax, HEAT_POINTS)
    drivers = [f"maximum temperature {tmax:.0f} C in the next 3 days"]
    if feels is not None and feels - tmax >= 4:
        score += 5
        drivers.append(f"heat index {feels:.0f} C is well above air temperature")
    return round(min(score, 100.0), 1), drivers


def _distance_factor(distance_km: float) -> float:
    if distance_km <= 50:
        return 1.0
    if distance_km <= 150:
        return 0.75
    if distance_km <= 300:
        return 0.5
    return 0.25


def quake_score(quake: dict[str, Any]) -> tuple[float, list[str]] | None:
    if quake.get("count") is None:
        return None
    strongest = quake.get("strongest")
    if not quake["count"] or not strongest:
        return 0.0, ["no significant earthquakes in the search window"]
    mag = strongest["magnitude"]
    dist = strongest.get("distance_km") or 0.0
    score = _interp(mag, QUAKE_POINTS) * _distance_factor(dist)
    drivers = [f"strongest recent earthquake M{mag:.1f}, {dist:.0f} km away"]
    return round(min(score, 100.0), 1), drivers


def assess(observations: dict[str, Any]) -> dict[str, Any]:
    """Score every hazard we have data for and pick the primary one."""
    scores: dict[str, float | None] = {"flood": None, "heatwave": None, "earthquake": None}
    drivers: dict[str, list[str]] = {}
    missing: list[str] = []

    weather = observations.get("weather")
    quake = observations.get("quake")

    checks = (
        ("flood", flood_score, weather),
        ("heatwave", heat_score, weather),
        ("earthquake", quake_score, quake),
    )
    for name, fn, data in checks:
        result = fn(data) if data else None
        if result is None:
            missing.append(name)
        else:
            scores[name], drivers[name] = result

    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return {"scores": scores, "drivers": drivers, "missing": missing,
                "hazard_type": "none", "risk_score": 0.0, "severity": "Unknown"}

    primary = max(available, key=lambda k: available[k])   # ties keep declaration order
    top = available[primary]
    return {
        "scores": scores,
        "drivers": drivers,
        "missing": missing,
        "hazard_type": primary if top >= NOISE_FLOOR else "none",
        "risk_score": top,
        "severity": severity_from_score(top),
    }