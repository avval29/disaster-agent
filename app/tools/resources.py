"""Resource planning tools: which resource types matter per hazard, stock, fallback dispatch."""
from __future__ import annotations

import math
from typing import Any

from app import db
from app.config import settings

RELEVANT: dict[str, list[str]] = {
    "flood": ["rescue_boat", "ambulance", "relief_kit", "generator", "medical_team"],
    "heatwave": ["water_tanker", "ambulance", "medical_team"],
    "earthquake": ["ambulance", "medical_team", "relief_kit", "generator"],
}


def stock_map() -> dict[tuple[str, str], int]:
    """{(resource_type, zone): quantity}."""
    stock: dict[tuple[str, str], int] = {}
    for r in db.get_resources():
        key = (r["type"], r["zone"])
        stock[key] = stock.get(key, 0) + r["quantity"]
    return stock


def fallback_dispatch(
    hazard: str, severity: str, needs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rule-based dispatch: a severity-dependent share of each relevant resource.

    Stock inside the affected zones stays there; stock outside is sent to the zone with
    the greatest need.
    """
    share = settings.resource_share.get(severity, 0.0)
    if share <= 0 or not needs:
        return []
    scope = {n["zone"] for n in needs}
    busiest = max(needs, key=lambda n: n["need"])["zone"]
    out: list[dict[str, Any]] = []
    for r in db.get_resources():
        if r["type"] not in RELEVANT.get(hazard, []):
            continue
        qty = math.ceil(r["quantity"] * share)
        if qty <= 0:
            continue
        out.append({
            "resource_type": r["type"],
            "from_zone": r["zone"],
            "to_zone": r["zone"] if r["zone"] in scope else busiest,
            "quantity": qty,
        })
    return out