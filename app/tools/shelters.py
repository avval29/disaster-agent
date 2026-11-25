"""Shelter planning tools: hazard scope, sheltering need, eligible shelters, allocation.

All arithmetic lives here so it is deterministic and testable. The Planner LLM only
proposes; the Critic re-checks against the same database and functions.
"""
from __future__ import annotations

from typing import Any

from app import db
from app.config import settings
from app.tools.base import haversine_km


def scope_zones(hazard: str) -> list[dict[str, Any]]:
    """Zones affected by a hazard type (floods: flood-prone, heat: heat-vulnerable, quake: all)."""
    zones = db.get_zones()
    if hazard == "flood":
        return [z for z in zones if z["flood_prone"]]
    if hazard == "heatwave":
        return [z for z in zones if z["heat_vulnerable"]]
    if hazard == "earthquake":
        return zones
    return []


def compute_needs(hazard: str, severity: str) -> list[dict[str, Any]]:
    """People per zone assumed to need shelter. Zero for severities without a fraction."""
    fraction = settings.at_risk_fraction.get(hazard, {}).get(severity, 0.0)
    return [
        {"zone": z["name"], "population": z["population"],
         "need": int(round(z["population"] * fraction))}
        for z in scope_zones(hazard)
    ]


def eligible_shelters(hazard: str) -> list[dict[str, Any]]:
    """Shelters with free places; for floods only flood-safe ones."""
    shelters = db.get_shelters(only_available=True)
    if hazard == "flood":
        shelters = [s for s in shelters if s["flood_safe"]]
    return shelters


def allocate(
    needs: list[dict[str, Any]], shelters: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Greedy allocation: biggest need first; same-zone shelters first, then nearest.

    Returns (assignments, unmet_by_zone). In an assignment, 'zone' is the zone where
    the evacuees live, not where the shelter is.
    """
    zones = {z["name"]: z for z in db.get_zones()}
    remaining = {s["id"]: s["available"] for s in shelters}
    assignments: list[dict[str, Any]] = []
    unmet: dict[str, int] = {}

    for item in sorted(needs, key=lambda n: -n["need"]):
        left = item["need"]
        if left <= 0:
            continue
        z = zones[item["zone"]]
        ranked = sorted(
            shelters,
            key=lambda s: (s["zone"] != item["zone"],
                           haversine_km(z["lat"], z["lon"], s["lat"], s["lon"])),
        )
        for s in ranked:
            if left <= 0:
                break
            take = min(left, remaining[s["id"]])
            if take > 0:
                assignments.append({"zone": item["zone"], "shelter_id": s["id"],
                                    "shelter_name": s["name"], "people": take})
                remaining[s["id"]] -= take
                left -= take
        if left > 0:
            unmet[item["zone"]] = left
    return assignments, unmet