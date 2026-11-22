"""Shared tool infrastructure: HTTP with retries, result envelope, cache fallback."""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import requests

from app import db
from app.config import settings

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Raised when a tool cannot obtain valid data."""


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any] | None = None
    source: str = "none"              # "live" | "cache" | "none"
    error: str | None = None
    latency_ms: int = 0
    cache_age_s: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with timeout, retries and exponential backoff.

    Client errors (4xx except 429) are not retried because retrying cannot fix them.
    """
    last_exc: Exception | None = None
    for attempt in range(1, settings.http_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=settings.http_timeout_s)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise ToolError(f"client error HTTP {status}") from exc
            last_exc = exc
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if attempt < settings.http_retries:
            time.sleep(settings.http_backoff_s * 2 ** (attempt - 1))
    raise ToolError(f"request failed after {settings.http_retries} attempts: {last_exc}")


def _simulated_failures() -> set[str]:
    """Demo switch: SIMULATE_API_FAILURE=weather,quake forces those tools to fail."""
    raw = os.getenv("SIMULATE_API_FAILURE", "")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def call_with_fallback(
    tool: str, cache_key: str, fetcher: Callable[[], dict[str, Any]]
) -> ToolResult:
    """Run a fetcher. On failure, serve recent cached data if available."""
    start = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    try:
        if tool in _simulated_failures():
            raise ToolError(f"simulated failure of '{tool}'")
        data = fetcher()
        db.cache_set(cache_key, data)
        return ToolResult(True, tool, data, "live", None, elapsed())
    except (ToolError, KeyError, TypeError, ValueError, IndexError) as exc:
        logger.warning("Tool '%s' live fetch failed: %s", tool, exc)
        hit = db.cache_get(cache_key, settings.cache_max_age_s)
        if hit is not None:
            data, age = hit
            return ToolResult(
                True, tool, data, "cache",
                f"live fetch failed ({exc}); served cached data",
                elapsed(), int(age),
            )
        return ToolResult(False, tool, None, "none", str(exc), elapsed())


def invalid_input(tool: str, message: str) -> ToolResult:
    return ToolResult(False, tool, None, "none", f"invalid input: {message}", 0)


def validate_coords(lat: Any, lon: Any) -> None:
    for name, val, lo, hi in (("latitude", lat, -90, 90), ("longitude", lon, -180, 180)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number")
        if not lo <= val <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))