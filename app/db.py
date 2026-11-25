"""SQLite schema, seeding and query helpers."""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    name TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    population INTEGER NOT NULL,
    flood_prone INTEGER NOT NULL DEFAULT 0,
    heat_vulnerable INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shelters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    zone TEXT NOT NULL REFERENCES zones(name),
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    capacity INTEGER NOT NULL,
    occupancy INTEGER NOT NULL DEFAULT 0,
    has_medical INTEGER NOT NULL DEFAULT 0,
    flood_safe INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    zone TEXT NOT NULL REFERENCES zones(name),
    quantity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    region TEXT,
    hazard_type TEXT,
    severity TEXT,
    status TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS alert_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    zone TEXT,
    language TEXT,
    channel TEXT,
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'SIMULATED_SENT'
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS api_cache (
    key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_alert_once
    ON alert_outbox (run_id, zone, language, channel);
    
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- seeding
def _read_csv(name: str) -> list[dict[str, str]]:
    path = settings.seed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Seed file missing: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("zones", "shelters", "resources")
    }


def seed_db(force: bool = False) -> dict[str, int]:
    """Load seed CSVs. Does nothing if data already exists unless force=True."""
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        if existing and not force:
            return _counts(conn)
        if force:
            conn.execute("DELETE FROM resources")
            conn.execute("DELETE FROM shelters")
            conn.execute("DELETE FROM zones")

        for r in _read_csv("zones.csv"):
            conn.execute(
                "INSERT INTO zones VALUES (?,?,?,?,?,?)",
                (r["name"], float(r["lat"]), float(r["lon"]), int(r["population"]),
                 int(r["flood_prone"]), int(r["heat_vulnerable"])),
            )
        for r in _read_csv("shelters.csv"):
            conn.execute(
                "INSERT INTO shelters (name, zone, lat, lon, capacity, occupancy, has_medical, flood_safe) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (r["name"], r["zone"], float(r["lat"]), float(r["lon"]), int(r["capacity"]),
                 int(r["occupancy"]), int(r["has_medical"]), int(r["flood_safe"])),
            )
        for r in _read_csv("resources.csv"):
            conn.execute(
                "INSERT INTO resources (type, name, zone, quantity) VALUES (?,?,?,?)",
                (r["type"], r["name"], r["zone"], int(r["quantity"])),
            )
        return _counts(conn)


# ---------------------------------------------------------------- queries
def get_zones() -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM zones ORDER BY name")]


def get_shelters(zone: str | None = None, only_available: bool = False) -> list[dict[str, Any]]:
    sql = (
        "SELECT id, name, zone, lat, lon, capacity, occupancy, "
        "capacity - occupancy AS available, has_medical, flood_safe FROM shelters"
    )
    clauses: list[str] = []
    params: list[Any] = []
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    if only_available:
        clauses.append("capacity - occupancy > 0")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY zone, name"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def get_resources(zone: str | None = None, rtype: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT id, type, name, zone, quantity FROM resources"
    clauses: list[str] = []
    params: list[Any] = []
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    if rtype:
        clauses.append("type = ?")
        params.append(rtype)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY type, zone"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


# ---------------------------------------------------------------- cache
def cache_set(key: str, payload: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (key, fetched_at, payload) VALUES (?,?,?)",
            (key, time.time(), json.dumps(payload)),
        )


def cache_get(key: str, max_age_s: int) -> tuple[dict[str, Any], float] | None:
    """Return (payload, age_seconds) if present and not older than max_age_s."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT fetched_at, payload FROM api_cache WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    age = time.time() - row["fetched_at"]
    if age > max_age_s:
        return None
    return json.loads(row["payload"]), age


# ---------------------------------------------------------------- audit
def log_event(run_id: str | None, actor: str, event: str, detail: Any = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, run_id, actor, event, detail) VALUES (?,?,?,?,?)",
            (now_iso(), run_id, actor, event,
             json.dumps(detail, default=str) if detail is not None else None),
        )


if __name__ == "__main__":
    init_db()
    print("Database ready:", settings.db_path)
    print("Seeded:", seed_db())