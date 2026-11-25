"""Restricted action tools for the Executor: the only code that changes the world.

Every state-changing operation goes through call_tool(), which enforces:
  1. a per-caller allow-list (least privilege),
  2. strict argument validation (Pydantic, unknown fields forbidden),
  3. idempotency via the action ledger (a repeated call is a no-op),
  4. the caller's transaction (the Executor runs a whole stage atomically).

Alert delivery is SIMULATED: messages are written to alert_outbox and nothing is sent.
Never call audit/db helpers while a stage transaction is open (SQLite would lock).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app import db
from app.security.injection_guard import looks_like_injection
from app.security.validators import has_link


class ActionRejected(Exception):
    """Permanent failure: the action is invalid or unsafe. Never retried."""


class PermissionDenied(ActionRejected):
    """The caller is not allowed to use this tool."""


class TransientToolError(Exception):
    """Temporary failure: safe to retry because the stage transaction is rolled back."""


# ------------------------------------------------------------------ argument schemas
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReserveShelterArgs(_Args):
    shelter_id: int = Field(gt=0, strict=True)
    zone: str = Field(min_length=1, max_length=60)          # zone where the evacuees live
    people: int = Field(gt=0, le=100_000, strict=True)


class DispatchArgs(_Args):
    resource_type: str = Field(min_length=1, max_length=40)
    from_zone: str = Field(min_length=1, max_length=60)
    to_zone: str = Field(min_length=1, max_length=60)
    quantity: int = Field(gt=0, le=1_000_000, strict=True)


class SendAlertArgs(_Args):
    zone: str = Field(min_length=1, max_length=60)
    language: Literal["en", "hi"]
    channel: Literal["sms"] = "sms"
    message: str = Field(min_length=1, max_length=600)


# ------------------------------------------------------------------ the tools
def _require_zone(conn: sqlite3.Connection, zone: str) -> None:
    if conn.execute("SELECT 1 FROM zones WHERE name = ?", (zone,)).fetchone() is None:
        raise ActionRejected(f"unknown zone '{zone}'")


def _reserve_shelter(conn: sqlite3.Connection, run_id: str, a: ReserveShelterArgs) -> dict[str, Any]:
    _require_zone(conn, a.zone)
    row = conn.execute(
        "SELECT name, capacity, occupancy FROM shelters WHERE id = ?", (a.shelter_id,)
    ).fetchone()
    if row is None:
        raise ActionRejected(f"shelter {a.shelter_id} does not exist")
    left = row["capacity"] - row["occupancy"]
    if a.people > left:
        raise ActionRejected(f"{row['name']}: only {left} places left, {a.people} requested")
    cur = conn.execute(
        "UPDATE shelters SET occupancy = occupancy + ? WHERE id = ? AND occupancy + ? <= capacity",
        (a.people, a.shelter_id, a.people),
    )
    if cur.rowcount != 1:
        raise ActionRejected(f"{row['name']}: reservation lost a race for capacity")
    return {"shelter": row["name"], "people": a.people}


def _dispatch_resource(conn: sqlite3.Connection, run_id: str, a: DispatchArgs) -> dict[str, Any]:
    _require_zone(conn, a.from_zone)
    _require_zone(conn, a.to_zone)
    rows = conn.execute(
        "SELECT id, quantity FROM resources WHERE type = ? AND zone = ? ORDER BY id",
        (a.resource_type, a.from_zone),
    ).fetchall()
    total = sum(r["quantity"] for r in rows)
    if total < a.quantity:
        raise ActionRejected(
            f"{a.resource_type} in {a.from_zone}: {total} in stock, {a.quantity} requested")
    left = a.quantity
    for r in rows:
        take = min(left, r["quantity"])
        if take:
            conn.execute("UPDATE resources SET quantity = quantity - ? WHERE id = ?", (take, r["id"]))
        left -= take
        if left == 0:
            break
    return {"resource_type": a.resource_type, "quantity": a.quantity,
            "from_zone": a.from_zone, "to_zone": a.to_zone}


def _send_alert(conn: sqlite3.Connection, run_id: str, a: SendAlertArgs) -> dict[str, Any]:
    _require_zone(conn, a.zone)
    if has_link(a.message):
        raise ActionRejected("alert contains a link or email address")
    if looks_like_injection(a.message):
        raise ActionRejected("alert contains instruction-like text")
    conn.execute(
        "INSERT INTO alert_outbox (run_id, created_at, zone, language, channel, message, status) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_id, db.now_iso(), a.zone, a.language, a.channel, a.message, "SIMULATED_SENT"),
    )
    return {"zone": a.zone, "language": a.language}


TOOLS: dict[str, tuple[type[_Args], Callable[..., dict[str, Any]]]] = {
    "reserve_shelter": (ReserveShelterArgs, _reserve_shelter),
    "dispatch_resource": (DispatchArgs, _dispatch_resource),
    "send_alert": (SendAlertArgs, _send_alert),
}

# Least privilege: which caller may use which tool. Only the Executor can act.
PERMISSIONS: dict[str, frozenset[str]] = {"executor": frozenset(TOOLS)}


# ------------------------------------------------------------------ failure simulation (demos/tests)
_calls: dict[str, int] = {}


def reset_simulation() -> None:
    _calls.clear()


def _simulated_failure(name: str) -> bool:
    """SIMULATE_TOOL_FAILURE=send_alert -> always fail; send_alert:2 -> fail the first 2 calls."""
    for part in os.getenv("SIMULATE_TOOL_FAILURE", "").split(","):
        tool, _, count = part.strip().lower().partition(":")
        if tool != name:
            continue
        _calls[name] = _calls.get(name, 0) + 1
        return _calls[name] <= int(count) if count.isdigit() else True
    return False


# ------------------------------------------------------------------ dispatcher
def _idempotency_key(run_id: str, seq: int, name: str, args: BaseModel) -> str:
    raw = f"{run_id}|{seq}|{name}|{json.dumps(args.model_dump(), sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def call_tool(
    conn: sqlite3.Connection, run_id: str, seq: int, name: str, args: dict[str, Any],
    caller: str = "executor",
) -> dict[str, Any]:
    if name not in PERMISSIONS.get(caller, frozenset()):
        raise PermissionDenied(f"'{caller}' is not allowed to call '{name}'")
    schema, fn = TOOLS[name]
    try:
        parsed = schema.model_validate(args)
    except ValidationError as exc:
        raise ActionRejected(f"invalid arguments for {name}: {exc.error_count()} problem(s)") from exc
    if _simulated_failure(name):
        raise TransientToolError(f"simulated failure of '{name}'")

    cur = conn.execute(
        "INSERT OR IGNORE INTO action_ledger "
        "(idempotency_key, run_id, tool, params_json, status, created_at) VALUES (?,?,?,?,?,?)",
        (_idempotency_key(run_id, seq, name, parsed), run_id, name,
         json.dumps(parsed.model_dump()), "DONE", db.now_iso()),
    )
    if cur.rowcount == 0:
        return {"tool": name, "status": "already_done"}
    try:
        detail = fn(conn, run_id, parsed)
    except sqlite3.IntegrityError as exc:
        raise ActionRejected(f"integrity violation in {name}: {exc}") from exc
    return {"tool": name, "status": "done", **detail}


def run_has_actions(run_id: str) -> bool:
    """True if this run already executed something (so a re-run is a resume)."""
    with db.get_conn() as conn:
        return conn.execute(
            "SELECT 1 FROM action_ledger WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone() is not None