"""Long-term memory in SQLite: past runs and notes.

Privacy: the raw request text is never stored, free text is redacted for emails
and phone numbers, and purge_older_than() enforces a retention period.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app import db
from app.security.injection_guard import sanitize_text

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")


def redact_pii(text: str) -> str:
    return _PHONE.sub("[redacted-phone]", _EMAIL.sub("[redacted-email]", text))


def _clean(text: str, limit: int) -> str:
    return redact_pii(sanitize_text(text, limit))


# ------------------------------------------------------------------ runs
def save_run(state: dict[str, Any], status: str | None = None) -> None:
    plan = state.get("plan") or {}
    summary = {
        "scenario": state.get("scenario"),
        "risk_score": state.get("risk_score"),
        "rationale": _clean(state.get("risk_rationale") or "", 500),
        "key_concerns": [_clean(c, 200) for c in (state.get("key_concerns") or [])[:3]],
        "plan_headline": _clean(str(plan.get("headline", "")), 200) if plan else None,
        "replan_count": state.get("replan_count", 0),
        "human_feedback": _clean(state.get("human_feedback") or "", 300),
        "data_sources": state.get("data_sources") or {},
    }
    with db.get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, created_at, region, hazard_type, severity, status, summary_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (state["run_id"], db.now_iso(), state.get("region"), state.get("hazard_type"),
             state.get("severity"), status or state.get("approval_status") or "assessed",
             json.dumps(summary, default=str)),
        )


def recall_similar_runs(
    hazard: str, severity: str | None = None, limit: int = 3, exclude_run_id: str | None = None
) -> list[dict[str, Any]]:
    """Same-hazard runs, same severity first, then most recent."""
    sql = ("SELECT run_id, created_at, region, hazard_type, severity, status, summary_json "
           "FROM runs WHERE hazard_type = ?")
    params: list[Any] = [hazard]
    if exclude_run_id:
        sql += " AND run_id != ?"
        params.append(exclude_run_id)
    sql += " ORDER BY (severity = ?) DESC, created_at DESC, rowid DESC LIMIT ?"
    params += [severity or "", limit]
    with db.get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        out.append(item)
    return out


# ------------------------------------------------------------------ notes
def save_note(kind: str, key: str | None, content: str) -> None:
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO memory (created_at, kind, key, content) VALUES (?,?,?,?)",
            (db.now_iso(), kind, key, _clean(content, 500)),
        )


def recall_notes(kind: str, key: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    sql = "SELECT id, created_at, kind, key, content FROM memory WHERE kind = ?"
    params: list[Any] = [kind]
    if key is not None:
        sql += " AND key = ?"
        params.append(key)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with db.get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


# ------------------------------------------------------------------ retention
def purge_older_than(days: int) -> dict[str, int]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with db.get_conn() as conn:
        mem = conn.execute("DELETE FROM memory WHERE created_at < ?", (cutoff,)).rowcount
        runs = conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,)).rowcount
    return {"memory": mem, "runs": runs}