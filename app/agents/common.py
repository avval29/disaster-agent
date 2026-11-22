"""Helpers shared by all agents."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

from app import db

logger = logging.getLogger(__name__)


def log_step(agent: str, action: str, ok: bool = True, **detail: Any) -> dict[str, Any]:
    """One entry for state['execution_log']."""
    return {"ts": db.now_iso(), "agent": agent, "action": action, "ok": ok, **detail}


def audit(run_id: str | None, actor: str, event: str, detail: Any = None) -> None:
    """Write to the audit table. A logging failure must never crash a response run."""
    try:
        db.log_event(run_id, actor, event, detail)
    except sqlite3.Error:
        logger.exception("audit log write failed")