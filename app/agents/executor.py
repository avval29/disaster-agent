"""Executor agent (tool-using): carries out an approved plan through restricted tools.

Gates, in order:  approval on record -> plan hash matches the approved plan -> plan mode is
executable -> pre-flight re-validation against current data (unless resuming a run).
Stage 1 (shelter reservations + resource dispatches) runs in ONE transaction: all or nothing.
Stage 2 (alerts) runs only after stage 1 has committed. Transient errors are retried.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

from app import db
from app.agents.approval import hash_plan
from app.agents.common import audit, log_step
from app.agents.critic import check_plan
from app.agents.planner import ADVISORY, RESPONSE
from app.config import settings
from app.tools import actions

AGENT = "executor"
MAX_ATTEMPTS = 3


def _build_calls(plan: dict[str, Any]) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    stage1: list[tuple[str, dict]] = []
    for a in plan.get("shelter_assignments") or []:
        stage1.append(("reserve_shelter", {
            "shelter_id": a.get("shelter_id"), "zone": a.get("zone"), "people": a.get("people")}))
    for d in plan.get("dispatches") or []:
        stage1.append(("dispatch_resource", {
            "resource_type": d.get("resource_type"), "from_zone": d.get("from_zone"),
            "to_zone": d.get("to_zone"), "quantity": d.get("quantity")}))
    stage2: list[tuple[str, dict]] = []
    seen: set[tuple[Any, str]] = set()
    for al in plan.get("alerts") or []:
        for lang, key in (("en", "english"), ("hi", "hindi")):
            if (al.get("zone"), lang) in seen:          # one alert per zone and language
                continue
            seen.add((al.get("zone"), lang))
            stage2.append(("send_alert", {
                "zone": al.get("zone"), "language": lang, "channel": "sms",
                "message": al.get(key)}))
    return stage1, stage2


def _run_stage(run_id: str, calls: list[tuple[str, dict]], start_seq: int):
    """One atomic stage with retries for transient errors. Returns (results, attempts)."""
    if not calls:
        return [], 0
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with db.get_conn() as conn:
                results = [actions.call_tool(conn, run_id, start_seq + i, name, args)
                           for i, (name, args) in enumerate(calls)]
            return results, attempt
        except (actions.TransientToolError, sqlite3.OperationalError) as exc:
            last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(settings.http_backoff_s * attempt)
    raise actions.TransientToolError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")


def _summarize(*result_lists: list[dict[str, Any]]) -> dict[str, Any]:
    results = [r for lst in result_lists for r in lst]
    done = [r for r in results if r["status"] == "done"]
    return {
        "reservations": sum(1 for r in done if r["tool"] == "reserve_shelter"),
        "people_sheltered": sum(r.get("people", 0) for r in done if r["tool"] == "reserve_shelter"),
        "dispatches": sum(1 for r in done if r["tool"] == "dispatch_resource"),
        "alerts_sent": sum(1 for r in done if r["tool"] == "send_alert"),
        "already_done": sum(1 for r in results if r["status"] == "already_done"),
    }


def _finish(run_id, status: str, summary: dict[str, Any], errors: list[str] | None = None,
            warnings: list[str] | None = None) -> dict[str, Any]:
    execution = {"status": status, **summary}
    audit(run_id, AGENT, "executed", execution)
    return {
        "execution": execution,
        "warnings": warnings or [],
        "errors": errors or [],
        "execution_log": [log_step(AGENT, "execute", ok=status == "completed", status=status)],
    }


def _blocked(run_id, reason: str) -> dict[str, Any]:
    return _finish(run_id, "blocked", {"reason": reason}, errors=[f"executor: blocked - {reason}"])


def _stale(state: dict[str, Any], plan: dict[str, Any], issues: list[dict[str, str]]) -> dict[str, Any]:
    previous = int(state.get("replan_count", 0) or 0)
    codes = [i["code"] for i in issues if i["level"] == "error"]
    update = _finish(state.get("run_id"), "stale_plan", {"reason": ", ".join(codes)},
                     warnings=["Executor: the approved plan is no longer valid against current data; "
                               "replanning (a new approval will be required)"])
    update.update({
        "critique": {"passed": False, "attempt": previous + 1, "issues": issues,
                     "plan_source": plan.get("source")},
        "approval_status": "pending",
        "approved_plan_hash": "",
        "replan_count": previous + 1,
    })
    return update


def executor_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    plan = state.get("plan") or {}

    if state.get("approval_status") != "approved":
        return _blocked(run_id, "no human approval on record")
    if state.get("approved_plan_hash") != hash_plan(plan):
        return _blocked(run_id, "the plan changed after it was approved")
    if plan.get("mode") not in (RESPONSE, ADVISORY):
        return _blocked(run_id, f"plan mode '{plan.get('mode')}' is not executable")

    if not actions.run_has_actions(run_id):            # a resumed run must not be re-validated
        issues = check_plan(plan, state)
        if any(i["level"] == "error" for i in issues):
            return _stale(state, plan, issues)

    stage1, stage2 = _build_calls(plan)
    try:
        r1, attempts1 = _run_stage(run_id, stage1, 0)
    except Exception as exc:
        return _finish(run_id, "failed", {"error": str(exc)[:300]}, errors=[
            f"executor: shelter/dispatch stage failed and was rolled back ({str(exc)[:200]}); "
            "no alerts were sent"])
    try:
        r2, attempts2 = _run_stage(run_id, stage2, len(stage1))
    except Exception as exc:
        summary = {**_summarize(r1), "stage1_attempts": attempts1, "error": str(exc)[:300]}
        return _finish(run_id, "partial", summary, errors=[
            f"executor: alerts were NOT sent ({str(exc)[:200]}); shelters and resources are already "
            "committed - send the alerts manually"])

    summary = {**_summarize(r1, r2), "stage1_attempts": attempts1, "alert_attempts": attempts2,
               "error": None}
    return _finish(run_id, "completed", summary)


def route_after_executor(state: dict[str, Any]) -> str:
    status = (state.get("execution") or {}).get("status")
    if status == "stale_plan":
        return "planner" if int(state.get("replan_count", 0) or 0) <= settings.max_replans else "escalate"
    if status == "blocked":
        return "escalate"
    return "finalize"