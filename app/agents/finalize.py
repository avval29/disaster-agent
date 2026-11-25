"""Escalation and finalisation: final outcome, human-readable report, long-term memory update."""
from __future__ import annotations

import sqlite3
from typing import Any

from app.agents.common import audit, log_step
from app.agents.planner import HUMAN_REVIEW
from app.memory import long_term

AGENT = "finalizer"
_FROM_EXECUTION = {"completed": "executed", "partial": "partial", "failed": "execution_failed"}


def outcome(state: dict[str, Any]) -> str:
    if state.get("status"):
        return state["status"]
    exec_status = (state.get("execution") or {}).get("status")
    if exec_status in _FROM_EXECUTION:
        return _FROM_EXECUTION[exec_status]
    if state.get("approval_status") == "rejected":
        return "rejected"
    if (state.get("plan") or {}).get("mode") == "no_action":
        return "no_action"
    return "unknown"


def escalate_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or {}
    execution = state.get("execution") or {}
    if plan.get("mode") == HUMAN_REVIEW:
        reason = "Risk could not be assessed from the available data."
    elif execution.get("status") == "blocked":
        reason = f"Execution blocked: {execution.get('reason')}."
    else:
        reason = f"No valid plan after {state.get('replan_count', 0)} attempts."
    errors = state.get("errors") or []
    if errors:
        reason += f" Last error: {errors[-1][:200]}"
    audit(state.get("run_id"), AGENT, "escalated", {"reason": reason})
    return {
        "status": "escalated",
        "escalation_reason": reason,
        "warnings": [f"ESCALATED TO A HUMAN: {reason}"],
        "errors": [],
        "execution_log": [log_step(AGENT, "escalate", ok=False)],
    }


def build_report(state: dict[str, Any], status: str) -> str:
    plan = state.get("plan") or {}
    ex = state.get("execution") or {}
    lines = [
        f"Run {state.get('run_id')} | {state.get('region')} | {state.get('hazard_type')} / "
        f"{state.get('severity')} (score {float(state.get('risk_score') or 0):.0f}/100)",
        f"Outcome: {status}",
    ]
    if plan.get("headline"):
        lines.append(f"Plan: {plan['headline']} [{plan.get('source')}]")
    if plan.get("unmet_people"):
        lines.append(f"Unmet shelter need: {plan['unmet_people']} people (capacity shortfall)")
    if state.get("escalation_reason"):
        lines.append(f"Escalation: {state['escalation_reason']}")
    if ex.get("status") in ("completed", "partial", "failed"):
        lines.append(
            f"Execution ({ex['status']}): {ex.get('reservations', 0)} shelter reservations "
            f"({ex.get('people_sheltered', 0)} people), {ex.get('dispatches', 0)} dispatches, "
            f"{ex.get('alerts_sent', 0)} alerts written to the simulated outbox")
    if state.get("human_rejections"):
        lines.append(f"Human rejections: {state['human_rejections']}")
    sources = state.get("data_sources") or {}
    if sources:
        lines.append("Data sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()))
    lines.append(f"Warnings: {len(state.get('warnings') or [])} | errors: {len(state.get('errors') or [])}")
    lines.append("Alert delivery is simulated; shelter and resource data are synthetic.")
    return "\n".join(lines)


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    status = outcome(state)
    report = build_report(state, status)
    warnings: list[str] = []
    try:
        long_term.save_run({**state, "status": status}, status=status)
    except sqlite3.Error as exc:
        warnings.append(f"could not save the run to long-term memory: {exc}")
    audit(state.get("run_id"), AGENT, "finalized", {"status": status})
    return {
        "status": status,
        "final_report": report,
        "warnings": warnings,
        "errors": [],
        "execution_log": [log_step(AGENT, "finalize", status=status)],
    }