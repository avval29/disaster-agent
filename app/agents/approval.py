"""Human approval gate (human-in-the-loop) using LangGraph interrupt().

The node pauses the graph and waits for a decision from a person (UI or CLI). The decision
is bound to a hash of the exact plan that was shown. Only a literal approved=True counts.
Nothing with side effects may run before interrupt(): the node re-executes on resume.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from langgraph.types import interrupt

from app.agents.common import audit, log_step
from app.security.injection_guard import sanitize_text

AGENT = "human_approval"
MAX_HUMAN_ROUNDS = 3


def hash_plan(plan: dict[str, Any]) -> str:
    canonical = json.dumps(plan, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def approval_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or {}
    plan_hash = hash_plan(plan)
    issues = (state.get("critique") or {}).get("issues", [])

    decision = interrupt({
        "kind": "plan_approval",
        "run_id": state.get("run_id"),
        "plan_hash": plan_hash,
        "hazard": state.get("hazard_type"),
        "severity": state.get("severity"),
        "headline": plan.get("headline"),
        "summary": plan.get("summary"),
        "warnings": [i["message"] for i in issues if i.get("level") == "warning"],
        "irreversible": [
            "Reserves shelter places",
            "Deducts resource stock",
            "Issues public alerts (simulated in this project)",
        ],
    })

    valid = isinstance(decision, dict)
    approved = valid and decision.get("approved") is True
    feedback = sanitize_text(str(decision.get("feedback", "")), 500) if valid else ""
    reviewer = sanitize_text(str(decision.get("reviewer", "human")), 60) if valid else "unknown"
    audit(state.get("run_id"), AGENT, "decision",
          {"approved": approved, "reviewer": reviewer, "plan_hash": plan_hash})

    if approved:
        return {
            "approval_status": "approved",
            "approved_plan_hash": plan_hash,
            "human_feedback": feedback,
            "warnings": [], "errors": [],
            "execution_log": [log_step(AGENT, "approved", reviewer=reviewer, plan_hash=plan_hash)],
        }
    rejections = int(state.get("human_rejections", 0) or 0) + 1
    return {
        "approval_status": "rejected",
        "approved_plan_hash": "",
        "human_feedback": feedback,
        "human_rejections": rejections,
        "replan_count": 0,                      # a human rejection gives the planner a fresh budget
        "warnings": [f"Plan rejected by human (round {rejections}); replanning"],
        "errors": [],
        "execution_log": [log_step(AGENT, "rejected", reviewer=reviewer, round=rejections)],
    }


def route_after_approval(state: dict[str, Any]) -> str:
    if state.get("approval_status") == "approved":
        return "executor"
    if int(state.get("human_rejections", 0) or 0) >= MAX_HUMAN_ROUNDS:
        return "finalize"
    return "planner"