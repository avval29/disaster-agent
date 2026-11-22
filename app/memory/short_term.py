"""Short-term (working) memory: a bounded situation brief built from the run state.

In LangGraph the state itself is the working memory. This turns the relevant
parts into compact text for the Planner and Critic prompts, sanitised because
some of it is LLM-generated.
"""
from __future__ import annotations

from typing import Any

from app.security.injection_guard import sanitize_text


def build_context(state: dict[str, Any], max_chars: int = 3000) -> str:
    score = float(state.get("risk_score") or 0)
    lines = [
        f"Region: {state.get('region', '?')} | Hazard: {state.get('hazard_type', '?')} | "
        f"Severity: {state.get('severity', '?')} (risk score {score:.0f}/100)"
    ]
    sources = state.get("data_sources") or {}
    if sources:
        lines.append("Data sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()))
    if state.get("risk_rationale"):
        lines.append("Assessment: " + sanitize_text(state["risk_rationale"], 600))
    for concern in (state.get("key_concerns") or [])[:3]:
        lines.append("Concern: " + sanitize_text(concern, 200))
    for warning in (state.get("warnings") or [])[:5]:
        lines.append("Warning: " + sanitize_text(warning, 200))
    if state.get("human_feedback"):
        lines.append("Human feedback on the previous plan: " + sanitize_text(state["human_feedback"], 400))
    return "\n".join(lines)[:max_chars]