"""Shared state passed between agents in the LangGraph graph.

Fields annotated with operator.add are *appended to* by each node instead of
overwritten, so every agent can add warnings, errors and log entries safely.
"""
import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

Severity = Literal["Low", "Moderate", "High", "Critical", "Unknown"]
SEVERITY_ORDER = ["Low", "Moderate", "High", "Critical"]
HazardType = Literal["flood", "heatwave", "earthquake", "none"]
ApprovalStatus = Literal["pending", "approved", "rejected", "not_required"]

APPEND_KEYS = ("warnings", "errors", "execution_log")


class DisasterState(TypedDict, total=False):
    # Identity and input
    run_id: str
    request: str
    region: str
    lat: float
    lon: float
    scenario: str | None                  # None = live data; otherwise a scenario name

    # Perception (Monitor)
    observations: dict[str, Any]          # {"weather": {...}, "quake": {...}}
    data_sources: dict[str, str]          # {"weather": "live" | "cache" | "scenario" | "none"}

    # Risk (Risk Analyst)
    hazard_type: str
    severity: str
    risk_score: float
    hazard_scores: dict[str, float | None]
    risk_rationale: str
    key_concerns: list[str]

    # Knowledge (Protocol Retriever)
    protocols: list[dict[str, Any]]
    past_cases: list[dict[str, Any]]      # similar past runs recalled from long-term memory

    # Planning loop (Planner / Critic)
    plan: dict[str, Any]
    critique: dict[str, Any]
    replan_count: int

    # Human-in-the-loop
    approval_status: str
    human_feedback: str
    human_rejections: int
    approved_plan_hash: str               # hash of the exact plan the human approved

    # Outcome (Executor)
    final_report: str
    execution: dict[str, Any]             # Executor result summary
    status: str                           # final outcome
    escalation_reason: str

    # Append-only trails
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    execution_log: Annotated[list[dict[str, Any]], operator.add]


def new_state(
    request: str, lat: float, lon: float, region: str = "Delhi", scenario: str | None = None
) -> DisasterState:
    """Create a fresh state for one run."""
    return DisasterState(
        run_id=uuid.uuid4().hex[:12],
        request=request,
        region=region,
        lat=lat,
        lon=lon,
        scenario=scenario,
        observations={},
        data_sources={},
        protocols=[],
        past_cases=[],
        replan_count=0,
        human_rejections=0,
        approved_plan_hash="",
        approval_status="pending",
        warnings=[],
        errors=[],
        execution_log=[],
    )


def apply_update(state: DisasterState, update: dict[str, Any]) -> DisasterState:
    """Merge a node's partial update into the state, the way LangGraph will.

    Append-only keys are extended; everything else is overwritten. Used in
    tests, scripts and the UI when calling nodes directly.
    """
    for key, value in update.items():
        if key in APPEND_KEYS:
            state[key] = list(state.get(key, [])) + list(value)
        else:
            state[key] = value
    return state