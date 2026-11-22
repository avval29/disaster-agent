"""Shared state passed between agents in the LangGraph graph.

Fields annotated with operator.add are *appended to* by each node instead of
overwritten, so every agent can add warnings, errors and log entries safely.
"""
import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

Severity = Literal["Low", "Moderate", "High", "Critical"]
SEVERITY_ORDER = ["Low", "Moderate", "High", "Critical"]
HazardType = Literal["flood", "heatwave", "earthquake"]
ApprovalStatus = Literal["pending", "approved", "rejected", "not_required"]


class DisasterState(TypedDict, total=False):
    # Identity and input
    run_id: str
    request: str
    region: str
    lat: float
    lon: float

    # Perception (Monitor)
    observations: dict[str, Any]          # {"weather": {...}, "quake": {...}}
    data_sources: dict[str, str]          # {"weather": "live" | "cache" | "none"}

    # Risk (Risk Analyst)
    hazard_type: str
    severity: str
    risk_score: float
    risk_rationale: str

    # Knowledge (Protocol Retriever)
    protocols: list[dict[str, Any]]

    # Planning loop (Planner / Critic)
    plan: dict[str, Any]
    critique: dict[str, Any]
    replan_count: int

    # Human-in-the-loop
    approval_status: str
    human_feedback: str

    # Outcome (Executor)
    final_report: str

    # Append-only trails
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    execution_log: Annotated[list[dict[str, Any]], operator.add]


def new_state(request: str, lat: float, lon: float, region: str = "Delhi") -> DisasterState:
    """Create a fresh state for one run."""
    return DisasterState(
        run_id=uuid.uuid4().hex[:12],
        request=request,
        region=region,
        lat=lat,
        lon=lon,
        observations={},
        data_sources={},
        protocols=[],
        replan_count=0,
        approval_status="pending",
        warnings=[],
        errors=[],
        execution_log=[],
    )