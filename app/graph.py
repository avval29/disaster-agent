"""LangGraph orchestrator: wires the agents into one graph with a human approval gate.

  monitor -> risk -> retriever -> planner <-> critic -> approval -> executor -> finalize
                                    ^            |  |        |          |
                                    |            |  |        +--reject--+--> planner
                                    +------------+  +--> escalate / finish
Routing is deterministic Python, not an LLM. The checkpointer lets the run pause at the
approval gate and resume when a human decides. (MemorySaver is in-memory: a pending approval
is lost if the process restarts.)
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.approval import approval_node, route_after_approval
from app.agents.critic import critic_node, route_after_critic
from app.agents.executor import executor_node, route_after_executor
from app.agents.finalize import escalate_node, finalize_node
from app.agents.monitor import monitor_node
from app.agents.planner import planner_node
from app.agents.retriever import retriever_node
from app.agents.risk_analyst import risk_analyst_node
from app.config import settings
from app.scenarios import SCENARIOS
from app.security.injection_guard import sanitize_text
from app.state import DisasterState, new_state
from app.tools.base import validate_coords


def _wrap(fn):
    """Unannotated wrapper so LangGraph does not try to infer a schema from type hints."""
    return lambda state: fn(state)


def build_graph(checkpointer=None):
    g = StateGraph(DisasterState)
    nodes = {
        "monitor": monitor_node, "risk": risk_analyst_node, "retriever": retriever_node,
        "planner": planner_node, "critic": critic_node, "approval": approval_node,
        "executor": executor_node, "escalate": escalate_node, "finalize": finalize_node,
    }
    for name, fn in nodes.items():
        g.add_node(name, _wrap(fn))

    g.add_edge(START, "monitor")
    g.add_edge("monitor", "risk")
    g.add_edge("risk", "retriever")
    g.add_edge("retriever", "planner")
    g.add_edge("planner", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "planner": "planner", "approval": "approval", "finish": "finalize", "escalate": "escalate"})
    g.add_conditional_edges("approval", route_after_approval, {
        "executor": "executor", "planner": "planner", "finalize": "finalize"})
    g.add_conditional_edges("executor", route_after_executor, {
        "planner": "planner", "escalate": "escalate", "finalize": "finalize"})
    g.add_edge("escalate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ------------------------------------------------------------------ small API for CLI/UI
def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 60}


def start_run(graph, request: str, lat: float | None = None, lon: float | None = None,
              scenario: str | None = None, region: str | None = None) -> str:
    """Validate input, start a run, and return its run_id. Stops at the approval gate if needed."""
    lat = settings.demo_lat if lat is None else lat
    lon = settings.demo_lon if lon is None else lon
    validate_coords(lat, lon)                                    # raises ValueError
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'")
    state = new_state(sanitize_text(request, 500), lat, lon,
                      region=region or settings.demo_city, scenario=scenario)
    graph.invoke(state, _config(state["run_id"]))
    return state["run_id"]


def pending_approval(graph, run_id: str) -> dict[str, Any] | None:
    """The approval request the run is waiting on, or None."""
    for task in graph.get_state(_config(run_id)).tasks:
        for item in getattr(task, "interrupts", ()):
            return item.value
    return None


def decide(graph, run_id: str, approved: bool, feedback: str = "", reviewer: str = "human") -> None:
    if pending_approval(graph, run_id) is None:
        raise ValueError("no approval is pending for this run")
    graph.invoke(Command(resume={"approved": approved, "feedback": feedback, "reviewer": reviewer}),
                 _config(run_id))


def run_state(graph, run_id: str) -> dict[str, Any]:
    return dict(graph.get_state(_config(run_id)).values)