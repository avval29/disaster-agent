"""Plan -> critique -> replan loop. In Phase 6 LangGraph's conditional edges replace this."""
from __future__ import annotations

from typing import Any

from app.agents.critic import critic_node, route_after_critic
from app.agents.planner import planner_node
from app.config import settings
from app.state import apply_update


def plan_until_valid(state: dict[str, Any]) -> str:
    """Mutates state; returns the final route ('approval', 'finish' or 'escalate')."""
    for _ in range(settings.max_replans + 2):      # hard guard against an infinite loop
        apply_update(state, planner_node(state))
        apply_update(state, critic_node(state))
        route = route_after_critic(state)
        if route != "planner":
            return route
    return "escalate"