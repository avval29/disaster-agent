import pytest

from app import db
from app.agents import monitor as monitor_mod
from app.graph import build_graph, decide, pending_approval, run_state, start_run
from app.memory import long_term
from app.tools.base import ToolResult

FLOOD = dict(request="demo", lat=28.6139, lon=77.2090, scenario="flood_heavy_rain")


@pytest.fixture
def graph():
    return build_graph()


def total_occupancy():
    return sum(s["occupancy"] for s in db.get_shelters())


def outbox_count():
    with db.get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM alert_outbox").fetchone()[0]


def test_pauses_for_approval_before_any_action(graph):
    run_id = start_run(graph, **FLOOD)
    pending = pending_approval(graph, run_id)
    assert pending and pending["kind"] == "plan_approval" and pending["plan_hash"]
    assert pending["hazard"] == "flood"
    assert "execution" not in run_state(graph, run_id)
    assert outbox_count() == 0 and total_occupancy() == 70       # nothing has happened yet


def test_approve_executes_and_remembers(graph):
    run_id = start_run(graph, **FLOOD)
    decide(graph, run_id, True, reviewer="tester")
    assert pending_approval(graph, run_id) is None
    state = run_state(graph, run_id)
    assert state["status"] == "executed"
    assert state["execution"]["people_sheltered"] == 1440
    assert outbox_count() == 6 and total_occupancy() == 70 + 1440
    assert "Outcome: executed" in state["final_report"]
    assert long_term.recall_similar_runs("flood")[0]["status"] == "executed"


def test_reject_replans_then_approve(graph):
    run_id = start_run(graph, **FLOOD)
    decide(graph, run_id, False, feedback="Use fewer shelters")
    assert pending_approval(graph, run_id) is not None           # a new plan awaits approval
    state = run_state(graph, run_id)
    assert state["human_rejections"] == 1 and "fewer shelters" in state["human_feedback"]
    assert outbox_count() == 0 and total_occupancy() == 70       # rejection executed nothing
    decide(graph, run_id, True)
    assert run_state(graph, run_id)["status"] == "executed"


def test_three_rejections_end_the_run(graph):
    run_id = start_run(graph, **FLOOD)
    for _ in range(3):
        decide(graph, run_id, False)
    assert pending_approval(graph, run_id) is None
    assert run_state(graph, run_id)["status"] == "rejected"
    assert outbox_count() == 0 and total_occupancy() == 70


def test_calm_needs_no_approval(graph):
    run_id = start_run(graph, request="demo", lat=28.6139, lon=77.2090, scenario="calm")
    assert pending_approval(graph, run_id) is None
    state = run_state(graph, run_id)
    assert state["status"] == "no_action" and state["approval_status"] == "not_required"


def test_no_data_escalates(graph, monkeypatch):
    down = lambda *a, **k: ToolResult(False, "x", None, "none", "down", 0)  # noqa: E731
    monkeypatch.setattr(monitor_mod, "get_weather_forecast", down)
    monkeypatch.setattr(monitor_mod, "get_recent_earthquakes", down)
    run_id = start_run(graph, request="live", lat=28.6139, lon=77.2090)
    assert pending_approval(graph, run_id) is None
    state = run_state(graph, run_id)
    assert state["status"] == "escalated" and state["escalation_reason"]
    assert outbox_count() == 0


def test_decide_without_pending_raises(graph):
    run_id = start_run(graph, request="demo", lat=28.6139, lon=77.2090, scenario="calm")
    with pytest.raises(ValueError):
        decide(graph, run_id, True)


def test_input_validation(graph):
    with pytest.raises(ValueError):
        start_run(graph, "x", 28.6, 77.2, scenario="not_a_scenario")
    with pytest.raises(ValueError):
        start_run(graph, "x", 95.0, 77.2)


def test_alert_failure_reported_as_partial(graph, monkeypatch):
    monkeypatch.setenv("SIMULATE_TOOL_FAILURE", "send_alert")
    run_id = start_run(graph, **FLOOD)
    decide(graph, run_id, True)
    state = run_state(graph, run_id)
    assert state["status"] == "partial"
    assert "Outcome: partial" in state["final_report"]
    assert outbox_count() == 0 and total_occupancy() == 70 + 1440


def test_stale_plan_after_approval_needs_new_approval(graph):
    run_id = start_run(graph, **FLOOD)
    first = pending_approval(graph, run_id)
    with db.get_conn() as conn:
        conn.execute("UPDATE shelters SET occupancy = capacity")  # world changes while the human thinks
    decide(graph, run_id, True)
    second = pending_approval(graph, run_id)
    assert second is not None and second["plan_hash"] != first["plan_hash"]
    assert run_state(graph, run_id)["execution"]["status"] == "stale_plan"
    assert outbox_count() == 0                                   # the old approval executed nothing
    decide(graph, run_id, True)
    state = run_state(graph, run_id)
    assert state["status"] == "executed" and state["execution"]["people_sheltered"] == 0