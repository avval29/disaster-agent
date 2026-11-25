from app import db
from app.agents.approval import hash_plan
from app.agents.critic import critic_node
from app.agents.executor import executor_node, route_after_executor
from app.agents.planner import planner_node
from app.state import apply_update, new_state


def approve(s):
    apply_update(s, {"approval_status": "approved", "approved_plan_hash": hash_plan(s["plan"])})
    return s


def approved_state(severity="High", hazard="flood"):
    s = new_state("t", 28.6139, 77.2090)
    apply_update(s, {"hazard_type": hazard, "severity": severity, "risk_score": 65.0})
    apply_update(s, planner_node(s))
    apply_update(s, critic_node(s))
    return approve(s)


def total_occupancy():
    return sum(s["occupancy"] for s in db.get_shelters())


def outbox_count():
    with db.get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM alert_outbox").fetchone()[0]


def total_stock():
    return sum(r["quantity"] for r in db.get_resources())


def test_executes_approved_plan():
    s = approved_state()
    before = total_occupancy()                                   # seed data: 70 already occupied
    upd = executor_node(s)
    ex = upd["execution"]
    assert ex["status"] == "completed"
    assert ex["reservations"] == len(s["plan"]["shelter_assignments"])
    assert ex["people_sheltered"] == 1440
    assert ex["alerts_sent"] == 6                                # 3 zones x 2 languages
    assert total_occupancy() == before + 1440
    assert outbox_count() == 6
    with db.get_conn() as conn:
        statuses = {r[0] for r in conn.execute("SELECT status FROM alert_outbox")}
    assert statuses == {"SIMULATED_SENT"}


def test_blocked_without_approval():
    s = approved_state()
    apply_update(s, {"approval_status": "pending"})
    before = total_occupancy()
    apply_update(s, executor_node(s))
    assert s["execution"]["status"] == "blocked"
    assert total_occupancy() == before and outbox_count() == 0
    assert route_after_executor(s) == "escalate"


def test_blocked_when_plan_tampered_after_approval():
    s = approved_state()
    s["plan"]["headline"] = "changed after approval"
    before = total_occupancy()
    apply_update(s, executor_node(s))
    assert s["execution"]["status"] == "blocked"
    assert "changed" in s["execution"]["reason"]
    assert total_occupancy() == before and outbox_count() == 0


def test_blocked_for_non_executable_mode():
    s = approved_state(severity="Low", hazard="none")            # a no_action plan
    apply_update(s, executor_node(s))
    assert s["execution"]["status"] == "blocked"


def test_stale_plan_triggers_replan():
    s = approved_state()
    with db.get_conn() as conn:
        conn.execute("UPDATE shelters SET occupancy = capacity")  # shelters fill up while the human thinks
    apply_update(s, executor_node(s))
    assert s["execution"]["status"] == "stale_plan"
    assert s["critique"]["passed"] is False
    assert s["replan_count"] == 1
    assert s["approval_status"] == "pending" and s["approved_plan_hash"] == ""
    assert outbox_count() == 0
    assert route_after_executor(s) == "planner"


def test_stale_plan_escalates_when_budget_spent():
    s = approved_state()
    with db.get_conn() as conn:
        conn.execute("UPDATE shelters SET occupancy = capacity")
    apply_update(s, {"replan_count": 3})
    apply_update(s, executor_node(s))
    assert s["replan_count"] == 4
    assert route_after_executor(s) == "escalate"


def test_transient_failures_are_retried(monkeypatch):
    monkeypatch.setenv("SIMULATE_TOOL_FAILURE", "send_alert:2")
    s = approved_state()
    ex = executor_node(s)["execution"]
    assert ex["status"] == "completed"
    assert ex["alert_attempts"] == 3
    assert outbox_count() == 6


def test_permanent_alert_failure_gives_partial(monkeypatch):
    monkeypatch.setenv("SIMULATE_TOOL_FAILURE", "send_alert")
    s = approved_state()
    before = total_occupancy()
    upd = executor_node(s)
    assert upd["execution"]["status"] == "partial"
    assert total_occupancy() == before + 1440                    # shelters stay committed
    assert outbox_count() == 0
    assert any("NOT sent" in e for e in upd["errors"])


def test_dispatch_failure_rolls_back_everything(monkeypatch):
    monkeypatch.setenv("SIMULATE_TOOL_FAILURE", "dispatch_resource")
    s = approved_state()
    occ, stk = total_occupancy(), total_stock()
    upd = executor_node(s)
    assert upd["execution"]["status"] == "failed"
    assert total_occupancy() == occ and total_stock() == stk     # atomic stage: nothing persisted
    assert outbox_count() == 0                                   # alerts were never sent


def test_rerun_is_idempotent():
    s = approved_state()
    before = total_occupancy()
    apply_update(s, executor_node(s))
    second = executor_node(s)["execution"]
    assert second["status"] == "completed"
    assert second["already_done"] > 0 and second["reservations"] == 0
    assert total_occupancy() == before + 1440
    assert outbox_count() == 6


def test_advisory_plan_sends_alerts_only():
    s = approved_state(severity="Moderate")
    before, stock_before = total_occupancy(), total_stock()
    ex = executor_node(s)["execution"]
    assert ex["status"] == "completed" and ex["reservations"] == 0 and ex["dispatches"] == 0
    assert total_occupancy() == before and total_stock() == stock_before
    assert outbox_count() == 6


def test_duplicate_alerts_are_deduplicated():
    s = approved_state()
    s["plan"]["alerts"].append(dict(s["plan"]["alerts"][0]))
    approve(s)
    assert executor_node(s)["execution"]["status"] == "completed"
    assert outbox_count() == 6