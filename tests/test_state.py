from app.state import apply_update, new_state


def test_apply_update_appends_and_overwrites():
    s = new_state("x", 1.0, 2.0)
    apply_update(s, {"warnings": ["a"], "severity": "High"})
    apply_update(s, {"warnings": ["b"], "severity": "Low"})
    assert s["warnings"] == ["a", "b"]
    assert s["severity"] == "Low"


def test_new_state_defaults():
    s = new_state("x", 1.0, 2.0, scenario="calm")
    assert len(s["run_id"]) == 12
    assert s["approval_status"] == "pending"
    assert s["scenario"] == "calm"
    assert s["errors"] == [] and s["execution_log"] == []