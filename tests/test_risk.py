import pytest

from app import llm
from app.agents.monitor import monitor_node
from app.agents.risk_analyst import RiskReview, risk_analyst_node
from app.config import settings
from app.scenarios import SCENARIOS, get_scenario
from app.state import apply_update, new_state

EXPECTED = {
    "calm": ("none", "Low"),
    "moderate_rain": ("flood", "Moderate"),
    "flood_heavy_rain": ("flood", "High"),
    "flood_extreme": ("flood", "Critical"),
    "heatwave_high": ("heatwave", "High"),
    "heatwave_severe": ("heatwave", "Critical"),
    "earthquake_m6": ("earthquake", "Critical"),
    "earthquake_distant_m4": ("earthquake", "Low"),
}


def run(scenario):
    state = new_state("test", 28.6139, 77.2090, scenario=scenario)
    apply_update(state, monitor_node(state))
    apply_update(state, risk_analyst_node(state))
    return state


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", "test-key")


def fake_review(severity):
    def _fake(system, user, schema):
        return RiskReview(recommended_severity=severity, rationale="test rationale",
                          key_concerns=["a", "b"])
    return _fake


@pytest.mark.parametrize("scenario,expected", list(EXPECTED.items()))
def test_rule_based_severity(scenario, expected):
    s = run(scenario)
    assert (s["hazard_type"], s["severity"]) == expected


def test_expectations_cover_all_scenarios():
    assert set(EXPECTED) == set(SCENARIOS)


def test_no_data_gives_unknown():
    state = new_state("test", 28.6, 77.2)
    apply_update(state, risk_analyst_node(state))
    assert state["severity"] == "Unknown"
    assert state["errors"]


def test_partial_data_still_assesses():
    state = new_state("test", 28.6, 77.2)
    state["observations"] = {"quake": get_scenario("earthquake_m6")["quake"]}
    apply_update(state, risk_analyst_node(state))
    assert state["hazard_type"] == "earthquake"
    assert state["severity"] == "Critical"
    assert any("flood" in w for w in state["warnings"])


def test_llm_escalation_is_capped_at_one_level(llm_on, monkeypatch):
    monkeypatch.setattr(llm, "ask_json", fake_review("Critical"))
    s = run("moderate_rain")          # rules say Moderate
    assert s["severity"] == "High"    # Moderate -> High, not Critical
    assert s["risk_score"] >= 50
    assert "capped" in s["risk_rationale"]


def test_llm_can_escalate_one_level(llm_on, monkeypatch):
    monkeypatch.setattr(llm, "ask_json", fake_review("Critical"))
    s = run("flood_heavy_rain")       # rules say High
    assert s["severity"] == "Critical"
    assert s["risk_score"] >= 75
    assert s["key_concerns"] == ["a", "b"]


def test_llm_cannot_lower_severity(llm_on, monkeypatch):
    monkeypatch.setattr(llm, "ask_json", fake_review("Low"))
    s = run("flood_heavy_rain")
    assert s["severity"] == "High"
    assert "conservative" in s["risk_rationale"]


def test_llm_failure_falls_back_to_rules(llm_on, monkeypatch):
    def down(system, user, schema):
        raise llm.LLMUnavailable("simulated outage")

    monkeypatch.setattr(llm, "ask_json", down)
    s = run("flood_heavy_rain")
    assert s["severity"] == "High"
    assert any("unavailable" in w for w in s["warnings"])
    assert "Rule-based" in s["risk_rationale"]


def test_prompt_contains_no_free_text(llm_on, monkeypatch):
    seen = {}

    def spy(system, user, schema):
        seen["user"] = user
        return RiskReview(recommended_severity="High", rationale="r", key_concerns=[])

    monkeypatch.setattr(llm, "ask_json", spy)
    run("earthquake_m6")
    assert "Sonipat" not in seen["user"]       # place names are excluded
    assert "<data>" in seen["user"]