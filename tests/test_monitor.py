import app.agents.monitor as monitor_mod
from app.agents.monitor import monitor_node
from app.state import new_state
from app.tools.base import ToolResult


def make_state(scenario=None):
    return new_state("test", 28.6139, 77.2090, scenario=scenario)


def ok(tool, source="live", error=None, age=None):
    data = {"summary": {}, "daily": []} if tool == "weather" else {"count": 0}
    return ToolResult(True, tool, data, source, error, 5, age)


def fail(tool):
    return ToolResult(False, tool, None, "none", "boom", 0)


def patch_tools(monkeypatch, weather, quake):
    monkeypatch.setattr(monitor_mod, "get_weather_forecast", lambda lat, lon: weather)
    monkeypatch.setattr(monitor_mod, "get_recent_earthquakes", lambda lat, lon: quake)


def test_scenario_mode():
    upd = monitor_node(make_state("flood_heavy_rain"))
    assert upd["data_sources"] == {"weather": "scenario", "quake": "scenario"}
    assert set(upd["observations"]) == {"weather", "quake"}
    assert any("SCENARIO MODE" in w for w in upd["warnings"])
    assert upd["errors"] == []


def test_unknown_scenario():
    upd = monitor_node(make_state("does_not_exist"))
    assert upd["observations"] == {}
    assert any("unknown scenario" in e for e in upd["errors"])


def test_live_success(monkeypatch):
    patch_tools(monkeypatch, ok("weather"), ok("quake"))
    upd = monitor_node(make_state())
    assert upd["data_sources"] == {"weather": "live", "quake": "live"}
    assert upd["warnings"] == [] and upd["errors"] == []


def test_cache_fallback_creates_warning(monkeypatch):
    patch_tools(monkeypatch, ok("weather", "cache", "live fetch failed", 1800), ok("quake"))
    upd = monitor_node(make_state())
    assert upd["data_sources"]["weather"] == "cache"
    assert any("cached" in w and "30 min" in w for w in upd["warnings"])


def test_all_tools_fail(monkeypatch):
    patch_tools(monkeypatch, fail("weather"), fail("quake"))
    upd = monitor_node(make_state())
    assert upd["observations"] == {}
    assert len(upd["errors"]) == 2
    assert upd["data_sources"] == {"weather": "none", "quake": "none"}