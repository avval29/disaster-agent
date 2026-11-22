"""Monitor agent (reactive): perceives the environment via tools or a scenario."""
from __future__ import annotations

from typing import Any

from app.agents.common import audit, log_step
from app.scenarios import get_scenario
from app.tools.base import ToolResult
from app.tools.quake import get_recent_earthquakes
from app.tools.weather import get_weather_forecast

AGENT = "monitor"


def _absorb(res: ToolResult, observations, sources, warnings, errors, log) -> None:
    log.append(log_step(AGENT, f"tool:{res.tool}", ok=res.ok, source=res.source,
                        latency_ms=res.latency_ms, error=res.error))
    if res.ok:
        observations[res.tool] = res.data
        sources[res.tool] = res.source
        if res.source == "cache":
            minutes = (res.cache_age_s or 0) // 60
            warnings.append(f"{res.tool}: live data unavailable, using cached data ({minutes} min old)")
    else:
        sources[res.tool] = "none"
        errors.append(f"{res.tool}: {res.error}")


def monitor_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    scenario = state.get("scenario")
    observations: dict[str, Any] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []
    log: list[dict[str, Any]] = []

    if scenario:
        try:
            sc = get_scenario(scenario)
        except KeyError:
            errors.append(f"monitor: unknown scenario '{scenario}'")
            log.append(log_step(AGENT, "load_scenario", ok=False, scenario=scenario))
        else:
            observations = {"weather": sc["weather"], "quake": sc["quake"]}
            sources = {"weather": "scenario", "quake": "scenario"}
            warnings.append(f"SCENARIO MODE: '{scenario}' - observations are simulated, not live")
            log.append(log_step(AGENT, "load_scenario", scenario=scenario))
    else:
        lat, lon = state["lat"], state["lon"]
        for res in (get_weather_forecast(lat, lon), get_recent_earthquakes(lat, lon)):
            _absorb(res, observations, sources, warnings, errors, log)

    audit(run_id, AGENT, "observed", {"sources": sources, "warnings": warnings, "errors": errors})
    return {
        "observations": observations,
        "data_sources": sources,
        "warnings": warnings,
        "errors": errors,
        "execution_log": log,
    }