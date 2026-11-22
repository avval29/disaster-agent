"""Run Monitor + Risk Analyst on scenarios or live data.

  python scripts/try_agents.py                          all scenarios, rules only
  python scripts/try_agents.py --scenario flood_heavy_rain --llm
  python scripts/try_agents.py --live
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.agents.monitor import monitor_node  # noqa: E402
from app.agents.risk_analyst import risk_analyst_node  # noqa: E402
from app.config import settings  # noqa: E402
from app.scenarios import SCENARIOS  # noqa: E402
from app.state import apply_update, new_state  # noqa: E402


def run(scenario: str | None) -> dict:
    state = new_state("demo run", settings.demo_lat, settings.demo_lon, scenario=scenario)
    apply_update(state, monitor_node(state))
    apply_update(state, risk_analyst_node(state))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--live", action="store_true", help="use live APIs instead of a scenario")
    parser.add_argument("--llm", action="store_true", help="enable LLM review (uses API quota)")
    args = parser.parse_args()

    if not args.llm:
        settings.llm_provider = "none"
    db.init_db()
    db.seed_db()

    names = [None] if args.live else ([args.scenario] if args.scenario else SCENARIOS)
    print(f"{'scenario':<24}{'hazard':<12}{'severity':<10}{'score':>6}  sources")
    for name in names:
        s = run(name)
        print(f"{(name or 'LIVE'):<24}{s.get('hazard_type', '-'):<12}{s.get('severity', '-'):<10}"
              f"{s.get('risk_score', 0):>6.1f}  {s.get('data_sources')}")
        if len(names) == 1:
            print("\nRationale:", s.get("risk_rationale"))
            print("Concerns :", s.get("key_concerns"))
            print("Scores   :", s.get("hazard_scores"))
            print("Warnings :", s.get("warnings"))
            print("Errors   :", s.get("errors"))


if __name__ == "__main__":
    main()