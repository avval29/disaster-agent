"""Run the full chain: Monitor -> Risk -> Retriever -> Planner <-> Critic.

  python scripts/try_plan.py --scenario flood_heavy_rain
  python scripts/try_plan.py --scenario flood_extreme --llm
  python scripts/try_plan.py --live
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.agents.monitor import monitor_node  # noqa: E402
from app.agents.planning_loop import plan_until_valid  # noqa: E402
from app.agents.retriever import retriever_node  # noqa: E402
from app.agents.risk_analyst import risk_analyst_node  # noqa: E402
from app.config import settings  # noqa: E402
from app.state import apply_update, new_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="flood_heavy_rain")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--llm", action="store_true", help="use the LLM (slower, uses quota)")
    args = parser.parse_args()
    if not args.llm:
        settings.llm_provider = "none"
    db.init_db()
    db.seed_db()

    start = time.perf_counter()
    state = new_state("demo run", settings.demo_lat, settings.demo_lon,
                      scenario=None if args.live else args.scenario)
    for node in (monitor_node, risk_analyst_node, retriever_node):
        apply_update(state, node(state))
    route = plan_until_valid(state)
    elapsed = time.perf_counter() - start

    plan = state["plan"]
    print(f"Hazard={state['hazard_type']} severity={state['severity']} score={state['risk_score']}")
    print(f"Protocols retrieved: {len(state['protocols'])}")
    print(f"\nROUTE: {route} | mode={plan['mode']} | source={plan['source']} | "
          f"replans={state.get('replan_count', 0)} | {elapsed:.1f}s")
    print(f"HEADLINE: {plan['headline']}")
    print(f"SUMMARY : {plan['summary']}")

    by_shelter: dict[str, int] = defaultdict(int)
    for a in plan["shelter_assignments"]:
        by_shelter[a["shelter_name"]] += a["people"]
    if by_shelter:
        print(f"\nShelters (unmet people: {plan['unmet_people']}):")
        for name, n in by_shelter.items():
            print(f"  {name:<34}{n:>6}")
    if plan["dispatches"]:
        print("\nDispatches:")
        for d in plan["dispatches"]:
            print(f"  {d['quantity']:>5} x {d['resource_type']:<13} {d['from_zone']} -> {d['to_zone']}")
    if plan["alerts"]:
        print("\nAlert drafts (first zone):")
        print("  EN:", plan["alerts"][0]["english"])
        print("  HI:", plan["alerts"][0]["hindi"])
    if plan["citations"]:
        print("\nCitations:", [(c["label"], c["source"], c["page"]) for c in plan["citations"]])

    print("\nCritic attempts:")
    for step in state["execution_log"]:
        if step["agent"] == "critic":
            print(f"  attempt {step['attempt']}: ok={step['ok']} source={step['plan_source']} "
                  f"errors={step['error_codes']} warnings={step['warning_codes']}")
    print("\nWarnings:", *state["warnings"], sep="\n  - ")
    print("Errors  :", state["errors"])


if __name__ == "__main__":
    main()