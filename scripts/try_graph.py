"""End-to-end run through the LangGraph graph with a real human approval step.

  python scripts/try_graph.py --scenario flood_heavy_rain                    (asks you)
  python scripts/try_graph.py --scenario flood_heavy_rain --decision approve
  python scripts/try_graph.py --scenario flood_heavy_rain --decision approve --sim-tool-failure send_alert:2
  python scripts/try_graph.py --scenario flood_heavy_rain --decision reject
  python scripts/try_graph.py --scenario flood_extreme --llm --decision approve
Approved runs change the database; run scripts/reset_demo.py to restore the seed data.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph import build_graph, decide, pending_approval, run_state, start_run  # noqa: E402


def show_plan(pending: dict, state: dict) -> None:
    plan = state["plan"]
    shelters = {a["shelter_id"] for a in plan["shelter_assignments"]}
    people = sum(a["people"] for a in plan["shelter_assignments"])
    print(f"\n=== APPROVAL REQUIRED: {pending['hazard']} / {pending['severity']} ===")
    print(f"Plan source : {plan['source']} (attempt {plan['attempt']}) | plan hash {pending['plan_hash']}")
    print(f"Headline    : {pending['headline']}")
    print(f"Summary     : {pending['summary']}")
    print(f"Shelters    : {people} people in {len(shelters)} shelters (unmet {plan['unmet_people']})")
    print(f"Dispatches  : {len(plan['dispatches'])} | Alerts: {len(plan['alerts'])} zones x 2 languages")
    for w in pending["warnings"]:
        print("  !", w)
    print("If approved, this will:", *pending["irreversible"], sep="\n  - ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="flood_heavy_rain")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--decision", choices=["approve", "reject"])
    parser.add_argument("--sim-tool-failure", help="e.g. send_alert:2 (fail twice) or send_alert (always)")
    parser.add_argument("--sim-api-failure", help="e.g. weather (live mode only)")
    args = parser.parse_args()

    if not args.llm:
        settings.llm_provider = "none"
    if args.sim_tool_failure:
        os.environ["SIMULATE_TOOL_FAILURE"] = args.sim_tool_failure
    if args.sim_api_failure:
        os.environ["SIMULATE_API_FAILURE"] = args.sim_api_failure
    db.init_db()
    db.seed_db()

    start = time.perf_counter()
    graph = build_graph()
    run_id = start_run(graph, "demo run", scenario=None if args.live else args.scenario)

    for _ in range(6):
        pending = pending_approval(graph, run_id)
        if pending is None:
            break
        show_plan(pending, run_state(graph, run_id))
        if args.decision == "approve":
            approved, feedback = True, ""
        elif args.decision == "reject":
            approved, feedback = False, "rejected by the --decision flag"
        else:
            ans = input("\nApprove? [y = approve, n = reject, or type feedback to reject]: ").strip()
            approved = ans.lower() in ("y", "yes")
            feedback = "" if ans.lower() in ("y", "yes", "n", "no") else ans
        print(f"-> decision: {'APPROVED' if approved else 'REJECTED'}")
        decide(graph, run_id, approved, feedback, reviewer="cli")

    state = run_state(graph, run_id)
    print(f"\n{'=' * 60}\n{state['final_report']}\n{'=' * 60}")
    if state.get("execution"):
        print("Execution:", json.dumps(state["execution"]))
    with db.get_conn() as conn:
        outbox = conn.execute("SELECT COUNT(*) FROM alert_outbox WHERE run_id = ?", (run_id,)).fetchone()[0]
        ledger = conn.execute("SELECT COUNT(*) FROM action_ledger WHERE run_id = ?", (run_id,)).fetchone()[0]
        occupancy = conn.execute("SELECT SUM(occupancy) FROM shelters").fetchone()[0]
    print(f"DB effects: {outbox} outbox rows, {ledger} ledger entries, total shelter occupancy {occupancy}")
    print("\nAgent trace:", " -> ".join(s["agent"] for s in state["execution_log"]))
    print("Warnings:", *state["warnings"], sep="\n  - ")
    print("Errors  :", state["errors"], f"| {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()