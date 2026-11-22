"""Try retrieval and memory.

  python scripts/try_rag.py --query "flood evacuation" --hazard flood
  python scripts/try_rag.py --scenario flood_heavy_rain [--llm] [--save]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.agents.monitor import monitor_node  # noqa: E402
from app.agents.retriever import retriever_node  # noqa: E402
from app.agents.risk_analyst import risk_analyst_node  # noqa: E402
from app.config import settings  # noqa: E402
from app.memory import long_term  # noqa: E402
from app.rag import search as rag_search  # noqa: E402
from app.state import apply_update, new_state  # noqa: E402


def show_hits(hits) -> None:
    for h in hits:
        flag = "  [SUSPICIOUS]" if h.get("suspicious") else ""
        print(f"- {h['source']} p.{h['page']} [{h['hazard']}] distance={h['distance']}{flag}")
        print("   ", h["text"][:220].replace("\n", " "), "...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--hazard")
    parser.add_argument("--scenario")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--save", action="store_true", help="save this run to long-term memory")
    args = parser.parse_args()

    if not args.llm:
        settings.llm_provider = "none"
    db.init_db()
    db.seed_db()
    print(f"Indexed chunks: {rag_search.indexed_count()}")

    if args.query:
        show_hits(rag_search.search(args.query, hazard=args.hazard, k=5))
        return
    if not args.scenario:
        parser.error("give --query or --scenario")

    state = new_state("demo run", settings.demo_lat, settings.demo_lon, scenario=args.scenario)
    apply_update(state, monitor_node(state))
    apply_update(state, risk_analyst_node(state))
    apply_update(state, retriever_node(state))
    print(f"\nHazard={state['hazard_type']} severity={state['severity']} score={state['risk_score']}")
    print(f"\nProtocols retrieved: {len(state['protocols'])}")
    show_hits(state["protocols"])
    print(f"\nSimilar past runs: {len(state['past_cases'])}")
    for p in state["past_cases"]:
        print(f"- {p['created_at']} {p['hazard_type']}/{p['severity']} status={p['status']}")
    print("\nWarnings:", state["warnings"])
    if args.save:
        long_term.save_run(state, status="assessed")
        print("Saved this run to long-term memory.")


if __name__ == "__main__":
    main()