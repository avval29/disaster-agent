"""Protocol Retriever agent (deliberative): RAG over guideline documents plus
recall of similar past runs from long-term memory.

Retrieved text is untrusted: suspicious chunks are dropped before any other
agent sees them. If nothing is indexed, the agent says so and does not invent
protocol content.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from app.agents.common import audit, log_step
from app.memory import long_term
from app.rag import search as rag_search

AGENT = "protocol_retriever"
MAX_PROTOCOLS = 6
PER_QUERY_K = 3
MAX_CHARS = 1200

QUERIES = {
    "flood": [
        "flood evacuation and shelter management guidelines",
        "flood early warning and public communication",
        "flood rescue relief and medical response operations",
    ],
    "heatwave": [
        "heat wave action plan cooling centres and public advisories",
        "heat stroke prevention medical preparedness hospitals",
        "drinking water supply and vulnerable groups during heat wave",
    ],
    "earthquake": [
        "earthquake search and rescue operations and response",
        "earthquake emergency shelter relief and medical response",
        "earthquake public communication and damage assessment",
    ],
}


def retriever_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    hazard = state.get("hazard_type")
    severity = state.get("severity")

    if hazard not in QUERIES:
        audit(run_id, AGENT, "skipped", {"hazard": hazard})
        return {
            "protocols": [], "past_cases": [], "warnings": [], "errors": [],
            "execution_log": [log_step(AGENT, "skip", reason="no actionable hazard")],
        }

    warnings: list[str] = []

    past: list[dict[str, Any]] = []
    try:
        past = long_term.recall_similar_runs(hazard, severity, limit=3, exclude_run_id=run_id)
    except sqlite3.Error as exc:
        warnings.append(f"memory recall failed: {exc}")

    protocols: list[dict[str, Any]] = []
    suspicious_ids: set[str] = set()
    try:
        if not rag_search.is_indexed():
            warnings.append(
                "No guideline documents are indexed (run: python -m app.rag.ingest); "
                "the plan will not be grounded in official protocols"
            )
        else:
            merged: dict[str, dict[str, Any]] = {}
            for query in QUERIES[hazard]:
                for hit in rag_search.search(query, hazard=hazard, k=PER_QUERY_K):
                    if hit["suspicious"]:
                        suspicious_ids.add(hit["id"])
                        continue
                    best = merged.get(hit["id"])
                    if best is None or hit["distance"] < best["distance"]:
                        merged[hit["id"]] = hit
            ranked = sorted(merged.values(), key=lambda h: h["distance"])[:MAX_PROTOCOLS]
            protocols = [
                {"id": h["id"], "text": h["text"][:MAX_CHARS], "source": h["source"],
                 "page": h["page"], "hazard": h["hazard"], "distance": h["distance"]}
                for h in ranked
            ]
            if suspicious_ids:
                warnings.append(
                    f"Dropped {len(suspicious_ids)} suspicious guideline chunk(s) that looked like "
                    "prompt injection"
                )
    except Exception as exc:  # embedding model download, Chroma errors, etc.
        protocols = []
        warnings.append(
            f"protocol retrieval failed ({type(exc).__name__}: {str(exc)[:120]}); "
            "continuing without protocols"
        )

    audit(run_id, AGENT, "retrieved", {
        "hazard": hazard, "protocols": len(protocols), "past_cases": len(past),
        "suspicious_dropped": len(suspicious_ids),
    })
    return {
        "protocols": protocols,
        "past_cases": past,
        "warnings": warnings,
        "errors": [],
        "execution_log": [log_step(AGENT, "retrieve", protocols=len(protocols), past_cases=len(past))],
    }