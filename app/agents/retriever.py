"""Protocol Retriever agent (deliberative): RAG over guideline documents plus
recall of similar past runs from long-term memory.

Retrieved text is untrusted: suspicious chunks are dropped before any other agent
sees them. Contents pages and spaced-out headings are dropped as low value. If nothing
is indexed, the agent says so and does not invent protocol content.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.agents.common import audit, log_step
from app.memory import long_term
from app.rag import search as rag_search

AGENT = "protocol_retriever"
MAX_PROTOCOLS = 6
PER_QUERY_K = 3
MAX_CHARS = 1200
MAX_DISTANCE = 1.0   # smaller = closer match; typical good hits are 0.5 to 0.9
MIN_KEEP = 3         # always keep at least this many of the best hits

QUERIES = {
    "flood": [
        "evacuation plan procedure and responsibilities of district administration",
        "flood shelter and relief camp arrangements and basic services",
        "flood warning dissemination to the public and communities at risk",
        "deployment of rescue boats and medical teams during floods",
    ],
    "heatwave": [
        "cooling centres operation location and services for vulnerable groups",
        "heat stroke prevention medical preparedness and hospital readiness",
        "drinking water distribution and public advisories during heat wave",
    ],
    "earthquake": [
        "search and rescue operations after earthquake in collapsed buildings",
        "emergency shelter relief camps and medical response after earthquake",
        "public information and damage assessment after earthquake",
    ],
}

_NUMERIC = re.compile(r"^\d[\d.]*$")


def is_low_value(text: str) -> bool:
    """Contents pages (many numbers) and spaced-out headings (many one-letter tokens)."""
    tokens = text.split()
    if len(tokens) < 15:
        return True
    numeric = sum(1 for t in tokens if _NUMERIC.match(t.strip(",;:()")))
    singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return numeric / len(tokens) > 0.10 or singles / len(tokens) > 0.08


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
    low_value = 0
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
                    if is_low_value(hit["text"]):
                        low_value += 1
                        continue
                    best = merged.get(hit["id"])
                    if best is None or hit["distance"] < best["distance"]:
                        merged[hit["id"]] = hit
            ranked = sorted(merged.values(), key=lambda h: h["distance"])
            close = [h for h in ranked if h["distance"] <= MAX_DISTANCE]
            if len(close) < MIN_KEEP:
                close = ranked[:MIN_KEEP]
            protocols = [
                {"id": h["id"], "text": h["text"][:MAX_CHARS], "source": h["source"],
                 "page": h["page"], "hazard": h["hazard"], "distance": h["distance"]}
                for h in close[:MAX_PROTOCOLS]
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
        "suspicious_dropped": len(suspicious_ids), "low_value_dropped": low_value,
    })
    return {
        "protocols": protocols,
        "past_cases": past,
        "warnings": warnings,
        "errors": [],
        "execution_log": [log_step(AGENT, "retrieve", protocols=len(protocols), past_cases=len(past))],
    }