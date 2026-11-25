"""Critic agent (evaluator): deterministic validation of the plan.

Every check recomputes from the database and the state; nothing in the plan is
trusted. Issues have level 'error' (blocks the plan) or 'warning' (shown to the human).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app import db
from app.agents.common import audit, log_step
from app.agents.planner import ADVISORY, HUMAN_REVIEW, NO_ACTION, RESPONSE, plan_mode
from app.config import settings
from app.security.injection_guard import looks_like_injection
from app.tools.resources import RELEVANT, stock_map
from app.tools.shelters import compute_needs

AGENT = "critic"
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LINK = re.compile(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+", re.IGNORECASE)
MAX_ALERT_CHARS = 600


def check_plan(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    def err(code: str, msg: str) -> None:
        issues.append({"level": "error", "code": code, "message": msg})

    def warn(code: str, msg: str) -> None:
        issues.append({"level": "warning", "code": code, "message": msg})

    hazard, severity = state.get("hazard_type", "none"), state.get("severity", "Unknown")
    expected = plan_mode(hazard, severity)
    if plan.get("mode") != expected:
        err("mode", f"plan mode '{plan.get('mode')}' does not match expected '{expected}'")
    if not str(plan.get("headline", "")).strip():
        err("headline", "plan has no headline")

    zones = {z["name"]: z for z in db.get_zones()}
    shelters = {s["id"]: s for s in db.get_shelters()}
    needs = ({n["zone"]: n["need"] for n in compute_needs(hazard, severity)}
             if expected in (RESPONSE, ADVISORY) else {})
    scope = set(needs)

    # ---- shelter assignments
    assignments = plan.get("shelter_assignments") or []
    if expected != RESPONSE and assignments:
        err("unexpected_assignments", f"a '{expected}' plan must not assign shelters")
    used: dict[int, int] = defaultdict(int)
    per_zone: dict[str, int] = defaultdict(int)
    for a in assignments:
        sid, people, zone = a.get("shelter_id"), a.get("people"), a.get("zone")
        shelter = shelters.get(sid)
        if shelter is None:
            err("unknown_shelter", f"shelter id {sid} does not exist")
            continue
        if not isinstance(people, int) or isinstance(people, bool) or people <= 0:
            err("bad_people", f"{shelter['name']}: people must be a positive whole number")
            continue
        if zone not in zones:
            err("unknown_zone", f"unknown zone '{zone}' in a shelter assignment")
            continue
        if zone not in scope:
            err("zone_out_of_scope", f"{zone} is not an affected zone for {hazard}")
            continue
        if hazard == "flood" and not shelter["flood_safe"]:
            err("unsafe_shelter", f"{shelter['name']} is not flood-safe")
        used[sid] += people
        per_zone[zone] += people
    for sid, n in used.items():
        if n > shelters[sid]["available"]:
            err("capacity", f"{shelters[sid]['name']}: {n} people assigned but only "
                            f"{shelters[sid]['available']} places available")

    if expected == RESPONSE:
        for zone, need in needs.items():
            if per_zone.get(zone, 0) > need * 1.1 + 5:
                err("over_allocated", f"{zone}: {per_zone[zone]} people assigned for a need of {need}")
        eligible = [s for s in shelters.values()
                    if s["available"] > 0 and (hazard != "flood" or s["flood_safe"])]
        spare = sum(max(0, s["available"] - used.get(s["id"], 0)) for s in eligible)
        short_total = sum(max(0, need - per_zone.get(z, 0)) for z, need in needs.items())
        if short_total > 0:
            gap = min(spare, short_total)
            tolerance = max(10, int(0.02 * sum(needs.values())))
            if gap > tolerance:
                err("under_allocated", f"{short_total} people have no shelter although {spare} "
                                       f"places are still free")
            else:
                warn("capacity_shortfall", f"{short_total} people cannot be sheltered with current "
                                           f"capacity; request mutual aid or open more shelters")

    # ---- resource dispatch
    dispatches = plan.get("dispatches") or []
    if expected != RESPONSE and dispatches:
        err("unexpected_dispatch", f"a '{expected}' plan must not dispatch resources")
    stock = stock_map()
    known_types = {t for t, _ in stock}
    sent: dict[tuple[str, str], int] = defaultdict(int)
    for d in dispatches:
        rtype, fz, tz, qty = d.get("resource_type"), d.get("from_zone"), d.get("to_zone"), d.get("quantity")
        if rtype not in known_types:
            err("unknown_resource", f"unknown resource type '{rtype}'")
            continue
        if fz not in zones or tz not in zones:
            err("unknown_zone", f"unknown zone in dispatch of {rtype}")
            continue
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            err("bad_quantity", f"{rtype}: quantity must be a positive whole number")
            continue
        if rtype not in RELEVANT.get(hazard, []):
            warn("irrelevant_resource", f"{rtype} is not a typical response resource for {hazard}")
        sent[(rtype, fz)] += qty
    for (rtype, fz), qty in sent.items():
        if qty > stock.get((rtype, fz), 0):
            err("insufficient_stock", f"{rtype} from {fz}: {qty} requested but only "
                                      f"{stock.get((rtype, fz), 0)} in stock")

    # ---- alerts
    alerts = plan.get("alerts") or []
    if expected in (RESPONSE, ADVISORY):
        missing = sorted(scope - {a.get("zone") for a in alerts})
        if missing:
            err("missing_alerts", f"no alert drafted for: {', '.join(missing)}")
    for a in alerts:
        zone = a.get("zone")
        en, hi = str(a.get("english", "")).strip(), str(a.get("hindi", "")).strip()
        if zone not in zones:
            err("unknown_zone", f"alert addressed to unknown zone '{zone}'")
        if not en or not hi:
            err("empty_alert", f"alert for {zone} must have English and Hindi text")
            continue
        if len(en) > MAX_ALERT_CHARS or len(hi) > MAX_ALERT_CHARS:
            err("alert_too_long", f"alert for {zone} exceeds {MAX_ALERT_CHARS} characters")
        if not DEVANAGARI.search(hi):
            err("hindi_script", f"Hindi alert for {zone} is not in Devanagari script")
        for text in (en, hi):
            if looks_like_injection(text):
                err("alert_injection", f"alert for {zone} contains instruction-like text")
            if LINK.search(text):
                err("alert_link", f"alert for {zone} contains a link or email address")

    # ---- grounding
    protocols = state.get("protocols") or []
    valid = {f"P{i + 1}" for i in range(len(protocols))}
    cited = plan.get("cited_protocols") or []
    bad = [c for c in cited if c not in valid]
    if bad:
        err("bad_citation", f"cites guideline passages that do not exist: {', '.join(bad)}")
    if expected in (RESPONSE, ADVISORY):
        if not protocols:
            warn("ungrounded", "plan is not grounded in official protocol text "
                               "(no guideline passages were retrieved)")
        elif not cited:
            warn("no_citations", "plan cites no guideline passages")
    return issues


def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or {}
    issues = check_plan(plan, state)
    errors = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]
    passed = not errors
    previous = int(state.get("replan_count", 0) or 0)

    update: dict[str, Any] = {
        "critique": {"passed": passed, "attempt": previous + 1, "issues": issues,
                     "plan_source": plan.get("source")},
        "warnings": [],
        "errors": [],
        "execution_log": [log_step(
            AGENT, "check", ok=passed, attempt=previous + 1, plan_source=plan.get("source"),
            error_codes=[i["code"] for i in errors], warning_codes=[i["code"] for i in warnings])],
    }
    if passed:
        update["warnings"] = [f"Critic: {w['message']}" for w in warnings]
        if plan.get("mode") == NO_ACTION:
            update["approval_status"] = "not_required"
    else:
        update["replan_count"] = previous + 1
        if previous + 1 > settings.max_replans:
            update["errors"] = [
                f"critic: no valid plan after {previous + 1} attempts; human review required"
            ]
    audit(state.get("run_id"), AGENT, "checked", {
        "passed": passed, "attempt": previous + 1,
        "error_codes": [i["code"] for i in errors]})
    return update


def route_after_critic(state: dict[str, Any]) -> str:
    """'planner' (replan) | 'approval' (ask the human) | 'finish' (nothing to do) | 'escalate'."""
    critique = state.get("critique") or {}
    mode = (state.get("plan") or {}).get("mode")
    if critique.get("passed"):
        if mode == NO_ACTION:
            return "finish"
        if mode == HUMAN_REVIEW:
            return "escalate"
        return "approval"
    if int(state.get("replan_count", 0) or 0) <= settings.max_replans:
        return "planner"
    return "escalate"