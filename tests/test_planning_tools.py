from app import db
from app.tools import resources, shelters


def test_scope_zones():
    assert {z["name"] for z in shelters.scope_zones("flood")} == {"North Delhi", "East Delhi", "Shahdara"}
    assert len(shelters.scope_zones("earthquake")) == 8
    assert shelters.scope_zones("none") == []


def test_needs_high_flood():
    needs = shelters.compute_needs("flood", "High")
    assert sum(n["need"] for n in needs) == 1440
    assert {n["zone"] for n in needs} == {"North Delhi", "East Delhi", "Shahdara"}


def test_needs_moderate_is_zero():
    assert all(n["need"] == 0 for n in shelters.compute_needs("flood", "Moderate"))


def test_allocation_conserves_people_and_respects_capacity():
    needs = shelters.compute_needs("flood", "Critical")          # 4800 people
    eligible = shelters.eligible_shelters("flood")                # 4180 places
    assignments, unmet = shelters.allocate(needs, eligible)
    assert sum(a["people"] for a in assignments) + sum(unmet.values()) == 4800
    assert sum(unmet.values()) == 620
    available = {s["id"]: s["available"] for s in eligible}
    used: dict[int, int] = {}
    for a in assignments:
        used[a["shelter_id"]] = used.get(a["shelter_id"], 0) + a["people"]
    assert all(used[i] <= available[i] for i in used)
    safe_ids = {s["id"] for s in db.get_shelters() if s["flood_safe"]}
    assert {a["shelter_id"] for a in assignments} <= safe_ids


def test_allocation_prefers_same_zone():
    needs = shelters.compute_needs("flood", "High")
    assignments, _ = shelters.allocate(needs, shelters.eligible_shelters("flood"))
    shahdara = [a for a in assignments if a["zone"] == "Shahdara"]
    assert any(a["shelter_name"] == "Relief Camp Shahdara" and a["people"] == 500 for a in shahdara)


def test_fallback_dispatch_within_stock_and_relevant():
    needs = shelters.compute_needs("flood", "High")
    out = resources.fallback_dispatch("flood", "High", needs)
    assert out
    stock = resources.stock_map()
    sent: dict[tuple[str, str], int] = {}
    for d in out:
        key = (d["resource_type"], d["from_zone"])
        sent[key] = sent.get(key, 0) + d["quantity"]
    assert all(q <= stock[k] for k, q in sent.items())
    types = {d["resource_type"] for d in out}
    assert types <= set(resources.RELEVANT["flood"])
    assert "water_tanker" not in types


def test_no_dispatch_for_moderate():
    needs = shelters.compute_needs("flood", "Moderate")
    assert resources.fallback_dispatch("flood", "Moderate", needs) == []