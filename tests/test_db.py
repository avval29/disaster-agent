from app import db


def test_seed_counts():
    assert len(db.get_zones()) == 8
    assert len(db.get_shelters()) == 11
    assert len(db.get_resources()) == 13


def test_shelters_by_zone():
    shelters = db.get_shelters(zone="East Delhi")
    assert len(shelters) == 2
    assert all(s["zone"] == "East Delhi" for s in shelters)


def test_shelter_availability_calculated():
    stadium = next(s for s in db.get_shelters() if s["name"] == "Indoor Stadium Central")
    assert stadium["available"] == 750  # capacity 800 - occupancy 50


def test_seed_is_idempotent():
    db.seed_db()
    db.seed_db()
    assert len(db.get_shelters()) == 11


def test_cache_expiry(monkeypatch):
    db.cache_set("k", {"a": 1})
    assert db.cache_get("k", max_age_s=3600)[0] == {"a": 1}
    real_time = db.time.time()
    monkeypatch.setattr(db.time, "time", lambda: real_time + 10_000)
    assert db.cache_get("k", max_age_s=3600) is None