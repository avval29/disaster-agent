import json

from app import db
from app.memory import long_term, short_term
from app.state import apply_update, new_state


def make_run(hazard="flood", severity="High", **extra):
    state = new_state("request mentioning secret@example.com", 28.6, 77.2, scenario="flood_heavy_rain")
    apply_update(state, {
        "hazard_type": hazard, "severity": severity, "risk_score": 65.5,
        "risk_rationale": "Heavy rain forecast.", "key_concerns": ["wet ground"],
        "approval_status": "approved", **extra,
    })
    return state


def test_save_and_recall_run():
    run = make_run()
    long_term.save_run(run)
    found = long_term.recall_similar_runs("flood")
    assert len(found) == 1
    assert found[0]["run_id"] == run["run_id"]
    assert found[0]["summary"]["risk_score"] == 65.5
    assert "secret@example.com" not in json.dumps(found[0])   # raw request is never stored
    assert long_term.recall_similar_runs("heatwave") == []


def test_recall_prefers_same_severity():
    critical, high = make_run(severity="Critical"), make_run(severity="High")
    long_term.save_run(critical)
    long_term.save_run(high)                                   # saved later, so more recent
    found = long_term.recall_similar_runs("flood", severity="Critical")
    assert found[0]["run_id"] == critical["run_id"]


def test_recall_excludes_current_run():
    run = make_run()
    long_term.save_run(run)
    assert long_term.recall_similar_runs("flood", exclude_run_id=run["run_id"]) == []


def test_save_note_redacts_pii():
    long_term.save_note("human_feedback", "flood", "Call Rahul on 9876543210 or rahul@example.com about boats")
    notes = long_term.recall_notes("human_feedback", key="flood")
    assert "9876543210" not in notes[0]["content"]
    assert "rahul@example.com" not in notes[0]["content"]
    assert "[redacted" in notes[0]["content"]


def test_purge_old():
    long_term.save_note("human_feedback", "flood", "old note")
    long_term.save_run(make_run())
    with db.get_conn() as conn:
        conn.execute("UPDATE memory SET created_at = '2020-01-01T00:00:00+00:00'")
        conn.execute("UPDATE runs SET created_at = '2020-01-01T00:00:00+00:00'")
    long_term.save_note("human_feedback", "flood", "fresh note")
    assert long_term.purge_older_than(30) == {"memory": 1, "runs": 1}
    assert len(long_term.recall_notes("human_feedback", limit=10)) == 1


def test_short_term_context_bounded():
    state = make_run(risk_rationale="Very long rationale. " * 200)
    ctx = short_term.build_context(state, max_chars=300)
    assert len(ctx) <= 300
    assert "flood" in ctx.lower()