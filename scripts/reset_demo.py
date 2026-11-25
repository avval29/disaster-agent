"""Restore demo data. Approved runs consume shelter places and stock; this resets them.

  python scripts/reset_demo.py                 reset shelters/resources, clear alerts and action ledger
  python scripts/reset_demo.py --wipe-memory   also clear runs, notes, audit log and API cache
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402


def main() -> None:
    db.init_db()
    tables = ["action_ledger", "alert_outbox"]
    if "--wipe-memory" in sys.argv:
        tables += ["runs", "memory", "audit_log", "api_cache"]
    with db.get_conn() as conn:
        for t in tables:
            conn.execute(f"DELETE FROM {t}")
    print("Cleared:", ", ".join(tables))
    print("Re-seeded:", db.seed_db(force=True))


if __name__ == "__main__":
    main()