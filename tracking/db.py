"""
tracking/db.py

All SQLite plumbing lives here. Everything else in the app talks to
functions in this file, never to sqlite3 directly — so if you later
outgrow SQLite and move to Postgres/MySQL/Supabase, this is the only
file that needs to change.
"""

import os
import sqlite3
from contextlib import contextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "metrics.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    topic               TEXT,
    attempts_used       INTEGER,
    approved            INTEGER,
    one_shot_approval   INTEGER,
    refusal_flagged     INTEGER,
    total_latency_s     REAL,
    total_tokens        INTEGER,
    est_total_cost_usd  REAL,
    num_llm_calls       INTEGER,
    created_at          TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    node            TEXT,
    attempt         INTEGER,
    model           TEXT,
    latency_s       REAL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    total_tokens    INTEGER,
    est_cost_usd    REAL,
    verdict         TEXT,
    error           TEXT,
    timestamp       TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def db_cursor():
    """Usage: with db_cursor() as cur: cur.execute(...)"""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read helpers — for dashboards, notebooks, or a resume screenshot of a
# query result showing approval rate / avg latency across runs.
# ---------------------------------------------------------------------------


def fetch_recent_runs(limit: int = 20) -> list:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]


def fetch_calls_for_run(run_id: str) -> list:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,)
        )
        return [dict(row) for row in cur.fetchall()]


def overall_stats() -> dict:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                   AS total_runs,
                AVG(approved) * 100.0                      AS approval_rate_pct,
                AVG(one_shot_approval) * 100.0             AS one_shot_rate_pct,
                AVG(refusal_flagged) * 100.0               AS refusal_rate_pct,
                AVG(total_latency_s)                       AS avg_latency_s,
                AVG(total_tokens)                          AS avg_tokens,
                AVG(est_total_cost_usd)                    AS avg_cost_usd
            FROM runs
            """
        )
        row = cur.fetchone()
        stats = dict(row) if row else {}

        cur.execute(
            """
            SELECT
                COUNT(*)                                          AS total_calls,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_calls
            FROM llm_calls
            """
        )
        call_row = cur.fetchone()
        if call_row and call_row["total_calls"]:
            stats["error_rate_pct"] = round(
                100.0 * call_row["error_calls"] / call_row["total_calls"], 2
            )
        else:
            stats["error_rate_pct"] = 0.0
        return stats



# ---------------------------------------------------------------------------
# Schema migrations — safely adds any columns present in SCHEMA but missing
# from the live DB. Runs on every startup; it's a no-op when the schema is
# already up-to-date, so it adds zero overhead in the normal case.
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "llm_calls": [
        ("verdict", "TEXT"),
        ("error",   "TEXT"),
    ],
    "runs": [],
}


def _migrate_db() -> None:
    with get_connection() as conn:
        for table, columns in _EXPECTED_COLUMNS.items():
            existing = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                    )
                    print(f"[db] migrated: added column '{col_name}' to '{table}'")


init_db()
_migrate_db()
