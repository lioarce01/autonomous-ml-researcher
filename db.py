"""
db.py — SQLite wrapper for experiment tracking.
All functions use stdlib sqlite3 only.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "experiments.db")

DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    val_loss    REAL,
    notes       TEXT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    kept        INTEGER NOT NULL DEFAULT 0,
    git_hash    TEXT
);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(DDL)
    conn.commit()
    return conn


def log(name: str, val_loss: float, notes: str = "") -> dict:
    """Insert experiment. Sets kept=1 if this is a new best val_loss."""
    conn = _connect()
    current_best = conn.execute(
        "SELECT MIN(val_loss) as best FROM experiments WHERE kept = 1"
    ).fetchone()["best"]

    kept = 1 if (current_best is None or val_loss < current_best) else 0

    cursor = conn.execute(
        "INSERT INTO experiments (name, val_loss, notes, kept) VALUES (?, ?, ?, ?)",
        (name, val_loss, notes, kept),
    )
    conn.commit()
    row_id = cursor.lastrowid
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (row_id,)).fetchone()
    result = dict(row)
    conn.close()
    return result


def update_git_hash(row_id: int, git_hash: str):
    conn = _connect()
    conn.execute("UPDATE experiments SET git_hash = ? WHERE id = ?", (git_hash, row_id))
    conn.commit()
    conn.close()


def get_top(n: int = 5) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY val_loss ASC LIMIT ?", (n,)
    ).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_recent_failures(n: int = 3) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiments WHERE kept = 0 ORDER BY timestamp DESC LIMIT ?", (n,)
    ).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_all() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY timestamp ASC"
    ).fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return result


def get_baseline() -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM experiments WHERE id = 1").fetchone()
    result = dict(row) if row else None
    conn.close()
    return result


def get_stats() -> dict:
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) as total, SUM(kept) as kept_count, MIN(val_loss) as best_val_loss "
        "FROM experiments"
    ).fetchone()
    result = dict(row)
    conn.close()
    return result


def get_failure_streak():
    """Count of consecutive non-improvements at the tail of experiment history."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kept FROM experiments ORDER BY id DESC"
        ).fetchall()
    streak = 0
    for r in rows:
        if r["kept"] == 0:
            streak += 1
        else:
            break
    return streak
