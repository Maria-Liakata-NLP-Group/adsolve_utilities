"""SQLite-backed job store.

Raw dataset content never enters this database -- it lives only in the job
directory and is deleted when the job finishes. Rows hold metadata and, until
delivery, numeric results.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id              TEXT PRIMARY KEY,
  status          TEXT NOT NULL,
  name            TEXT NOT NULL,
  metric_ids      TEXT NOT NULL,
  sensitive       INTEGER NOT NULL,
  job_dir         TEXT NOT NULL,
  metadata        TEXT NOT NULL,
  callback_url    TEXT,
  callback_token  TEXT,
  callback_header TEXT,
  results         TEXT,
  error           TEXT,
  callback_status TEXT,
  submitted_at    TEXT NOT NULL,
  started_at      TEXT,
  finished_at     TEXT,
  delivered_at    TEXT,
  purged_at       TEXT
);
"""


@dataclass(frozen=True)
class Job:
    id: str
    status: str
    name: str
    metric_ids: List[str]
    sensitive: bool
    job_dir: str
    metadata: Dict[str, Any]
    callback_url: Optional[str]
    callback_token: Optional[str]
    callback_header: Optional[str]
    results: Optional[dict]
    error: Optional[str]
    callback_status: Optional[str]
    submitted_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    delivered_at: Optional[str]
    purged_at: Optional[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        status=row["status"],
        name=row["name"],
        metric_ids=json.loads(row["metric_ids"]),
        sensitive=bool(row["sensitive"]),
        job_dir=row["job_dir"],
        metadata=json.loads(row["metadata"]),
        callback_url=row["callback_url"],
        callback_token=row["callback_token"],
        callback_header=row["callback_header"],
        results=json.loads(row["results"]) if row["results"] else None,
        error=row["error"],
        callback_status=row["callback_status"],
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        delivered_at=row["delivered_at"],
        purged_at=row["purged_at"],
    )


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, name: str, metric_ids: List[str], sensitive: bool,
               job_dir: str, metadata: Dict[str, Any],
               callback_url: Optional[str],
               callback_token: Optional[str] = None,
               callback_header: Optional[str] = None,
               job_id: Optional[str] = None) -> str:
        """Insert a queued job. The row is complete the moment it becomes visible.

        Callers pass `job_id` when the job directory is named after it, so the
        inputs can be written before the insert -- the worker must never be able
        to claim a job whose job_dir is not yet set.
        """
        job_id = job_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (id, status, name, metric_ids, sensitive, job_dir,
                                     metadata, callback_url, callback_token,
                                     callback_header, callback_status, submitted_at)
                   VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, name, json.dumps(metric_ids), int(sensitive), job_dir,
                 json.dumps(metadata), callback_url, callback_token, callback_header,
                 "pending" if callback_url else "not_configured", _now()),
            )
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_job(row) if row else None

    def claim_next(self) -> Optional[Job]:
        """Atomically take the oldest queued job and mark it running."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY submitted_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (_now(), row["id"]),
            )
            conn.execute("COMMIT")
            updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return _to_job(updated)

    def mark_succeeded(self, job_id: str, results: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='succeeded', results=?, finished_at=? WHERE id=?",
                (json.dumps(results), _now(), job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error, _now(), job_id),
            )

    def set_callback_status(self, job_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET callback_status=? WHERE id=?", (status, job_id))

    def mark_delivered(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET callback_status='sent', delivered_at=? WHERE id=?",
                (_now(), job_id),
            )

    def purge(self, job_id: str) -> None:
        """Drop results but keep the row, so GET can report 'delivered, purged'."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET results=NULL, purged_at=? WHERE id=?", (_now(), job_id)
            )

    def sweep_orphans(self) -> int:
        """Fail jobs left 'running' by a crash or restart. Returns how many."""
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET status='failed', finished_at=?,
                          error='Job was interrupted by a server restart.'
                   WHERE status='running'""",
                (_now(),),
            )
            return cursor.rowcount

    def expired(self, now: datetime, retention_hours: int) -> List[Job]:
        cutoff = (now - timedelta(hours=retention_hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            ).fetchall()
        return [_to_job(row) for row in rows]

    def delete(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
