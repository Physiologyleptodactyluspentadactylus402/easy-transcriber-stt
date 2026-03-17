from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from app.providers.base import Job


class History:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    input_filenames TEXT,
                    provider_name TEXT,
                    model_id TEXT,
                    status TEXT,
                    created_at REAL,
                    duration_sec REAL,
                    output_paths TEXT,
                    error_message TEXT
                )
            """)

    def save(self, job: Job, duration_sec: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (id, input_filenames, provider_name, model_id, status,
                     created_at, duration_sec, output_paths, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    json.dumps([p.name for p in job.input_files]),
                    job.provider_name,
                    job.model_id,
                    job.status,
                    job.created_at,
                    duration_sec,
                    json.dumps([str(p).replace("\\", "/") for p in job.output_files]),
                    job.error_message,
                ),
            )

    def list(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["input_filenames"] = json.loads(d["input_filenames"])
            d["output_paths"] = json.loads(d["output_paths"])
            result.append(d)
        return result

    def delete(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
