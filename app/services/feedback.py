"""Durable evaluation records. Predictions are always taken from the server."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class EvaluationStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            legacy = any(
                column[1] == "demo" for column in db.execute("PRAGMA table_info(evaluations)")
            )
            if legacy:
                # Preserve the original table when migrating existing user records.
                db.execute("ALTER TABLE evaluations RENAME TO evaluations_backup_v1")
            db.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    input TEXT NOT NULL, payload TEXT NOT NULL,
                    feedback TEXT
                )
            """)
            if legacy:
                db.execute("""
                    INSERT INTO evaluations (id, created_at, input, payload, feedback)
                    SELECT id, created_at, input, payload, feedback
                    FROM evaluations_backup_v1 WHERE demo=0
                """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, text: str, payload: dict) -> str:
        identifier = uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO evaluations (id, created_at, input, payload) VALUES (?, ?, ?, ?)",
                (
                    identifier,
                    datetime.now(UTC).isoformat(),
                    text,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        return identifier

    def get(self, identifier: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM evaluations WHERE id=?", (identifier,)).fetchone()
        return self._record(row) if row else None

    @staticmethod
    def _record(row) -> dict:
        payload = json.loads(row[3])
        if isinstance(payload.get("runtime"), dict):
            payload["runtime"].pop("demo", None)
        return {
            "request_id": row[0],
            "created_at": row[1],
            "input": row[2],
            **payload,
            "feedback": json.loads(row[4]) if row[4] else None,
        }

    def feedback(self, identifier: str, feedback: dict) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE evaluations SET feedback=? WHERE id=?",
                (json.dumps(feedback, ensure_ascii=False), identifier),
            )
            return cursor.rowcount == 1

    def records(self):
        with self.connect() as db:
            for row in db.execute("SELECT * FROM evaluations ORDER BY created_at"):
                yield self._record(row)

    def summary(self) -> dict:
        with self.connect() as db:
            total, reviewed, correct = db.execute("""
                SELECT COUNT(*), COUNT(feedback),
                  COALESCE(SUM(json_extract(feedback, '$.correct') = 1), 0)
                FROM evaluations
            """).fetchone()
        return {
            "total": total,
            "reviewed": reviewed,
            "correct": correct,
            "human_accuracy": correct / reviewed if reviewed else None,
        }
