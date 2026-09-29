import json
import sqlite3

from app.services.feedback import EvaluationStore


def test_legacy_records_migrate_without_losing_original_table(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE evaluations (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, input TEXT NOT NULL,
                payload TEXT NOT NULL, demo INTEGER NOT NULL, feedback TEXT
            )
        """)
        for identifier, sample in [("real", False), ("sample", True)]:
            payload = {"runtime": {"demo": sample, "model": "test"}, "data": {"target": "sales"}}
            db.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    "2026-01-01",
                    "prompt",
                    json.dumps(payload),
                    int(sample),
                    json.dumps({"correct": True}),
                ),
            )
    store = EvaluationStore(str(path))
    assert store.summary() == {"total": 1, "reviewed": 1, "correct": 1, "human_accuracy": 1.0}
    record = store.get("real")
    assert record["feedback"]["correct"] is True
    assert "demo" not in record["runtime"]
    assert store.get("sample") is None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM evaluations_backup_v1").fetchone()[0] == 2
        assert "demo" not in [r[1] for r in db.execute("PRAGMA table_info(evaluations)")]
    assert EvaluationStore(str(path)).summary() == store.summary()


def test_new_schema_accepts_records_after_migration(tmp_path):
    store = EvaluationStore(str(tmp_path / "feedback.sqlite3"))
    identifier = store.add("prompt", {"runtime": {"model": "test"}, "data": {"target": "sales"}})
    assert store.get(identifier)["input"] == "prompt"
    assert store.feedback(identifier, {"correct": False})
    assert list(store.records())[0]["feedback"] == {"correct": False}
    assert store.summary()["human_accuracy"] == 0
