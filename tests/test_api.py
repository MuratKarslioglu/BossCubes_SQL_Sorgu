import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.errors import ModelInferenceError
from app.inference.extractor import ExtractionOutcome, GenerationTiming
from app.inference.parsing import ExtractionResult
from app.schemas.query import QueryRequest


class FakeEngine:
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0
        self.error = None
        self.entered = None
        self.release = None

    def start(self):
        self.calls += 1

    def info(self):
        return {"runtime": "fake"}

    def extract(self, text):
        if self.entered:
            self.entered.set()
            self.release.wait(timeout=5)
        if self.error:
            raise self.error
        return ExtractionOutcome(
            ExtractionResult.ok(QueryRequest(operation="count", target="sales")),
            GenerationTiming(5.0, 10, 30, total_ms=6.0),
            False,
        )


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, FEEDBACK_DB=str(tmp_path / "feedback.sqlite3"))
    app = create_app(settings, engine_factory=FakeEngine)
    with TestClient(app) as client:
        yield client


def test_model_loaded_once_and_requests_are_saved(client):
    for _ in range(2):
        response = client.post("/extract", json={"text": "Satış sayısı"})
        assert response.status_code == 200
        assert len(response.json()["request_id"]) == 32
        assert response.json()["timing"]["total_ms"] == 6
    assert client.app.state.engine.calls == 1
    assert client.get("/health").json()["evaluation"]["total"] == 2


def test_relative_feedback_path_does_not_depend_on_working_directory(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("app.paths.PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, FEEDBACK_DB="storage/feedback/evaluation.sqlite3")
    with TestClient(create_app(settings, engine_factory=FakeEngine)) as client:
        assert client.get("/health").status_code == 200
    assert (project / "storage/feedback/evaluation.sqlite3").is_file()
    assert not (tmp_path / "storage").exists()


@pytest.mark.parametrize("text", ["   ", "\u200b", "x" * 4001])
def test_invalid_inputs_are_not_saved(client, text):
    assert client.post("/extract", json={"text": text}).status_code == 422
    assert client.get("/health").json()["evaluation"]["total"] == 0


def test_expected_json_comparison_and_feedback_export(client):
    expected = QueryRequest(operation="sum", target="sales_amount").model_dump(mode="json")
    response = client.post("/extract", json={"text": "Toplam satış", "expected": expected}).json()
    assert not response["comparison"]["semantic_match"]
    assert response["comparison"]["different_fields"] == ["operation", "target"]
    feedback = client.post(
        "/feedback",
        json={
            "request_id": response["request_id"],
            "correct": False,
            "category": "operation",
            "note": "Toplam istedim",
        },
    )
    assert feedback.status_code == 200
    record = json.loads(client.get("/export").text)
    assert record["data"]["operation"] == "count"
    assert record["feedback"]["note"] == "Toplam istedim"
    assert record["feedback"]["comparison"]["semantic_match"] is False
    assert feedback.json()["evaluation"]["human_accuracy"] == 0


def test_feedback_cannot_forge_prediction_or_reference_missing_record(client):
    assert (
        client.post("/feedback", json={"request_id": "x" * 32, "correct": True}).status_code == 404
    )
    response = client.post("/extract", json={"text": "Satışlar"}).json()
    assert (
        client.post(
            "/feedback",
            json={
                "request_id": response["request_id"],
                "correct": True,
                "data": {"operation": "sum"},
            },
        ).status_code
        == 422
    )


def test_feedback_survives_restart_and_does_not_double_count(tmp_path):
    settings = Settings(_env_file=None, FEEDBACK_DB=str(tmp_path / "persistent.sqlite"))
    with TestClient(create_app(settings, FakeEngine)) as client:
        result = client.post("/extract", json={"text": "Satışlar"}).json()
        for correct in [False, True]:
            client.post("/feedback", json={"request_id": result["request_id"], "correct": correct})
    with TestClient(create_app(settings, FakeEngine)) as client:
        assert client.get("/health").json()["evaluation"] == {
            "total": 1,
            "reviewed": 1,
            "correct": 1,
            "human_accuracy": 1.0,
        }


def test_csv_neutralizes_formula_and_quotes_multiline_prompt(client):
    client.post("/extract", json={"text": '=HYPERLINK("bad")'})
    assert "'=HYPERLINK" in client.get("/export?format=csv").text


def test_generation_failure_releases_lock_and_hides_traceback(client):
    client.app.state.engine.error = ModelInferenceError("secret diagnostic")
    failed = client.post("/extract", json={"text": "Satışlar"})
    assert failed.status_code == 500
    assert "secret" not in failed.text
    client.app.state.engine.error = None
    assert client.post("/extract", json={"text": "Satışlar"}).status_code == 200


def test_second_request_is_rejected_while_gpu_is_busy(client):
    engine = client.app.state.engine
    engine.entered = threading.Event()
    engine.release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/extract", json={"text": "Satışlar"})
        assert engine.entered.wait(timeout=3)
        try:
            assert client.post("/extract", json={"text": "Bir başka istek"}).status_code == 429
        finally:
            engine.release.set()
        assert first.result().status_code == 200


def test_failed_startup_keeps_ui_and_health_accessible(tmp_path):
    class BrokenEngine(FakeEngine):
        def start(self):
            raise RuntimeError("private path")

    settings = Settings(_env_file=None, FEEDBACK_DB=str(tmp_path / "broken.sqlite"))
    with TestClient(create_app(settings, BrokenEngine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 503
        assert client.post("/extract", json={"text": "Satışlar"}).status_code == 503
        assert "private path" not in client.get("/health").text
