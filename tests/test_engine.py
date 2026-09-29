from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from app.config import Settings  # noqa: E402
from app.errors import ModelInferenceError  # noqa: E402
from app.inference.engine import InferenceEngine  # noqa: E402
from app.inference.extractor import ExtractionOutcome, GenerationTiming  # noqa: E402
from app.inference.parsing import ExtractionResult  # noqa: E402
from app.schemas.query import QueryRequest  # noqa: E402


def test_engine_loads_prepares_and_warms_up_once(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        BASE_MODEL="test",
        ADAPTER_PATH=str(tmp_path),
        PROMPT_FORMAT="plain",
        WARMUP_RUNS=1,
        ALLOW_RETRY=False,
    )
    model = SimpleNamespace(model=Mock(), tokenizer=Mock(), device="cpu", dtype="float32")
    load = Mock(return_value=model)
    adapter = Mock(return_value=model.model)
    prepare = Mock()
    outcome = ExtractionOutcome(
        ExtractionResult.ok(QueryRequest(operation="count", target="sales")),
        GenerationTiming(1, 2, 3, total_ms=2),
        False,
    )
    extract = Mock(return_value=outcome)
    monkeypatch.setattr("app.inference.model_loader.load_base_model", load)
    monkeypatch.setattr("app.inference.model_loader.load_adapter", adapter)
    monkeypatch.setattr("app.inference.model_loader.prepare_for_inference", prepare)
    monkeypatch.setattr("app.inference.extractor.extract", extract)
    engine = InferenceEngine(settings)
    engine.start()
    engine.start()
    engine.extract("ilk")
    engine.extract("ikinci")
    load.assert_called_once()
    adapter.assert_called_once()
    prepare.assert_called_once_with(model, merge_adapter=True)
    assert extract.call_count == 3
    assert engine.info()["cold_request_ms"] == 2
    assert extract.call_args.kwargs["allow_retry"] is False
    assert extract.call_args.kwargs["max_input_tokens"] == 1024


def test_engine_resolves_relative_adapter_from_project(monkeypatch, tmp_path):
    monkeypatch.setattr("app.paths.PROJECT_ROOT", tmp_path / "project")
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        BASE_MODEL="org/model",
        ADAPTER_PATH="storage/models/latest",
        WARMUP_RUNS=0,
        PROMPT_FORMAT="plain",
    )
    loaded = SimpleNamespace(model=Mock(), tokenizer=Mock(), device="cpu", dtype="float32")
    load = Mock(return_value=loaded)
    adapter = Mock(return_value=loaded.model)
    monkeypatch.setattr("app.inference.model_loader.load_base_model", load)
    monkeypatch.setattr("app.inference.model_loader.load_adapter", adapter)
    monkeypatch.setattr("app.inference.model_loader.prepare_for_inference", Mock())
    InferenceEngine(settings).start()
    assert load.call_args.args[0] == "org/model"
    assert adapter.call_args.args[1] == tmp_path / "project/storage/models/latest"


def test_unstarted_engine_fails_cleanly():
    engine = InferenceEngine(Settings(_env_file=None))
    with pytest.raises(ModelInferenceError):
        engine.extract("sorgu")
