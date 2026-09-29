import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from app.cli import merge as merge_adapter
from app.cli import optimize


def test_entrypoint_without_args_does_not_start_training(monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("help must not import command modules")

    monkeypatch.setattr(main.importlib, "import_module", forbidden)
    assert main.main([]) == 0
    assert "serve" in capsys.readouterr().out


def test_entrypoint_forwards_command_arguments(monkeypatch):
    received = []
    command = SimpleNamespace(main=lambda args: received.extend(args) or 7)
    monkeypatch.setattr(main.importlib, "import_module", lambda name: command)
    assert main.main(["serve", "--port", "8001"]) == 7
    assert received == ["--port", "8001"]


def test_optimizer_rejects_faster_but_less_accurate_candidates(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        directory = Path(command[command.index("--output_dir") + 1])
        directory.mkdir(parents=True)
        index = len(commands) - 1
        accuracy = [1.0, 1.0, 0.99, 0.9][index]
        report = {
            "exact_match_rate": accuracy,
            "semantic_match_rate": accuracy,
            "schema_validity_rate": accuracy,
            "latency": {"mean_ms": [100, 90, 30, 20][index]},
        }
        (directory / "report.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(optimize.subprocess, "run", run)
    assert (
        optimize.main(
            [
                "--model_name",
                "test",
                "--adapter_path",
                str(tmp_path / "adapter"),
                "--test_file",
                str(tmp_path / "test.jsonl"),
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    report = json.loads(next((tmp_path / "out").glob("*/comparison.json")).read_text())
    assert report["recommended"] == "merged"
    assert report["scenarios"]["merged_sdpa"]["accuracy_retained"] is False
    assert report["scenarios"]["merged_128_tokens"]["accuracy_retained"] is False
    assert all(c[c.index("--sample_seed") + 1] == "42" for c in commands)
    assert all(c[c.index("--batch_size") + 1] == "1" for c in commands)


def test_optimizer_reports_subprocess_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(
        optimize.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    assert (
        optimize.main(
            [
                "--model_name",
                "test",
                "--adapter_path",
                str(tmp_path / "adapter"),
                "--test_file",
                str(tmp_path / "test.jsonl"),
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    report = json.loads(next((tmp_path / "out").glob("*/comparison.json")).read_text())
    assert report["recommended"] is None


def test_merge_refuses_to_overwrite_existing_model(tmp_path):
    existing = tmp_path / "weights.safetensors"
    existing.write_text("preserve")
    with pytest.raises(SystemExit):
        merge_adapter.main(
            [
                "--model_name",
                "unused",
                "--adapter_path",
                "unused",
                "--output_dir",
                str(tmp_path),
            ]
        )
    assert existing.read_text() == "preserve"
