"""Configuration and command routing must work independently of the shell directory."""

import subprocess
import sys
from pathlib import Path

import pytest

from app import paths
from app.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_adapter_is_complete():
    adapter = REPO_ROOT / "model_adapter"

    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").stat().st_size > 1_000_000


def test_configured_relative_path_is_project_relative(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    assert paths.project_path("storage/feedback/evaluation.sqlite3") == (
        project / "storage/feedback/evaluation.sqlite3"
    )


def test_absolute_path_is_preserved(tmp_path):
    assert paths.project_path(tmp_path / "custom.sqlite3") == tmp_path / "custom.sqlite3"


def test_private_env_takes_priority_over_development_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("BASE_MODEL=private/model\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/development.env").write_text("BASE_MODEL=dev/model\n")
    monkeypatch.delenv("BASE_MODEL", raising=False)
    assert paths.default_env_file() == tmp_path / ".env"
    assert get_settings().base_model == "private/model"


def test_development_profile_is_used_without_private_env(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/development.env").write_text("BASE_MODEL=dev/model\n")
    monkeypatch.delenv("BASE_MODEL", raising=False)
    assert get_settings().base_model == "dev/model"


def test_hugging_face_id_is_not_treated_as_a_local_path():
    assert paths.model_reference("organization/model") == "organization/model"
    assert paths.model_reference("") == ""


def test_local_model_directory_is_project_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    model = tmp_path / "storage/models/merged"
    model.mkdir(parents=True)
    assert paths.model_reference("storage/models/merged") == str(model)
    assert paths.model_reference("./storage/models/missing") == str(
        tmp_path / "storage/models/missing"
    )


@pytest.mark.parametrize(
    "command",
    [
        "serve",
        "infer",
        "benchmark",
        "optimize",
        "diagnose",
        "merge",
        "train",
        "validate",
        "import-external",
        "import-kaggle",
    ],
)
def test_all_commands_show_help_from_another_directory(command, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_serve_rejects_missing_explicit_env_file(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "serve",
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 2
    assert "ayar dosyasi bulunamadi" in proc.stderr
