"""Resolve configured paths independently of the process working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    """Relative configuration paths belong to the project, not the shell's cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_env_file() -> Path:
    """Prefer private settings; otherwise use the checked-in development profile."""
    private_env = PROJECT_ROOT / ".env"
    return private_env if private_env.is_file() else PROJECT_ROOT / "configs" / "development.env"


def model_reference(value: str) -> str:
    """Anchor local configured models while leaving Hugging Face IDs unchanged."""
    if not value:
        return value
    path = project_path(value)
    if path.exists() or Path(value).is_absolute() or value.startswith(("./", "../", "~")):
        return str(path)
    return value
