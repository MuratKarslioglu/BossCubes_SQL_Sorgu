"""Read-only environment diagnostics; never change clocks or power mode."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
import subprocess
from pathlib import Path


def system_report() -> dict:
    report = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {},
        "warnings": [],
    }
    for name in ("torch", "transformers", "peft", "trl", "fastapi"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    version = report["packages"].get("torch") or ""
    if match := re.match(r"(\d+)\.(\d+)", version):
        if tuple(map(int, match.groups())) < (2, 5):
            report["warnings"].append("Bu Transformers 5.17 yigini PyTorch >= 2.5 gerektirir.")
    for label, path in (("l4t", "/etc/nv_tegra_release"), ("board", "/proc/device-tree/model")):
        candidate = Path(path)
        if candidate.exists():
            report[label] = candidate.read_text().strip("\x00\n")
    try:
        import torch

        report["cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            arch = f"sm_{capability[0]}{capability[1]}"
            arch_list = torch.cuda.get_arch_list()
            report["gpu"] = {
                "name": torch.cuda.get_device_name(),
                "capability": list(capability),
                "compiled_architectures": arch_list,
                "native_arch_present": arch in arch_list,
                "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            }
    except Exception as exc:
        report["torch_error"] = type(exc).__name__
    if shutil.which("nvpmodel"):
        try:
            process = subprocess.run(
                ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=False
            )
            report["power_mode"] = process.stdout.strip() or process.stderr.strip()
        except (OSError, subprocess.TimeoutExpired):
            report["power_mode"] = "unavailable"
    return report
