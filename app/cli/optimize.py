#!/usr/bin/env python3
"""Compare warm single-request latency and accuracy in fresh model processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.paths import PROJECT_ROOT, project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path", required=True, type=Path)
    parser.add_argument("--test_file", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--output_dir", type=Path, default=project_path("artifacts/benchmarks/optimization")
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit pozitif olmali")
    # Child processes run from the project root, so preserve caller-relative paths first.
    args.adapter_path = args.adapter_path.resolve()
    args.test_file = args.test_file.resolve()
    args.output_dir = args.output_dir.resolve()
    if Path(args.model_name).exists():
        args.model_name = str(Path(args.model_name).resolve())
    root = args.output_dir / datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    scenarios = {
        "adapter": [],
        "merged": ["--merge_adapter"],
        "merged_sdpa": ["--merge_adapter", "--attention", "sdpa"],
        "merged_128_tokens": ["--merge_adapter", "--max_new_tokens", "128"],
    }
    results = {}
    for name, options in scenarios.items():
        output = root / name
        command = [
            sys.executable,
            "-m",
            "app.cli.benchmark",
            "--model_name",
            args.model_name,
            "--adapter_path",
            str(args.adapter_path),
            "--test_file",
            str(args.test_file),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch_size",
            "1",
            "--warmup_runs",
            "1",
            "--limit",
            str(args.limit),
            "--sample_seed",
            str(args.sample_seed),
            "--output_dir",
            str(output),
            "--dump_predictions",
            str(output / "predictions.jsonl"),
            *options,
        ]
        print(f"\nKarsilastirma: {name}", flush=True)
        process = subprocess.run(command, check=False, cwd=PROJECT_ROOT)
        reports = sorted(output.glob("*.json"))
        results[name] = {
            "exit_code": process.returncode,
            "command": command,
            "report": json.loads(reports[-1].read_text()) if reports else None,
        }
    baseline = results["adapter"]["report"]
    eligible = []
    if baseline:
        for name, item in results.items():
            report = item["report"]
            if not report:
                continue
            retained = all(
                report[metric] >= baseline[metric]
                for metric in ("exact_match_rate", "semantic_match_rate", "schema_validity_rate")
            )
            item["accuracy_retained"] = retained
            if retained:
                eligible.append((report["latency"]["mean_ms"], name))
    summary = {
        "scenarios": results,
        "recommended": min(eligible)[1] if eligible else None,
        "note": "Ayni sinirli setteki olcum; tum test ve OOD setiyle dogrulanmalidir.",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nKarsilastirma raporu: {root / 'comparison.json'}")
    return 1 if any(item["exit_code"] for item in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
