#!/usr/bin/env python3
"""Project entry point. Running without a command never starts training."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "serve": "app.cli.serve",
    "infer": "app.cli.infer",
    "benchmark": "app.cli.benchmark",
    "optimize": "app.cli.optimize",
    "diagnose": "app.cli.diagnose",
    "merge": "app.cli.merge",
    "validate": "scripts.data.validate",
    "import-external": "scripts.data.import_external_sql",
    "import-kaggle": "scripts.data.import_kaggle_sql",
    "train": "training.train",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    module = importlib.import_module(COMMANDS[args.command])
    return module.main(argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
