#!/usr/bin/env python3
"""Print Jetson/macOS/CPU runtime information without modifying the host."""

import argparse
import json


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    from app.inference.diagnostics import system_report

    print(json.dumps(system_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
