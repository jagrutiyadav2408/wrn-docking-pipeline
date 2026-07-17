#!/usr/bin/env python3
"""CLI: run a prospective screening campaign from a JSON config.

Identical orchestration to the benchmark entry point; for unlabelled libraries
(no actives/decoys) it produces a ranked hit list and skips ROC/EF metrics.

    python scripts/run_screen.py --config config/wrn_screen.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmark import BenchmarkRunner
from scripts.run_benchmark import parse_overrides  # reuse dotted-override parser


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a prospective screening campaign.")
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    known, rest = ap.parse_known_args()
    metrics = BenchmarkRunner(logging.DEBUG if known.verbose else logging.INFO).run(
        known.config, parse_overrides(rest))
    print(f"\nScreen complete: {metrics.n_decoys} compounds ranked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
