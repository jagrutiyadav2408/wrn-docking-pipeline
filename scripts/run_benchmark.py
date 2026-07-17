#!/usr/bin/env python3
"""CLI: run a retrospective benchmark from a JSON config.

    python scripts/run_benchmark.py --config config/benchmark_hivpr.json
    python scripts/run_benchmark.py --config c.json --target.pdb_id 1W82 --ligands.subset_size 40
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmark import BenchmarkRunner


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Turn ``['--target.pdb_id', '1W82', ...]`` into ``{'target.pdb_id': '1W82'}``."""
    overrides: dict[str, str] = {}
    i = 0
    while i < len(pairs):
        key = pairs[i]
        if not key.startswith("--") or "." not in key:
            raise SystemExit(f"unrecognised argument: {key}")
        if i + 1 >= len(pairs):
            raise SystemExit(f"missing value for {key}")
        overrides[key.lstrip("-")] = pairs[i + 1]
        i += 2
    return overrides


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a retrospective docking benchmark.")
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    known, rest = ap.parse_known_args()
    overrides = parse_overrides(rest)
    metrics = BenchmarkRunner(logging.DEBUG if known.verbose else logging.INFO).run(
        known.config, overrides)
    print(f"\nROC-AUC = {metrics.roc_auc:.3f}" if metrics.roc_auc == metrics.roc_auc
          else "\nScreen complete (unlabelled library; no ROC-AUC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
