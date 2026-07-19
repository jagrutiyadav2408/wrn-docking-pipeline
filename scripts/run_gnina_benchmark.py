#!/usr/bin/env python3
"""CLI: run the retrospective benchmark with the GNINA CNN scorer.

Requires a working `gnina` (Linux/CUDA, or Docker/WSL2 on Windows) - see
README_GNINA.md. If gnina is unreachable this reports the AutoDock baseline and
exits without inventing CNN numbers.

    python scripts/run_gnina_benchmark.py --config config/benchmark_gnina.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmark_gnina import run_dekois_benchmark_gnina


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the DEKOIS/DUD-E benchmark with gnina.")
    ap.add_argument("--config", required=True, help="Path to the gnina benchmark JSON config.")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    result = run_dekois_benchmark_gnina(cfg)
    if not result.get("gnina_available"):
        print("\ngnina unavailable - AutoDock baseline reported only.")
        return 1
    print(f"\ngnina CNN ROC-AUC = {result['gnina_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
