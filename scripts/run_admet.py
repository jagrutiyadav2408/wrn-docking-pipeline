#!/usr/bin/env python3
"""CLI: ADMET-profile the compounds of a finished screen/benchmark.

Reads the docking ranking produced by run_benchmark/run_screen, resolves each
compound's SMILES (from the ligand library source), runs the ADMET profiler, and
writes an ADMET workbook merged with the docking scores.

    python scripts/run_admet.py --config config/screen_9s18.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.admet import AdmetProfiler
from src.config import PipelineConfig
from src.util import configure_logging, progress, resolve_tool, run

logger = logging.getLogger("docksuite.run_admet")


def smiles_for_ids(ids, cfg) -> dict[str, str]:
    """Resolve SMILES for each compound id from the configured library source."""
    source = cfg.get("ligands.source")
    path = Path(cfg.get("ligands.path")) if cfg.get("ligands.path") else None
    if source == "MOL2_DIR":
        obabel = resolve_tool("obabel")
        out: dict[str, str] = {}
        for cid in progress(ids, "smiles"):
            mol2 = path / f"{cid}.mol2"
            if not mol2.is_file():
                continue
            r = run([obabel, str(mol2), "-osmi"], check=False)
            if r.stdout.strip():
                out[cid] = r.stdout.strip().split("\t")[0].split()[0]
        return out
    # SMI / DEKOIS / CUSTOM: id -> smiles straight from the file
    mapping: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[1]] = parts[0]
    return {cid: mapping.get(cid) for cid in ids if mapping.get(cid)}


def main() -> int:
    ap = argparse.ArgumentParser(description="ADMET-profile a finished screen.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ranking", help="Override ranking .xlsx path.")
    args = ap.parse_args()
    configure_logging(logging.INFO)

    cfg = PipelineConfig(args.config)
    bench_dir = Path(cfg.get("output.benchmark_dir"))
    pdb_id = cfg.get("target.pdb_id")
    ranking_path = Path(args.ranking) if args.ranking else bench_dir / f"{pdb_id}_benchmark_results.xlsx"
    if not ranking_path.is_file():
        logger.error("ranking file not found: %s (run the screen first)", ranking_path)
        return 1

    # Prefer the consensus ranking (mean dG over replicates) when present; a
    # single-run ranking is not reproducible enough to select leads from.
    sheets = pd.ExcelFile(ranking_path).sheet_names
    if "Consensus_Ranking" in sheets:
        cons = pd.read_excel(ranking_path, sheet_name="Consensus_Ranking")
        ranking = cons.rename(columns={"Compound_ID": "id", "mean_dG": "dG"})[
            ["id", "dG", "std_dG", "rank_stability"]]
        logger.info("using CONSENSUS ranking (%d compounds, mean dG over replicates)", len(ranking))
    else:
        ranking = pd.read_excel(ranking_path, sheet_name="Ranking")
        logger.warning("no Consensus_Ranking sheet; falling back to single-run Ranking")
    logger.info("loaded %d ranked compounds from %s", len(ranking), ranking_path.name)

    smi = smiles_for_ids(ranking["id"].tolist(), cfg)
    ranking = ranking[ranking["id"].isin(smi)].copy()
    ranking["smiles"] = ranking["id"].map(smi)
    logger.info("resolved SMILES for %d/%d compounds", len(ranking), len(smi) or len(ranking))

    mode = cfg.get("admet.mode", "ALL")
    if mode == "NONE":
        mode = "ALL"                                   # explicit ADMET run overrides NONE
    profiler = AdmetProfiler(mode=mode, top_n=cfg.get("admet.top_n"))
    profile = profiler.profile(ranking[["id", "dG", "smiles"]])

    xlsx = bench_dir / f"{pdb_id}_admet_profile.xlsx"
    leads = profile[profile["Risk_Tier"].isin(["PASS", "MODERATE"])]
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        profile.to_excel(xw, sheet_name="ADMET_Profile", index=False)
        leads.to_excel(xw, sheet_name="Leads", index=False)
        (profile["Risk_Tier"].value_counts().rename_axis("Risk_Tier")
         .reset_index(name="count")).to_excel(xw, sheet_name="Risk_Summary", index=False)

    # terminal summary
    print("\n" + "=" * 60)
    print(f"  ADMET PROFILE — {pdb_id}  ({len(profile)} compounds)")
    print("=" * 60)
    for tier in ("PASS", "MODERATE", "HIGH RISK", "EXCLUDE"):
        print(f"  {tier:<10}: {(profile['Risk_Tier'] == tier).sum()}")
    print("-" * 60)
    print("  Top leads (best dG among PASS/MODERATE):")
    cols = [c for c in ("id", "dG", "MW", "cLogP", "QED", "hERG", "AMES", "DILI",
                        "HIA_Hou", "Risk_Tier") if c in leads.columns]
    print(leads.head(10)[cols].to_string(index=False))
    print(f"\n  Workbook -> {xlsx}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
