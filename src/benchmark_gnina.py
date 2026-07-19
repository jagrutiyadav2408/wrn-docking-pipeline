#!/usr/bin/env python3
"""
benchmark_gnina.py -- retrospective DEKOIS 2.0 validation with the GNINA backend.

Reuses the already-prepared receptor.pdbqt and ligand pdbqts from the AutoDock-GPU
run (target-agnostic: everything comes from config). Runs gnina (dock or rescore),
computes ROC-AUC / EF / BEDROC, and compares against the AutoDock-GPU baseline.

HONEST-EXECUTION CONTRACT
  * If gnina is not runnable on this host, the script reports the AutoDock-GPU
    baseline, prints the install hint, and exits WITHOUT inventing gnina numbers.
  * All gnina metrics printed are computed from real gnina output only.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .gnina_interface import (GninaInterface, gate_charge_cap, gate_shape_filter,
                              gate_torsional_integrity, radius_of_gyration)

logger = logging.getLogger("benchmark_gnina")
_ROW = re.compile(r"^\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|", re.M)


# --------------------------------------------------------------------------- #
# metrics (shared definitions with the AutoDock benchmark)
# --------------------------------------------------------------------------- #
def enrichment_factor(scores, labels, frac: float) -> float:
    order = np.argsort(-np.asarray(scores, float))
    y = np.asarray(labels)[order]
    n_top = max(1, int(round(frac * len(y))))
    hit = y.sum() / len(y)
    return float((y[:n_top].sum() / n_top) / hit) if hit else 0.0


def bedroc(scores, labels, alpha: float = 20.0) -> float:
    order = np.argsort(-np.asarray(scores, float))
    y = np.asarray(labels)[order]
    N, n = len(y), int(y.sum())
    if n == 0 or n == N:
        return float("nan")
    Ra = n / N
    ranks = np.where(y == 1)[0] + 1
    rie = np.sum(np.exp(-alpha * ranks / N)) / (
        n / N * (1 - math.exp(-alpha)) / (math.exp(alpha / N) - 1))
    fac = Ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * Ra))
    return float(rie * fac + 1 / (1 - math.exp(alpha * (1 - Ra))))


def all_metrics(scores, labels) -> dict:
    from sklearn.metrics import roc_auc_score
    return {"ROC_AUC": float(roc_auc_score(labels, scores)),
            "EF_1pct": enrichment_factor(scores, labels, 0.01),
            "EF_5pct": enrichment_factor(scores, labels, 0.05),
            "EF_10pct": enrichment_factor(scores, labels, 0.10),
            "BEDROC": bedroc(scores, labels, 20.0)}


# --------------------------------------------------------------------------- #
# AutoDock-GPU baseline (reuse existing .dlg files)
# --------------------------------------------------------------------------- #
def autodock_largest_cluster_dG(dlg: Path) -> Optional[float]:
    if not dlg.is_file():
        return None
    rows = [(float(m[2]), int(m[5])) for m in _ROW.finditer(dlg.read_text(errors="ignore"))]
    return max(rows, key=lambda x: (x[1], -x[0]))[0] if rows else None


# --------------------------------------------------------------------------- #
# main benchmark
# --------------------------------------------------------------------------- #
def run_dekois_benchmark_gnina(config: dict) -> dict:
    """Run the DEKOIS benchmark with gnina and compare to AutoDock-GPU.
    All paths and target identity come from `config`; nothing is hardcoded."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bench = Path(config["BENCH_DIR"])
    ligdir = Path(config.get("LIGAND_PDBQT_DIR", bench / "ligands_pdbqt"))
    ad_out = Path(config["AUTODOCK_DLG_DIR"])
    gnina_out = Path(config.get("GNINA_OUTPUT_DIR", bench.parent / "gnina_output"))
    gnina_out.mkdir(parents=True, exist_ok=True)
    figdir = Path(config.get("FIGURE_DIR", bench.parent / "figures"))
    figdir.mkdir(parents=True, exist_ok=True)
    receptor = Path(config["RECEPTOR_PDBQT"])
    center = tuple(config["BOX_CENTER"])
    L = float(config["BOX_SIZE"])
    box = (L, L, L)
    actives = set(config["ACTIVE_IDS"])
    compounds = list(config["ACTIVE_IDS"]) + list(config["DECOY_IDS"])
    ad_baseline_auc = float(config.get("AUTODOCK_AUC", float("nan")))
    native_rg = config.get("NATIVE_RG")

    gi = GninaInterface(config)
    have_gnina = gi.validate_installation()

    # ---------- AutoDock-GPU baseline from existing dlgs ---------------- #
    ad_rows = []
    for cid in compounds:
        dG = autodock_largest_cluster_dG(ad_out / f"{cid}.dlg")
        if dG is not None:
            ad_rows.append({"Compound_ID": cid, "is_active": int(cid in actives), "dG": dG})
    ad_df = pd.DataFrame(ad_rows)
    if not ad_df.empty:
        ad_metrics = all_metrics((-ad_df["dG"]).to_numpy(), ad_df["is_active"].to_numpy())
        logger.info("AutoDock-GPU baseline ROC-AUC=%.3f (n=%d)", ad_metrics["ROC_AUC"], len(ad_df))
    else:
        ad_metrics = {}
        logger.warning("no AutoDock dlgs found under %s", ad_out)

    if not have_gnina:
        logger.error("GNINA UNAVAILABLE -- reporting AutoDock baseline only.\n%s", gi.install_hint())
        _print_summary(ad_metrics, None, None, ad_baseline_auc)
        return {"gnina_available": False, "autodock_metrics": ad_metrics,
                "gnina_metrics": None, "install_hint": gi.install_hint()}

    # ---------- quality gates + gnina execution ------------------------ #
    gi.write_versions(bench.parent / "versions.txt",
                      gpu=config.get("GPU", "unknown"), cuda=config.get("CUDA", "unknown"))
    mode = config.get("GNINA_MODE", "dock")
    gnina_rows, shape_log = [], []
    try:
        from tqdm import tqdm
        it = tqdm(compounds, desc=f"gnina:{mode}")
    except Exception:
        it = compounds

    for cid in it:
        lig = ligdir / f"{cid}.pdbqt"
        if not lig.is_file():
            logger.warning("missing ligand pdbqt: %s", cid)
            continue
        gate_torsional_integrity(lig)
        gate_charge_cap(lig, cap=float(config.get("CHARGE_CAP", 1.0)))
        rg, flagged = gate_shape_filter(lig, native_rg, float(config.get("RG_FACTOR", 1.5)))
        if flagged:
            shape_log.append(f"{cid}\tRg={rg:.2f}")

        prefix = str(gnina_out / cid)
        if mode == "rescore":
            res = gi.run_gnina_rescore(str(receptor), str(lig), str(lig))
        else:
            res = gi.run_gnina_dock(str(receptor), str(lig), prefix, center, box)
        caff = res.get("cnn_affinity") if res.get("cnn_affinity") is not None else res.get("CNNaffinity")
        if caff is None:
            logger.warning("no CNN score for %s (%s)", cid, res.get("error"))
            continue
        gnina_rows.append({"Compound_ID": cid, "is_active": int(cid in actives),
                           "CNNaffinity": caff, "CNNscore": res.get("cnn_score"),
                           "minimizedAffinity": res.get("minimized_affinity")})

    gn_df = pd.DataFrame(gnina_rows)
    if gn_df.empty:
        logger.error("gnina produced no scores; cannot compute metrics")
        return {"gnina_available": True, "autodock_metrics": ad_metrics,
                "gnina_metrics": None, "error": "no gnina scores"}

    gn_metrics = all_metrics(gn_df["CNNaffinity"].to_numpy(), gn_df["is_active"].to_numpy())

    # ---------- comparison + outputs ----------------------------------- #
    merged = gi.compare_with_autodock(ad_df, gn_df) if not ad_df.empty else pd.DataFrame()
    figs = _make_figures(ad_df, gn_df, merged, ad_metrics, gn_metrics, figdir)

    xlsx = bench.parent / "gnina_benchmark_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        gn_df.sort_values("CNNaffinity", ascending=False).to_excel(xw, "Gnina_Ranking", index=False)
        if not ad_df.empty:
            ad_df.sort_values("dG").to_excel(xw, "AutoDock_Ranking", index=False)
        pd.DataFrame([{"Backend": "AutoDock-GPU", **ad_metrics},
                      {"Backend": "Gnina-CNN", **gn_metrics}]).to_excel(xw, "Metrics", index=False)
        if not merged.empty:
            merged.to_excel(xw, "Comparison", index=False)
    if not merged.empty:
        merged.to_csv(bench.parent / "gnina_comparison.csv", index=False)

    _print_summary(ad_metrics, gn_metrics, merged, ad_baseline_auc)
    return {"gnina_available": True,
            "gnina_auc": gn_metrics["ROC_AUC"],
            "autodock_auc": ad_metrics.get("ROC_AUC"),
            "improvement": (gn_metrics["ROC_AUC"] / ad_metrics["ROC_AUC"]) if ad_metrics.get("ROC_AUC") else None,
            "top_n_overlap": merged.attrs.get("top_n_overlap") if not merged.empty else None,
            "rank_correlation": merged.attrs.get("spearman") if not merged.empty else None,
            "gnina_metrics": gn_metrics, "autodock_metrics": ad_metrics,
            "figure_paths": figs}


def _make_figures(ad_df, gn_df, merged, ad_m, gn_m, figdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    figs = []

    # ROC comparison
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    if not ad_df.empty:
        fpr, tpr, _ = roc_curve(ad_df["is_active"], -ad_df["dG"])
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AutoDock-GPU (AUC={ad_m['ROC_AUC']:.3f})")
    fpr, tpr, _ = roc_curve(gn_df["is_active"], gn_df["CNNaffinity"])
    ax.plot(fpr, tpr, color="#d62728", lw=2, label=f"Gnina-CNN (AUC={gn_m['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="k", alpha=.4, label="random (0.50)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC: AutoDock vs Gnina")
    ax.legend(loc="lower right"); fig.tight_layout()
    p = figdir / "figure_roc_comparison.png"; fig.savefig(p); plt.close(fig); figs.append(str(p))

    if not merged.empty:
        # score scatter
        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
        for lab, c in [("Active", "#d62728"), ("Decoy", "#1f77b4")]:
            sub = merged[merged["is_active"] == (1 if lab == "Active" else 0)]
            ax.scatter(sub["dG"], sub["CNNaffinity"], c=c, s=40, edgecolor="k", lw=.3, label=lab)
        ax.set_xlabel("AutoDock-GPU dG (kcal/mol)"); ax.set_ylabel("Gnina CNNaffinity")
        ax.set_title(f"AD4 vs Gnina  (Pearson r={merged.attrs.get('pearson', float('nan')):.2f})")
        ax.legend(); fig.tight_layout()
        p = figdir / "figure_gnina_comparison.png"; fig.savefig(p); plt.close(fig); figs.append(str(p))

        # rank-rank
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.scatter(merged["rank_ad"], merged["rank_gnina"], c="#555", s=40)
        ax.set_xlabel("AutoDock rank"); ax.set_ylabel("Gnina rank")
        ax.set_title(f"Rank-rank (Spearman rho={merged.attrs.get('spearman', float('nan')):.2f})")
        fig.tight_layout()
        p = figdir / "figure_rank_correlation.png"; fig.savefig(p); plt.close(fig); figs.append(str(p))
    return figs


def _print_summary(ad_m: dict, gn_m: Optional[dict], merged, ad_baseline_auc: float):
    print("\n" + "=" * 60)
    print("  BACKEND COMPARISON  (DEKOIS benchmark)")
    print("=" * 60)
    ad_auc = ad_m.get("ROC_AUC", ad_baseline_auc)
    print(f"  AutoDock-GPU AUC:   {ad_auc:.3f}" if ad_auc == ad_auc else "  AutoDock-GPU AUC:   n/a")
    if gn_m is None:
        print("  Gnina CNN AUC:      NOT RUN (gnina unavailable on this host)")
        print("=" * 60)
        return
    print(f"  Gnina CNN AUC:      {gn_m['ROC_AUC']:.3f}")
    if ad_auc == ad_auc and ad_auc > 0:
        print(f"  Improvement:        {gn_m['ROC_AUC'] / ad_auc:.2f}x")
    print(f"  Gnina EF 1/5/10%:   {gn_m['EF_1pct']:.2f} / {gn_m['EF_5pct']:.2f} / {gn_m['EF_10pct']:.2f}")
    print(f"  Gnina BEDROC:       {gn_m['BEDROC']:.3f}")
    if merged is not None and not merged.empty:
        print(f"  Rank corr (rho):    {merged.attrs.get('spearman', float('nan')):.3f}")
        print(f"  Top-{merged.attrs.get('top_n', 10)} overlap:      "
              f"{merged.attrs.get('top_n_overlap', 0)}/{merged.attrs.get('top_n', 10)} compounds")
    print("=" * 60)


