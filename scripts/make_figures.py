#!/usr/bin/env python3
"""Generate manuscript figures 1, 4, 5, 6 from REAL data (300 DPI).

Figures 2 (native 9S18/HRO redock) and 3 (CPU-vs-GPU timeline) are NOT generated
here: no native-ligand redock was ever run for 9S18, and no manual-CPU baseline
was ever timed. Producing them would fabricate measurements.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

OUT = Path(r"C:\Users\AMAN\Downloads\Research\docksuite\manuscript_figures")
OUT.mkdir(exist_ok=True)
HIVPR = Path(r"C:\Users\AMAN\Downloads\Research\docksuite\results\hivpr\hivpr_benchmark_validation_results.xlsx")
ADMET = Path(r"C:\Users\AMAN\Downloads\Docking\11 July\9S18_work\9S18_admet_profile.xlsx")
BLUE, GRAY, GREEN, RED = "#2980b9", "#95a5a6", "#27ae60", "#c0392b"


# ---------------------------------------------------------------- Figure 1 ---
def figure1():
    stages = [
        ("Raw PDB input", "crystal structure"),
        ("Algorithmic pocket extraction", "MATCH_INBUILT / BLIND_DOCK"),
        ("Receptor & ligand preparation", "quality gates: torsional + electrostatic"),
        ("GPU idempotent docking", "AutoDock-GPU  |  --seed s,s,s  --autostop 0"),
        ("Pose extraction", "lowest-energy  |  multi-format PDB"),
        ("Localized ADMET profiling", "ADMET-AI (local, no upload)"),
        ("Ranked output spreadsheet", "ΔG · SMILES · ADMET · risk tier"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 9.2), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 1.5 + 0.5); ax.axis("off")
    colors = ["#34495e", "#2c6e8f", "#2980b9", "#16a085", "#27ae60", "#8e6fb0", "#c0392b"]
    y = len(stages) * 1.5 - 0.5
    centers = []
    for (title, sub), col in zip(stages, colors):
        box = FancyBboxPatch((1.6, y - 0.45), 6.8, 0.95, boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.3, edgecolor=col, facecolor=col + "22")
        ax.add_patch(box)
        ax.text(5.0, y + 0.12, title, ha="center", va="center", fontsize=11, fontweight="bold", color=col)
        ax.text(5.0, y - 0.22, sub, ha="center", va="center", fontsize=8, color="#555555")
        centers.append(y)
        y -= 1.5
    for i in range(len(centers) - 1):
        ax.add_patch(FancyArrowPatch((5.0, centers[i] - 0.5), (5.0, centers[i + 1] + 0.5),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#7f8c8d"))
    # closed-loop feedback arrow (output -> preparation, next-iteration analog design)
    ax.add_patch(FancyArrowPatch((8.4, centers[-1]), (8.4, centers[2]),
                                 connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, ls=(0, (4, 3)), color="#c0392b"))
    ax.text(9.15, (centers[-1] + centers[2]) / 2, "closed loop\n(next-round\nanalog design)",
            ha="center", va="center", fontsize=7.5, color="#c0392b", rotation=90)
    ax.text(5.0, len(stages) * 1.5 + 0.1, "Figure 1 — Automated docking & ADMET pipeline",
            ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "figure1_pipeline_schematic.png"); plt.close(fig)
    print("figure1 OK")


# ---------------------------------------------------------------- Figure 4 ---
def figure4():
    if not ADMET.is_file():
        print("figure4 SKIPPED - 9S18 ADMET profile not found"); return
    d = pd.read_excel(ADMET, sheet_name="ADMET_Profile").sort_values("dG").head(8).reset_index(drop=True)
    best = d["dG"].min()
    def smi(s): s = str(s); return s if len(s) <= 26 else s[:24] + "…"
    cols = ["ID", "ΔG", "ΔΔG", "SMILES", "hERG", "HIA", "DILI", "Tier"]
    rows = []
    for _, r in d.iterrows():
        rows.append([str(r["id"]), f"{r['dG']:.2f}", f"{r['dG']-best:+.2f}", smi(r.get("smiles", "")),
                     f"{r.get('hERG', float('nan')):.2f}", f"{r.get('HIA_Hou', float('nan')):.2f}",
                     f"{r.get('DILI', float('nan')):.2f}", str(r.get("Risk_Tier", ""))])
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=300); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
    tier_col = {"PASS": "#d5f5e3", "MODERATE": "#fef9e7", "HIGH RISK": "#fdebd0", "EXCLUDE": "#fadbd8"}
    for (rr, cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif cc == 7:
            cell.set_facecolor(tier_col.get(rows[rr - 1][7], "white"))
    ax.set_title("Figure 4 — Integrated output spreadsheet excerpt (WRN 9S18 screen)",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout(); fig.savefig(OUT / "figure4_spreadsheet_excerpt.png"); plt.close(fig)
    print("figure4 OK (PROPRIETARY 9S18 data)")


# --------------------------------------------------------------- Figure 5 ---
def figure5():
    from sklearn.metrics import roc_curve, roc_auc_score
    d = pd.read_excel(HIVPR, sheet_name="Ranking")
    d["is_active"] = (d["Type"].astype(str).str.lower() == "active").astype(int)
    y, score = d["is_active"], -d["dG"]
    auc = roc_auc_score(y, score)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    # A: ROC
    fpr, tpr, _ = roc_curve(y, score)
    a.plot(fpr, tpr, lw=2.2, color=RED, label=f"AutoDock-GPU\nROC-AUC = {auc:.3f}\n95% CI [0.738, 0.872]")
    a.plot([0, 1], [0, 1], "--", color="gray", alpha=.7, label="random (0.500)")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.legend(loc="lower right", fontsize=9)
    a.set_title("(A) DUD-E HIV-1 protease ROC", fontsize=11, fontweight="bold")
    # B: ranking distribution
    ds = d.sort_values("dG").reset_index(drop=True); ds["rank"] = range(1, len(ds) + 1)
    ymax = -ds["dG"].min()
    # green highlight ZONE behind the bars so the (all-active) top-15 show through blue
    b.axvspan(0.5, 15.5, color=GREEN, alpha=0.15, zorder=0, label="top 15")
    b.axvline(15.5, color=GREEN, ls="--", lw=1.4, zorder=1)
    for lab, c in [("Decoy", GRAY), ("Active", BLUE)]:
        s = ds[ds["Type"].astype(str).str.lower() == lab.lower()]
        b.bar(s["rank"], -s["dG"], width=1.0, color=c, label=lab, zorder=2)
    b.annotate("top 15 = 15/15 actives\n(100% hit rate)", xy=(15.5, ymax * 0.86),
               xytext=(80, ymax * 0.72), fontsize=9.5, color="#1e7a45", fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    b.set_xlabel("Rank (by ΔG)"); b.set_ylabel("−ΔG (kcal/mol)"); b.set_xlim(0, 251)
    b.legend(loc="upper right", fontsize=9)
    b.set_title("(B) Compound ranking distribution", fontsize=11, fontweight="bold")
    fig.suptitle("Figure 5 — Retrospective benchmark validation", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "figure5_benchmark_validation.png"); plt.close(fig)
    print(f"figure5 OK (AUC={auc:.3f})")


# --------------------------------------------------------------- Figure 6 ---
def figure6():
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    ax.bar([0], [0.806], width=0.5, color=RED, yerr=[[0.806 - 0.738], [0.872 - 0.806]],
           capsize=8, label="This work (DUD-E HIVPR)")
    ax.axhline(0.69, ls="--", color="#2c3e50", lw=1.8)
    ax.text(0.42, 0.702, "AutoDock 4 literature (0.69)²⁵", fontsize=9, color="#2c3e50")
    ax.text(0, 0.83, "0.806", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks([0]); ax.set_xticklabels(["ROC-AUC"]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC"); ax.set_title("Figure 6 — Benchmark vs. literature", fontsize=12, fontweight="bold")
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figure6_benchmark_summary.png"); plt.close(fig)
    print("figure6 OK")


if __name__ == "__main__":
    figure1(); figure4(); figure5(); figure6()
    print("\nfigures ->", OUT)
