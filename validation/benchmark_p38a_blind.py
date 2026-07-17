#!/usr/bin/env python3
"""
DEKOIS p38a benchmark - BLIND_DOCK mode.
Identical to MATCH_INBUILT run except the grid box spans the whole Ca backbone
(covering DFG + hinge), capped at AutoGrid's 126-pt limit. Reuses the already
prepped receptor.pdbqt and the 40 subset ligand pdbqts (no re-prep). Docks into
a separate output dir, computes raw + size-normalized ROC-AUC / EF / BEDROC and
compares against the MATCH_INBUILT result (0.40).
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import os
# Portable data root: set $DOCKSUITE_DATA to your working dir
_BASE = Path(os.environ.get("DOCKSUITE_DATA", "./validation_data")).resolve()

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score, roc_curve
RDLogger.DisableLog("rdApp.*")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
BENCH   = _BASE / "p38a_benchmark"
LIGDIR  = BENCH / "ligands_pdbqt"
OUTPUT  = _BASE / "p38a_output_blind"
RECEPTOR_PDB = BENCH / "1W82.pdb"
RECEPTOR = BENCH / "receptor.pdbqt"
SMI     = BENCH / "DEKOIS2" / "DEKOIS2" / "p38-alpha" / "active_decoys.smi"
N_ACT, N_DEC = 10, 30
BOX_BUFFER, SPACING, NPTS_CAP = 4.0, 0.375, 126
NRUN, HEUR, ASTOP = 100, 1, 1
SEED = 42
MATCH_AUC = 0.398          # MATCH_INBUILT reference (rounded 0.40)
AUTOGRID = r"C:\Program Files (x86)\The Scripps Research Institute\Autodock\4.2.6\autogrid4.exe"
GPU      = str(_BASE / "tools" / "AutoDock-GPU.exe")
_ROW = re.compile(r"^\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|", re.MULTILINE)


def log(m=""):
    print(m, flush=True)


def run(cmd, cwd=None, check=True):
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(map(str, cmd))}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    return p


def subset():
    a, d = [], []
    for ln in SMI.read_text().splitlines():
        p = ln.split()
        if len(p) < 2:
            continue
        if p[1].startswith("BDB") and len(a) < N_ACT:
            a.append(p[1])
        elif p[1].startswith("ZINC") and len(d) < N_DEC:
            d.append(p[1])
    return a, d, {p.split()[1]: p.split()[0] for p in SMI.read_text().splitlines() if len(p.split()) >= 2}


def atom_types(pdbqt: Path):
    seen = []
    for l in pdbqt.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) > 77:
            t = l[77:].strip().split()[0]
            if t and t not in seen:
                seen.append(t)
    return seen


def blind_box():
    lines = RECEPTOR_PDB.read_text(errors="ignore").splitlines()
    ca = [l for l in lines if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    xs = [float(l[30:38]) for l in ca]; ys = [float(l[38:46]) for l in ca]; zs = [float(l[46:54]) for l in ca]
    center = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)
    span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    L = span + 2*BOX_BUFFER
    npts = int(math.ceil(L / SPACING)); npts += npts % 2
    capped = min(npts, NPTS_CAP)
    log(f"  BLIND box: Ca centroid ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) | "
        f"Ca span {span:.1f} A -> want {npts} pts, CAPPED {capped} = {capped*SPACING:.1f} A")
    return center, capped


def build_maps(center, npts, lig_types):
    (BENCH / "blind").mkdir(exist_ok=True)
    rec_types = atom_types(RECEPTOR)
    maps = "\n".join(f"map receptor.{t}.map" for t in lig_types)
    (BENCH / "receptor_blind.gpf").write_text(
        f"npts {npts} {npts} {npts}\ngridfld receptor.maps.fld\nspacing {SPACING}\n"
        f"receptor_types {' '.join(rec_types)}\nligand_types {' '.join(lig_types)}\n"
        f"receptor receptor.pdbqt\ngridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        f"smooth 0.5\n{maps}\nelecmap receptor.e.map\ndsolvmap receptor.d.map\ndielectric -0.1465\n")
    log("  running autogrid4 on 126-pt box (may take a few minutes)...")
    run([AUTOGRID, "-p", "receptor_blind.gpf", "-l", "receptor_blind.glg"], cwd=BENCH)
    log("  maps built. copying to blind output dir...")
    OUTPUT.mkdir(exist_ok=True)
    for mp in list(BENCH.glob("receptor.*.map")) + [BENCH/"receptor.maps.fld", BENCH/"receptor.maps.xyz"]:
        shutil.copy(mp, OUTPUT / mp.name)


def dlg_done(dlg: Path):
    return dlg.is_file() and dlg.stat().st_size > 0 and "CLUSTERING HISTOGRAM" in dlg.read_text(errors="ignore")


def dock(cid):
    dlg = OUTPUT / f"{cid}.dlg"
    if dlg_done(dlg):
        return "skip"
    src = LIGDIR / f"{cid}.pdbqt"
    if not src.is_file():
        return "nolig"
    shutil.copy(src, OUTPUT / src.name)
    r = subprocess.run([GPU, "--ffile", "receptor.maps.fld", "--lfile", src.name, "--resnam", cid,
                        "--nrun", str(NRUN), "--heuristics", str(HEUR), "--autostop", str(ASTOP),
                        "--devnum", "1"], cwd=str(OUTPUT), capture_output=True, text=True)
    if r.returncode != 0:
        dlg.unlink(missing_ok=True)
        return "fail"
    return "docked"


def largest_cluster_dG(dlg: Path):
    rows = [(float(m[2]), int(m[5])) for m in _ROW.finditer(dlg.read_text(errors="ignore"))]
    if not rows:
        return None
    return max(rows, key=lambda x: (x[1], -x[0]))[0]


def ef(scores, y, frac):
    o = np.argsort(-np.asarray(scores)); yy = np.asarray(y)[o]
    nt = max(1, int(round(frac * len(yy))))
    return float((yy[:nt].sum() / nt) / (yy.sum() / len(yy)))


def bedroc(scores, y, alpha=20.0):
    o = np.argsort(-np.asarray(scores)); yy = np.asarray(y)[o]
    N = len(yy); n = int(yy.sum()); Ra = n / N
    ranks = np.where(yy == 1)[0] + 1
    s = np.sum(np.exp(-alpha * ranks / N))
    rie = s / (n / N * (1 - math.exp(-alpha)) / (math.exp(alpha / N) - 1))
    fac = Ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * Ra))
    return float(rie * fac + 1 / (1 - math.exp(alpha * (1 - Ra))))


def metrics(scores, y):
    return {"ROC_AUC": float(roc_auc_score(y, scores)), "EF_1pct": ef(scores, y, 0.01),
            "EF_5pct": ef(scores, y, 0.05), "EF_10pct": ef(scores, y, 0.10),
            "BEDROC": bedroc(scores, y, 20.0)}


def boot(scores, y, n=1000):
    rng = np.random.default_rng(SEED); s = np.asarray(scores); yy = np.asarray(y); idx = np.arange(len(yy)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(yy[b])) == 2:
            out.append(roc_auc_score(yy[b], s[b]))
    return float(np.std(out))


def main():
    act, dec, smi_map = subset()
    ids = act + dec
    lig_types = []
    for cid in ids:
        for t in atom_types(LIGDIR / f"{cid}.pdbqt"):
            if t not in lig_types:
                lig_types.append(t)
    log(f"  subset: {len(act)} actives + {len(dec)} decoys | ligand atom types: {' '.join(lig_types)}")

    center, npts = blind_box()
    if not (OUTPUT / "receptor.maps.fld").is_file():
        build_maps(center, npts, lig_types)
    else:
        log("  blind maps already present -> reuse")

    log("  docking 40 compounds (BLIND box)...")
    tally = {}
    for i, cid in enumerate(ids, 1):
        st = dock(cid); tally[st] = tally.get(st, 0) + 1
        log(f"    [{i:2d}/40] {cid:14s} {st}")
    log(f"  dock tally: {tally}")

    rows = []
    for cid in ids:
        dG = largest_cluster_dG(OUTPUT / f"{cid}.dlg")
        if dG is None:
            log(f"    WARN no score: {cid}"); continue
        mm = Chem.MolFromSmiles(smi_map[cid])
        rows.append({"Compound_ID": cid, "Type": "Active" if cid.startswith("BDB") else "Decoy",
                     "is_active": int(cid.startswith("BDB")), "dG_raw": dG,
                     "Heavy_Atoms": mm.GetNumHeavyAtoms()})
    df = pd.DataFrame(rows)
    y = df["is_active"].to_numpy()

    slope, intercept = np.polyfit(df["Heavy_Atoms"], df["dG_raw"], 1)
    df["dG_normalized"] = df["dG_raw"] - (slope * df["Heavy_Atoms"] + intercept)
    r_before = float(np.corrcoef(df["Heavy_Atoms"], df["dG_raw"])[0, 1])
    r_after = float(np.corrcoef(df["Heavy_Atoms"], df["dG_normalized"])[0, 1])

    m_raw = metrics((-df["dG_raw"]).to_numpy(), y)
    m_norm = metrics((-df["dG_normalized"]).to_numpy(), y)
    sd_raw = boot((-df["dG_raw"]).to_numpy(), y); sd_norm = boot((-df["dG_normalized"]).to_numpy(), y)

    df.sort_values("dG_raw").to_csv(BENCH / "blind_raw_ranking.csv", index=False)
    with pd.ExcelWriter(BENCH / "p38a_blind_results.xlsx", engine="openpyxl") as xw:
        df.sort_values("dG_raw").to_excel(xw, sheet_name="Raw_Ranking", index=False)
        df.sort_values("dG_normalized").to_excel(xw, sheet_name="Normalized_Ranking", index=False)
        pd.DataFrame([{"Method": "BLIND raw", **m_raw}, {"Method": "BLIND +SizeNorm", **m_norm}]).to_excel(
            xw, sheet_name="Metrics", index=False)

    # ---- ROC figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    for lab, sc, c in [(f"BLIND raw (AUC={m_raw['ROC_AUC']:.3f})", -df["dG_raw"], "#2ca02c"),
                       (f"BLIND +norm (AUC={m_norm['ROC_AUC']:.3f})", -df["dG_normalized"], "#d62728")]:
        fpr, tpr, _ = roc_curve(y, sc); ax.plot(fpr, tpr, lw=2, color=c, label=lab)
    ax.plot([0, 1], [0, 1], ":", color="k", alpha=.4, label="random (0.50)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"p38a BLIND_DOCK vs MATCH_INBUILT ({MATCH_AUC:.2f})"); ax.legend(loc="lower right")
    (_BASE / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(str(_BASE / "figures" / "figure8_blind_vs_match.png")); plt.close(fig)

    log("\n" + "=" * 76)
    log("  BLIND_DOCK RESULTS  (40-compound subset)")
    log("=" * 76)
    log(f"  {'Method':<20}{'ROC-AUC':>10}{'EF_1%':>8}{'EF_5%':>8}{'EF_10%':>9}{'BEDROC':>9}")
    log(f"  {'MATCH_INBUILT raw':<20}{MATCH_AUC:>10.3f}{0.0:>8.2f}{0.0:>8.2f}{0.0:>9.2f}{0.057:>9.3f}")
    log(f"  {'BLIND raw':<20}{m_raw['ROC_AUC']:>10.3f}{m_raw['EF_1pct']:>8.2f}{m_raw['EF_5pct']:>8.2f}"
        f"{m_raw['EF_10pct']:>9.2f}{m_raw['BEDROC']:>9.3f}")
    log(f"  {'BLIND +SizeNorm':<20}{m_norm['ROC_AUC']:>10.3f}{m_norm['EF_1pct']:>8.2f}{m_norm['EF_5pct']:>8.2f}"
        f"{m_norm['EF_10pct']:>9.2f}{m_norm['BEDROC']:>9.3f}")
    log("-" * 76)
    log(f"  ROC-AUC (raw)  : {MATCH_AUC:.3f} (MATCH)  ->  {m_raw['ROC_AUC']:.3f} (BLIND)   "
        f"delta {m_raw['ROC_AUC']-MATCH_AUC:+.3f}   (+/-{sd_raw:.3f})")
    log(f"  ROC-AUC (norm) : {m_norm['ROC_AUC']:.3f} (+/-{sd_norm:.3f})")
    log(f"  corr(dG,Heavy) : {r_before:.3f} -> {r_after:.3f} after norm")
    log(f"  Excel  -> {BENCH/'p38a_blind_results.xlsx'}")
    log(f"  Figure -> figures/figure8_blind_vs_match.png")
    log("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
