#!/usr/bin/env python3
"""
Retrospective benchmark validation of the docking framework against the
DEKOIS 2.0 p38a (1W82 / L10) set: SMILES -> 3D -> pdbqt -> AutoDock-GPU ->
ROC-AUC / EF / BEDROC, with native redocking as positive control.

NOTE ON TOOLING: MGLTools (prepare_receptor4/ligand4.py) is not installed, so
receptor/ligand prep use OpenBabel with equivalent operations - the SAME path
the production pipeline uses. Native ligand residue is 'L10' (config '095' is
absent in 1W82). These are the only deviations; all logic is per spec.
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
BENCH   = _BASE / "p38a_benchmark"
OUTPUT  = _BASE / "p38a_output"
SMI     = BENCH / "DEKOIS2" / "DEKOIS2" / "p38-alpha" / "active_decoys.smi"
RECEPTOR_PDB = BENCH / "1W82.pdb"
NATIVE_RES   = "L10"                    # corrected from '095'
N_ACTIVES, N_DECOYS = 10**9, 10**9      # full DEKOIS set (all 40 actives + all decoys)
BOX_BUFFER, SPACING = 4.0, 0.375
NRUN, HEUR, ASTOP = 100, 1, 1
CLUSTER_RMSD = 2.0
SEED = 42
BUFFERS = {"HOH", "WAT", "SO4", "GOL", "EDO", "PEG", "MSE", "ACT", "CL", "NA",
           "MG", "CA", "K", "ZN", "MPD", "DMS", "IOD", "PO4", "BME"}

KNOWN_DIRS = [
    Path(r"C:\Program Files (x86)\The Scripps Research Institute\Autodock\4.2.6"),
    _BASE / "tools",
    Path(sys.executable).parent / "Scripts", Path(sys.executable).parent,
]
TOOLS: dict[str, str] = {}
_ROW = re.compile(r"^\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|", re.MULTILINE)


def log(m=""):
    print(m, flush=True)


def which(name):
    if (p := shutil.which(name)):
        return p
    for d in KNOWN_DIRS:
        for e in (d / name, d / f"{name}.exe"):
            if e.is_file():
                return str(e)
    return None


def run(cmd, cwd=None, check=True):
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    return p


def atom_types(pdbqt: Path):
    seen = []
    for l in pdbqt.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) > 77:
            t = l[77:].strip().split()[0]
            if t and t not in seen:
                seen.append(t)
    return seen


def heavy_coords(path: Path):
    pts = []
    for l in path.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 54:
            t = l[76:].strip().split()[0] if l[76:].strip() else ""
            if t in ("H", "HD", "HS"):
                continue
            pts.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return pts


# --------------------------------------------------------------------------- #
# Step 2: extract 10 actives + 30 decoys
# --------------------------------------------------------------------------- #
def extract_subsets():
    actives, decoys = [], []
    for line in SMI.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        smi, cid = parts[0], parts[1]
        if cid.startswith("BDB") and len(actives) < N_ACTIVES:
            actives.append((smi, cid))
        elif cid.startswith("ZINC") and len(decoys) < N_DECOYS:
            decoys.append((smi, cid))
    (BENCH / "actives_full.smi").write_text("\n".join(f"{s} {c}" for s, c in actives) + "\n")
    (BENCH / "decoys_full.smi").write_text("\n".join(f"{s} {c}" for s, c in decoys) + "\n")
    log(f"  actives: {len(actives)} (BDB) | decoys: {len(decoys)} (ZINC)")
    return actives, decoys


# --------------------------------------------------------------------------- #
# Step 3: SMILES -> 3D -> pdbqt  (+ charge proportional scaling)
# --------------------------------------------------------------------------- #
def scale_charges(pdbqt: Path):
    lines = pdbqt.read_text(errors="ignore").splitlines()
    qs = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 76:
            try:
                qs.append(abs(float(l[66:76])))
            except ValueError:
                pass
    mx = max(qs) if qs else 0
    if mx <= 1.0:
        return 0
    f = 1.0 / mx
    out, n = [], 0
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 76:
            try:
                q = float(l[66:76]) * f
                l = l[:66] + f"{q:+.3f}".rjust(10) + l[76:]; n += 1
            except ValueError:
                pass
        out.append(l)
    pdbqt.write_text("\n".join(out) + "\n")
    return n


def smiles_to_pdbqt(smi, name, outdir: Path):
    pdbqt = outdir / f"{name}.pdbqt"
    if pdbqt.is_file() and pdbqt.stat().st_size > 0:      # resumable: reuse existing prep
        td = next((int(l.split()[1]) for l in pdbqt.read_text().splitlines() if l.startswith("TORSDOF")), 0)
        return pdbqt, td
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=SEED) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    sdf = outdir / f"{name}.sdf"
    Chem.SDWriter(str(sdf)).write(m)
    mol2 = outdir / f"{name}.mol2"
    run([TOOLS["obabel"], sdf, "-O", mol2], check=False)
    pdbqt = outdir / f"{name}.pdbqt"
    # no -xh -> non-polar H merged (united atom); active torsion tree; Gasteiger
    run([TOOLS["obabel"], mol2, "-O", pdbqt, "--partialcharge", "gasteiger"], check=False)
    if not pdbqt.is_file():
        return None
    scale_charges(pdbqt)
    td = next((int(l.split()[1]) for l in pdbqt.read_text().splitlines() if l.startswith("TORSDOF")), 0)
    sdf.unlink(missing_ok=True)
    return pdbqt, td


# --------------------------------------------------------------------------- #
# Step 4/5: box + receptor
# --------------------------------------------------------------------------- #
def box_and_receptor():
    lines = RECEPTOR_PDB.read_text(errors="ignore").splitlines()
    lig = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_RES and l[76:78].strip() != "H"]
    xs = [float(l[30:38]) for l in lig]; ys = [float(l[38:46]) for l in lig]; zs = [float(l[46:54]) for l in lig]
    center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    L = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) + 2*BOX_BUFFER
    npts = int(math.ceil(L / SPACING)); npts += npts % 2
    log(f"  Pocket center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
        f"Box size: {L:.1f} A ({npts} pts) | Grid spacing: {SPACING} A")

    nat = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_RES]
    (BENCH / "native_reference_input.pdb").write_text("\n".join(nat + ["END"]) + "\n")
    prot = [l[:16] + " " + l[17:] for l in lines if l.startswith("ATOM") and l[16:17] in (" ", "A")]
    (BENCH / "protein_clean.pdb").write_text("\n".join(prot + ["TER", "END"]) + "\n")

    receptor = BENCH / "receptor.pdbqt"
    run([TOOLS["obabel"], BENCH / "protein_clean.pdb", "-O", receptor, "-xr", "-p", "7.4", "--partialcharge", "gasteiger"])
    return center, npts, receptor


def build_maps(receptor, center, npts, all_lig_types):
    # receptor.pdbqt already lives in BENCH (where autogrid runs) - no copy needed
    rec_types = atom_types(receptor)
    maps = "\n".join(f"map receptor.{t}.map" for t in all_lig_types)
    (BENCH / "receptor.gpf").write_text(
        f"npts {npts} {npts} {npts}\ngridfld receptor.maps.fld\nspacing {SPACING}\n"
        f"receptor_types {' '.join(rec_types)}\nligand_types {' '.join(all_lig_types)}\n"
        f"receptor receptor.pdbqt\ngridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        f"smooth 0.5\n{maps}\nelecmap receptor.e.map\ndsolvmap receptor.d.map\ndielectric -0.1465\n")
    run([TOOLS["autogrid4"], "-p", "receptor.gpf", "-l", "receptor.glg"], cwd=BENCH)
    return BENCH / "receptor.maps.fld"


# --------------------------------------------------------------------------- #
# docking + parsing
# --------------------------------------------------------------------------- #
def dlg_done(dlg: Path):
    return dlg.is_file() and dlg.stat().st_size > 0 and "CLUSTERING HISTOGRAM" in dlg.read_text(errors="ignore")


def dock(pdbqt: Path, resnam, cwd: Path):
    dlg = cwd / f"{resnam}.dlg"
    if dlg_done(dlg):
        return "skip"
    shutil.copy(pdbqt, cwd / pdbqt.name)
    r = subprocess.run([TOOLS["AutoDock-GPU"], "--ffile", "receptor.maps.fld", "--lfile", pdbqt.name,
                        "--resnam", resnam, "--nrun", str(NRUN), "--heuristics", str(HEUR),
                        "--autostop", str(ASTOP), "--devnum", "1"], cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        (cwd / f"{resnam}.dlg").unlink(missing_ok=True)
        return "fail"
    return "docked"


def largest_cluster(dlg: Path):
    rows = [(int(m[1]), float(m[2]), int(m[3]), int(m[5])) for m in _ROW.finditer(dlg.read_text(errors="ignore"))]
    if not rows:
        return None
    r = max(rows, key=lambda x: (x[3], -x[1]))
    return {"energy": r[1], "run": r[2], "size": r[3]}


def atoms_for_run(dlg: Path, run_no):
    cur, cr, tbl = [], None, {}
    for l in dlg.read_text(errors="ignore").splitlines():
        if not l.startswith("DOCKED:"):
            continue
        b = l[8:]
        if b.strip().startswith("MODEL"):
            cur, cr = [], None
        elif "Run =" in b:
            if (mm := re.search(r"Run\s*=\s*(\d+)", b)):
                cr = int(mm.group(1))
        elif b[:6].strip() in ("ATOM", "HETATM"):
            cur.append(b.rstrip())
        elif b.strip() == "ENDMDL" and cr is not None:
            tbl[cr] = cur
    return tbl.get(run_no)


def write_pose(atoms, name, e, sz, out: Path):
    if not atoms:
        return
    tmp = out.with_suffix(".t.pdbqt"); tmp.write_text("ROOT\n" + "\n".join(atoms) + "\nENDROOT\nTORSDOF 0\n")
    raw = out.with_suffix(".r.pdb"); run([TOOLS["obabel"], tmp, "-O", raw], check=False)
    ec, body, s = {}, [], 0
    for l in raw.read_text(errors="ignore").splitlines():
        if not l.startswith(("ATOM", "HETATM")):
            continue
        s += 1; x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        el = (l[76:78].strip() or l[12:14].strip()).capitalize()
        ec[el] = ec.get(el, 0) + 1; an = f"{el}{ec[el]}"; nf = f"{an:<4}" if len(el) == 2 else f" {an:<3}"
        body.append(f"HETATM{s:>5} {nf} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
    con = [l for l in raw.read_text(errors="ignore").splitlines() if l.startswith("CONECT")]
    out.write_text("\n".join([f"REMARK  {name} dG {e} cluster {sz}/{NRUN}"] + body + con + ["END"]) + "\n")
    tmp.unlink(missing_ok=True); raw.unlink(missing_ok=True)


def rmsd(ref: Path, pose: Path):
    a, b = heavy_coords(ref), heavy_coords(pose)
    if not a or not b:
        return None
    n = min(len(a), len(b))
    order = math.sqrt(sum((a[i][0]-b[i][0])**2+(a[i][1]-b[i][1])**2+(a[i][2]-b[i][2])**2 for i in range(n))/n)
    nn = math.sqrt(sum(min((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2 for q in b) for p in a)/len(a))
    return round(min(order, nn), 2)


# --------------------------------------------------------------------------- #
# Step 10: metrics
# --------------------------------------------------------------------------- #
def enrichment(scores, labels, frac):
    order = np.argsort(-np.asarray(scores)); y = np.asarray(labels)[order]
    ntop = max(1, int(round(frac * len(y))))
    return float((y[:ntop].sum() / ntop) / (y.sum() / len(y)))


def bedroc(scores, labels, alpha=20.0):
    order = np.argsort(-np.asarray(scores)); y = np.asarray(labels)[order]
    N = len(y); n = int(y.sum()); Ra = n / N
    ranks = np.where(y == 1)[0] + 1
    s = np.sum(np.exp(-alpha * ranks / N))
    rie = s / (n / N * (1 - math.exp(-alpha)) / (math.exp(alpha / N) - 1))
    fac = Ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * Ra))
    return float(rie * fac + 1 / (1 - math.exp(alpha * (1 - Ra))))


def metrics(df):
    from sklearn.metrics import roc_auc_score
    sc = (-df["dG"]).to_numpy(); y = df["is_active"].to_numpy()
    res = {"ROC_AUC": float(roc_auc_score(y, sc)),
           "EF_1pct": enrichment(sc, y, 0.01), "EF_5pct": enrichment(sc, y, 0.05),
           "EF_10pct": enrichment(sc, y, 0.10), "BEDROC_a20": bedroc(sc, y, 20.0)}
    # bootstrap CI
    rng = np.random.default_rng(SEED); aucs, ef10 = [], []
    idx = np.arange(len(y))
    for _ in range(1000):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], sc[b])); ef10.append(enrichment(sc[b], y[b], 0.10))
    res["ROC_AUC_std"] = float(np.std(aucs)); res["EF_10pct_std"] = float(np.std(ef10))
    return res


# --------------------------------------------------------------------------- #
def main():
    log("=" * 70); log("  DEKOIS 2.0 p38a RETROSPECTIVE BENCHMARK  (1W82 / L10)"); log("=" * 70)
    for t in ("obabel", "autogrid4", "AutoDock-GPU"):
        TOOLS[t] = which(t)
        if not TOOLS[t]:
            log(f"[FATAL] missing critical dependency: {t}"); return 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ligdir = BENCH / "ligands_pdbqt"; ligdir.mkdir(exist_ok=True)
    (BENCH / "config_snapshot.txt").write_text(
        f"NATIVE_RES={NATIVE_RES}\nNRUN={NRUN}\nBUFFER={BOX_BUFFER}\nSPACING={SPACING}\n"
        f"CLUSTER_RMSD={CLUSTER_RMSD}\nSEED={SEED}\nprep=OpenBabel(MGLTools absent)\n")

    log("\n[2] Extract 10 actives + 30 decoys"); actives, decoys = extract_subsets()
    queue = [(s, c, 1) for s, c in actives] + [(s, c, 0) for s, c in decoys]

    log("\n[3] SMILES -> 3D -> pdbqt (RDKit seed 42 + MMFF; charge scaling)")
    prepared, rigid, capped = [], [], 0
    for smi, cid, act in queue:
        r = smiles_to_pdbqt(smi, cid, ligdir)
        if r is None:
            log(f"    [WARN] prep failed: {cid}"); continue
        pdbqt, td = r
        if td == 0:
            rigid.append(cid)
        prepared.append((pdbqt, cid, act))
    log(f"    prepared {len(prepared)}/{len(queue)} | rigid(TORSDOF=0): {rigid or 'none'}")

    log("\n[4/5] Box + receptor prep")
    center, npts, receptor = box_and_receptor()

    log("\n[6] AutoGrid maps")
    lig_types = []
    for p, _, _ in prepared:
        for t in atom_types(p):
            if t not in lig_types:
                lig_types.append(t)
    fld = build_maps(receptor, center, npts, lig_types)
    log(f"    receptor.maps.fld ready | ligand types: {' '.join(lig_types)}")

    log("\n[7] Native redocking control")
    nat_pdbqt = ligdir / "native_reference.pdbqt"
    run([TOOLS["obabel"], BENCH / "native_reference_input.pdb", "-O", nat_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"])
    dock(nat_pdbqt, "native_control", BENCH)
    bc = largest_cluster(BENCH / "native_control.dlg")
    baseline, rd = None, None
    if bc:
        write_pose(atoms_for_run(BENCH / "native_control.dlg", bc["run"]), "native_control",
                   bc["energy"], bc["size"], BENCH / "native_control_top1.pdb")
        baseline = bc["energy"]
        rd = rmsd(nat_pdbqt, BENCH / "native_control_top1.pdb")
        verdict = "PASS: Redocking validation successful" if (rd is not None and rd <= 2.0) else "WARNING: redock RMSD > 2 A"
        log(f"    {verdict} (RMSD {rd} A)")
        log(f"    SYSTEM CONTROL BASELINE SCORE: {baseline:.2f} kcal/mol (cluster {bc['size']}/{NRUN})")

    log(f"\n[8] Idempotent batch docking ({len(prepared)} ligands)")
    # maps live in BENCH; batch docking runs in OUTPUT -> copy them so --ffile resolves
    for mp in list(BENCH.glob("receptor.*.map")) + [BENCH / "receptor.maps.fld", BENCH / "receptor.maps.xyz"]:
        if mp.is_file():
            shutil.copy(mp, OUTPUT / mp.name)
    for i, (p, cid, act) in enumerate(prepared, 1):
        st = dock(p, cid, OUTPUT)
        if st == "fail":
            log(f"    [ERR] {cid} crashed - will retry next run")

    log("\n[9] Pose extraction + score compilation")
    recs = []
    for p, cid, act in prepared:
        dlg = OUTPUT / f"{cid}.dlg"
        bc = largest_cluster(dlg) if dlg_done(dlg) else None
        if not bc:
            continue
        write_pose(atoms_for_run(dlg, bc["run"]), cid, bc["energy"], bc["size"], OUTPUT / f"{cid}_top1.pdb")
        recs.append({"Compound_ID": cid, "Type": "Active" if act else "Decoy", "is_active": act,
                     "dG": bc["energy"], "Cluster_Size": bc["size"],
                     "ddG_vs_Control": round(bc["energy"] - baseline, 2) if baseline else None})
    df = pd.DataFrame(recs).sort_values("dG").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    log("\n[10] Benchmark metrics")
    mets = metrics(df)
    for k, v in mets.items():
        log(f"    {k}: {v:.3f}")

    log("\n[11] Export")
    xlsx = BENCH / "p38a_benchmark_validation_results.xlsx"
    df.to_excel(xlsx, index=False, engine="openpyxl")

    log("\n" + "=" * 70); log("  BENCHMARK SUMMARY"); log("=" * 70)
    log("| Rank | Compound | Type | dG | Cluster |")
    log("|---|---|---|---|---|")
    for _, r in df.head(10).iterrows():
        star = " *" if r["is_active"] else ""
        log(f"| {r['Rank']} | {r['Compound_ID']}{star} | {r['Type']} | {r['dG']:.2f} | {r['Cluster_Size']} |")
    log("")
    log(f"  ROC-AUC        : {mets['ROC_AUC']:.3f} +/- {mets['ROC_AUC_std']:.3f}")
    log(f"  EF 1% / 5% / 10%: {mets['EF_1pct']:.2f} / {mets['EF_5pct']:.2f} / {mets['EF_10pct']:.2f}")
    log(f"  BEDROC (a=20)  : {mets['BEDROC_a20']:.3f}")
    log(f"  Redock RMSD    : {rd} A  |  Control baseline: {baseline} kcal/mol")
    log(f"  Actives in top 10: {int(df.head(10)['is_active'].sum())}/10")
    log(f"  Excel: {xlsx}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
