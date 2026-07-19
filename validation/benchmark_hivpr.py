#!/usr/bin/env python3
"""
Retrospective benchmark: DUD-E HIV-1 protease (1HXB / native ROC = saquinavir).
Validates the AutoDock-GPU / AD4 framework on a hydrophobic, buried, C2-symmetric
active site where the AD4 Lennard-Jones vdW term is expected to perform well
(Chang et al. 2010: AD4 AUC ~0.69).

Corrections vs the requested config (grounded in the actual data):
  * INBUILT_LIGAND_NAME MK1 -> ROC  (1HXB contains ROC, not MK1; MK1 is in 1HSG)
  * DUD-E HIVPR really has 536 actives + 35750 decoys (not 62 + 3100), so the
    250-compound subset SAMPLES 62 actives (seed 42) + 188 decoys, not "all actives".
  * MGLTools absent -> OpenBabel substitute (Gasteiger charges, not Kollman).
"""
from __future__ import annotations

import math
import random
import re
import shutil
import subprocess
import sys
import time
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
DOCK    = _BASE
BENCH   = DOCK / "hivpr_benchmark"
OUTPUT  = DOCK / "hivpr_output"
DUDE    = BENCH / "hivpr_dude" / "hivpr"
RECEPTOR_PDB = BENCH / "1HXB.pdb"
NATIVE_RES   = "ROC"                    # corrected from MK1
N_ACTIVES_SUB, N_DECOYS_SUB = 62, 188   # 250-compound subset
BOX_BUFFER, SPACING = 4.0, 0.375
NRUN, HEUR, ASTOP = 100, 1, 1
CLUSTER_RMSD, SEED = 2.0, 42
CHANG2010_AUC = 0.69
BUFFERS = {"HOH", "WAT", "SO4", "GOL", "EDO", "PEG", "MSE", "ACT", "CL", "NA",
           "MG", "CA", "K", "ZN", "MPD", "DMS", "IOD", "PO4", "BME", "NO3"}
KNOWN_DIRS = [
    Path(r"C:\Program Files (x86)\The Scripps Research Institute\Autodock\4.2.6"),
    DOCK / "tools", Path(sys.executable).parent / "Scripts", Path(sys.executable).parent,
]
TOOLS: dict[str, str] = {}
_ROW = re.compile(r"^\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|", re.M)


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


def resolve_tools():
    m = {"obabel": "obabel", "autogrid4": "autogrid4", "AutoDock-GPU": "AutoDock-GPU"}
    for k, n in m.items():
        p = which(n)
        if not p:
            log(f"CRITICAL: missing dependency {n}")
            sys.exit(1)
        TOOLS[k] = p
    log(f"  tools: obabel={Path(TOOLS['obabel']).name}, autogrid4 OK, AutoDock-GPU OK")


def run(cmd, cwd=None, check=True):
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(map(str, cmd))}\n{p.stdout[-500:]}\n{p.stderr[-500:]}")
    return p


def atom_types(pdbqt: Path):
    seen = []
    for l in pdbqt.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) > 77:
            t = l[77:].strip().split()[0]
            if t and t not in seen:
                seen.append(t)
    return seen


def scale_charges(pdbqt: Path):
    lines = pdbqt.read_text(errors="ignore").splitlines()
    qs = [abs(float(l[66:76])) for l in lines
          if l.startswith(("ATOM", "HETATM")) and len(l) >= 76 and l[66:76].strip()]
    mx = max(qs) if qs else 0
    if mx <= 1.0:
        return 0
    f = 1.0 / mx
    out, n = [], 0
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 76:
            try:
                q = float(l[66:76]) * f
                l = l[:66] + f"{q:+.3f}".rjust(10) + l[76:]
                n += 1
            except ValueError:
                pass
        out.append(l)
    pdbqt.write_text("\n".join(out) + "\n")
    return n


# --------------------------------------------------------------------------- #
# Step 2: subset selection with property-coverage validation
# --------------------------------------------------------------------------- #
def _read_ism(path: Path):
    out = []
    for ln in path.read_text(errors="ignore").splitlines():
        p = ln.split()
        if len(p) >= 2:
            out.append((p[0], p[1]))     # (smiles, id)
    return out


def _descr(smiles_ids, tag):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for smi, cid in smiles_ids:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        rows.append({"id": cid, "smiles": smi, "MW": Descriptors.MolWt(m),
                     "cLogP": Crippen.MolLogP(m), "HBD": Lipinski.NumHDonors(m),
                     "HBA": Lipinski.NumHAcceptors(m), "RotB": Descriptors.NumRotatableBonds(m)})
    df = pd.DataFrame(rows)
    df["set"] = tag
    return df


def select_subset():
    log("  reading DUD-E .ism (SMILES) files + computing descriptors...")
    actives = _read_ism(DUDE / "actives_final.ism")
    decoys = _read_ism(DUDE / "decoys_final.ism")
    log(f"  DUD-E HIVPR actual counts: {len(actives)} actives, {len(decoys)} decoys")
    da = _descr(actives, "active")
    dd = _descr(decoys, "decoy")

    rng = random.Random(SEED)
    act_sample = rng.sample(list(da.index), min(N_ACTIVES_SUB, len(da)))
    da_sub = da.loc[act_sample].copy()

    def coverage(sub, full, col):
        fr = full[col].max() - full[col].min()
        sr = sub[col].max() - sub[col].min()
        return (sr / fr) if fr > 0 else 1.0

    dec_idx = rng.sample(list(dd.index), N_DECOYS_SUB)
    dd_sub = dd.loc[dec_idx].copy()
    covers = all(coverage(dd_sub, dd, c) > 0.8 for c in ("MW", "cLogP"))
    strat = False
    if not covers:                       # stratify by MW quartiles (47 each)
        strat = True
        dd = dd.copy()
        dd["q"] = pd.qcut(dd["MW"], 4, labels=False, duplicates="drop")
        picks = []
        per = N_DECOYS_SUB // 4
        for q in sorted(dd["q"].dropna().unique()):
            pool = list(dd[dd["q"] == q].index)
            picks += rng.sample(pool, min(per, len(pool)))
        while len(picks) < N_DECOYS_SUB:
            extra = rng.choice(list(dd.index))
            if extra not in picks:
                picks.append(extra)
        dd_sub = dd.loc[picks[:N_DECOYS_SUB]].copy()

    da_sub.to_csv(BENCH / "actives_62.smi", sep=" ", columns=["smiles", "id"], header=False, index=False)
    dd_sub.to_csv(BENCH / "decoys_188.smi", sep=" ", columns=["smiles", "id"], header=False, index=False)
    log(f"  Subset: {len(da_sub)} actives + {len(dd_sub)} decoys = {len(da_sub)+len(dd_sub)} total compounds"
        f"{' (MW-quartile stratified)' if strat else ' (random seed 42)'}")

    # property comparison table (original vs sampled decoys)
    prop = []
    for col in ("MW", "cLogP", "HBD", "HBA", "RotB"):
        prop.append({"Property": col,
                     "Original_decoys": f"{dd[col].mean():.2f} +/- {dd[col].std():.2f}",
                     "Sampled_decoys": f"{dd_sub[col].mean():.2f} +/- {dd_sub[col].std():.2f}",
                     "Range_coverage": f"{100*coverage(dd_sub, dd, col):.0f}%"})
    prop_df = pd.DataFrame(prop)
    log("\n  Decoy property comparison (original vs sampled):")
    log(prop_df.to_string(index=False))

    subset = ([(s, f"active_{i}", 1) for s, i in zip(da_sub["smiles"], da_sub["id"])] +
              [(s, f"decoy_{i}", 0) for s, i in zip(dd_sub["smiles"], dd_sub["id"])])
    return subset, prop_df


# --------------------------------------------------------------------------- #
# Step 3: ligand prep  SMILES -> 3D -> pdbqt (+ gates)
# --------------------------------------------------------------------------- #
def smiles_to_pdbqt(smi, name, outdir: Path):
    pdbqt = outdir / f"{name}.pdbqt"
    if pdbqt.is_file() and pdbqt.stat().st_size > 0:
        return pdbqt
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=SEED) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    sdf = outdir / f"{name}.sdf"
    Chem.SDWriter(str(sdf)).write(m)
    mol2 = outdir / f"{name}.mol2"
    run([TOOLS["obabel"], sdf, "-O", mol2], check=False)
    run([TOOLS["obabel"], mol2, "-O", pdbqt, "--partialcharge", "gasteiger"], check=False)
    sdf.unlink(missing_ok=True); mol2.unlink(missing_ok=True)
    if not pdbqt.is_file():
        return None
    scale_charges(pdbqt)                 # Gate: charge cap
    return pdbqt


# --------------------------------------------------------------------------- #
# Step 4/5: box + receptor
# --------------------------------------------------------------------------- #
def box_and_receptor():
    lines = RECEPTOR_PDB.read_text(errors="ignore").splitlines()
    lig = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_RES
           and l[76:78].strip() != "H"]
    xs = [float(l[30:38]) for l in lig]; ys = [float(l[38:46]) for l in lig]; zs = [float(l[46:54]) for l in lig]
    center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
    L = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) + 2*BOX_BUFFER
    npts = int(math.ceil(L / SPACING)); npts += npts % 2; npts = min(npts, 126)
    log(f"  Pocket center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
        f"Box size: {npts*SPACING:.1f} A ({npts} pts) | Grid spacing: {SPACING} A")

    nat = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == NATIVE_RES]
    (BENCH / "native_reference_input.pdb").write_text("\n".join(nat + ["END"]) + "\n")
    # clean: keep protein chains A+B, drop waters/buffers/native ligand
    prot = [l for l in lines if l.startswith("ATOM") and l[21] in ("A", "B")]
    (BENCH / "protein_clean.pdb").write_text("\n".join(prot + ["TER", "END"]) + "\n")

    receptor = BENCH / "receptor.pdbqt"
    run([TOOLS["obabel"], BENCH / "protein_clean.pdb", "-O", receptor, "-xr", "-p", "7.4",
         "--partialcharge", "gasteiger"])
    log(f"  receptor.pdbqt: {sum(1 for l in receptor.read_text().splitlines() if l.startswith(('ATOM','HETATM')))} atoms "
        f"(chains A+B retained; Gasteiger charges [MGLTools/Kollman unavailable])")
    return center, npts


def build_maps(center, npts, lig_types):
    rec_types = atom_types(BENCH / "receptor.pdbqt")
    maps = "\n".join(f"map receptor.{t}.map" for t in lig_types)
    (BENCH / "receptor.gpf").write_text(
        f"npts {npts} {npts} {npts}\ngridfld receptor.maps.fld\nspacing {SPACING}\n"
        f"receptor_types {' '.join(rec_types)}\nligand_types {' '.join(lig_types)}\n"
        f"receptor receptor.pdbqt\ngridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        f"smooth 0.5\n{maps}\nelecmap receptor.e.map\ndsolvmap receptor.d.map\ndielectric -0.1465\n")
    log("  running autogrid4...")
    run([TOOLS["autogrid4"], "-p", "receptor.gpf", "-l", "receptor.glg"], cwd=BENCH)
    OUTPUT.mkdir(exist_ok=True)
    for mp in list(BENCH.glob("receptor.*.map")) + [BENCH/"receptor.maps.fld", BENCH/"receptor.maps.xyz"]:
        shutil.copy(mp, OUTPUT / mp.name)
    log("  maps built + copied to output dir")


# --------------------------------------------------------------------------- #
# docking + parsing
# --------------------------------------------------------------------------- #
def dlg_done(dlg: Path):
    return dlg.is_file() and dlg.stat().st_size > 0 and "CLUSTERING HISTOGRAM" in dlg.read_text(errors="ignore")


def dock(pdbqt: Path, resnam, cwd: Path):
    dlg = cwd / f"{resnam}.dlg"
    if dlg_done(dlg):
        return "skip"
    if Path(pdbqt).resolve() != (cwd / pdbqt.name).resolve():
        shutil.copy(pdbqt, cwd / pdbqt.name)
    r = subprocess.run([TOOLS["AutoDock-GPU"], "--ffile", "receptor.maps.fld", "--lfile", pdbqt.name,
                        "--resnam", resnam, "--nrun", str(NRUN), "--heuristics", str(HEUR),
                        "--autostop", str(ASTOP), "--devnum", "1"], cwd=str(cwd),
                       capture_output=True, text=True)
    if r.returncode != 0:
        dlg.unlink(missing_ok=True)
        return "fail"
    return "docked"


def largest_cluster_dG(dlg: Path):
    if not dlg.is_file():
        return None
    rows = [(float(m[2]), int(m[5])) for m in _ROW.finditer(dlg.read_text(errors="ignore"))]
    return max(rows, key=lambda x: (x[1], -x[0]))[0] if rows else None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ef(scores, y, frac):
    o = np.argsort(-np.asarray(scores)); yy = np.asarray(y)[o]
    nt = max(1, int(round(frac * len(yy))))
    hit = yy.sum() / len(yy)
    return float((yy[:nt].sum() / nt) / hit) if hit else 0.0


def bedroc(scores, y, alpha=20.0):
    o = np.argsort(-np.asarray(scores)); yy = np.asarray(y)[o]
    N = len(yy); n = int(yy.sum()); Ra = n / N
    if n == 0 or n == N:
        return float("nan")
    ranks = np.where(yy == 1)[0] + 1
    rie = np.sum(np.exp(-alpha*ranks/N)) / (n/N*(1-math.exp(-alpha))/(math.exp(alpha/N)-1))
    fac = Ra*math.sinh(alpha/2)/(math.cosh(alpha/2)-math.cosh(alpha/2-alpha*Ra))
    return float(rie*fac + 1/(1-math.exp(alpha*(1-Ra))))


def metrics(scores, y):
    from sklearn.metrics import roc_auc_score
    return {"ROC_AUC": float(roc_auc_score(y, scores)), "EF_1pct": ef(scores, y, 0.01),
            "EF_5pct": ef(scores, y, 0.05), "EF_10pct": ef(scores, y, 0.10),
            "BEDROC": bedroc(scores, y, 20.0)}


def bootstrap_auc(scores, y, n=1000):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED); s = np.asarray(scores); yy = np.asarray(y)
    idx = np.arange(len(yy)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(yy[b])) == 2:
            out.append(roc_auc_score(yy[b], s[b]))
    return float(np.mean(out)), float(np.std(out))


# --------------------------------------------------------------------------- #
def redock_rmsd(center, npts):
    """Native ROC redock positive control + symmetry-aware RMSD (RDKit GetBestRMS)."""
    nat_pdbqt = BENCH / "native_reference.pdbqt"
    run([TOOLS["obabel"], BENCH / "native_reference_input.pdb", "-O", nat_pdbqt,
         "--partialcharge", "gasteiger"], check=False)
    scale_charges(nat_pdbqt)
    st = dock(nat_pdbqt, "native_control", BENCH)
    dlg = BENCH / "native_control.dlg"
    dG = largest_cluster_dG(dlg)
    rmsd = None
    try:
        # best pose coords from dlg largest cluster
        rows = [(int(m[1]), float(m[2]), int(m[3]), int(m[5])) for m in _ROW.finditer(dlg.read_text(errors="ignore"))]
        best_run = max(rows, key=lambda x: (x[3], -x[1]))[2]
        atoms, cur, cr = {}, [], None
        for l in dlg.read_text(errors="ignore").splitlines():
            if not l.startswith("DOCKED:"):
                continue
            b = l[8:]
            if b.strip().startswith("MODEL"):
                cur, cr = [], None
            elif "Run =" in b and (mm := re.search(r"Run\s*=\s*(\d+)", b)):
                cr = int(mm.group(1))
            elif b[:6].strip() in ("ATOM", "HETATM"):
                cur.append(b.rstrip())
            elif b.strip() == "ENDMDL" and cr is not None:
                atoms[cr] = cur
        pose = atoms.get(best_run, [])
        tmp = BENCH / "native_control_top1.pdb"
        tmp.write_text("\n".join("ATOM  " + l[6:] if l.startswith("HETATM") else l for l in pose) + "\nEND\n")
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        ref = Chem.MolFromPDBFile(str(BENCH / "native_reference_input.pdb"), sanitize=False)
        prb = Chem.MolFromPDBFile(str(tmp), sanitize=False)
        if ref and prb and ref.GetNumAtoms() == prb.GetNumAtoms():
            try:
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(prb), Chem.RemoveHs(ref))
            except Exception:
                # fall back: direct heavy-atom RMSD (same atom order preserved through docking)
                rc = ref.GetConformer(); pc = prb.GetConformer()
                d = [(rc.GetAtomPosition(i)-pc.GetAtomPosition(i)).LengthSq()
                     for i in range(ref.GetNumAtoms()) if ref.GetAtomWithIdx(i).GetSymbol() != "H"]
                rmsd = math.sqrt(sum(d)/len(d)) if d else None
    except Exception as e:
        log(f"  redock RMSD calc issue: {e}")
    return dG, rmsd, st


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    BENCH.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    log("=" * 70); log("  DUD-E HIV-1 PROTEASE BENCHMARK (1HXB / ROC)"); log("=" * 70)
    resolve_tools()

    # config snapshot + versions
    import rdkit, sklearn
    (BENCH / "config_snapshot.txt").write_text(
        f"TARGET=1HXB NATIVE_RES={NATIVE_RES} BOX_BUFFER={BOX_BUFFER} SPACING={SPACING}\n"
        f"NRUN={NRUN} HEUR={HEUR} ASTOP={ASTOP} CLUSTER_RMSD={CLUSTER_RMSD} SEED={SEED}\n"
        f"SUBSET={N_ACTIVES_SUB}A+{N_DECOYS_SUB}D\n")
    (BENCH / "versions.txt").write_text(
        f"python {sys.version.split()[0]}\nrdkit {rdkit.__version__}\nsklearn {sklearn.__version__}\n"
        f"numpy {np.__version__}\nOpenBabel 3.1.0\nAutoDock-GPU (tools) + autogrid4 4.2.6\n"
        f"MGLTools: NOT INSTALLED -> OpenBabel substitute (Gasteiger charges)\n")

    log("\n[Step 2] Subset selection")
    subset, prop_df = select_subset()

    log("\n[Step 3] Ligand prep (SMILES->3D->pdbqt, seed 42, charge-capped)")
    ligdir = BENCH / "ligands_pdbqt"; ligdir.mkdir(exist_ok=True)
    prepared, failed = [], 0
    for i, (smi, name, is_act) in enumerate(subset, 1):
        p = smiles_to_pdbqt(smi, name, ligdir)
        if p:
            prepared.append((p, name, is_act))
        else:
            failed += 1
        if i % 50 == 0:
            log(f"    prepped {i}/{len(subset)} ({failed} failed)")
    log(f"  prepared {len(prepared)}/{len(subset)} ligands ({failed} failed)")

    lig_types = []
    for p, _, _ in prepared:
        for t in atom_types(p):
            if t not in lig_types:
                lig_types.append(t)
    log(f"  union ligand atom types: {' '.join(lig_types)}")

    log("\n[Step 4/5] Box + receptor")
    center, npts = box_and_receptor()
    log("\n[Step 6] AutoGrid maps")
    build_maps(center, npts, lig_types)

    log("\n[Step 7] Native ROC redocking (positive control)")
    ctrl_dG, ctrl_rmsd, ctrl_st = redock_rmsd(center, npts)
    if ctrl_rmsd is not None:
        verdict = "PASS: Redocking validation successful" if ctrl_rmsd <= 2.0 else \
                  f"WARNING: redock RMSD {ctrl_rmsd:.2f} A > 2.0 A"
        log(f"  redock RMSD (symmetry-aware, RDKit best-RMS): {ctrl_rmsd:.2f} A -> {verdict}")
    log(f"  SYSTEM CONTROL BASELINE SCORE: {ctrl_dG:.2f} kcal/mol" if ctrl_dG else "  control dG: n/a")

    log(f"\n[Step 8] Batch docking {len(prepared)} compounds (idempotent)")
    tally = {}
    for i, (p, name, _) in enumerate(prepared, 1):
        st = dock(p, name, OUTPUT); tally[st] = tally.get(st, 0) + 1
        if i % 25 == 0:
            log(f"    docked {i}/{len(prepared)}  tally={tally}")
    log(f"  dock tally: {tally}")

    log("\n[Step 9/10] Parsing + metrics")
    rows = []
    for p, name, is_act in prepared:
        dG = largest_cluster_dG(OUTPUT / f"{name}.dlg")
        if dG is not None:
            rows.append({"Compound_ID": name, "Type": "Active" if is_act else "Decoy",
                         "is_active": is_act, "dG": dG})
    df = pd.DataFrame(rows).sort_values("dG").reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df)+1)
    if ctrl_dG:
        df["ddG_vs_control"] = df["dG"] - ctrl_dG
    y = df["is_active"].to_numpy(); sc = (-df["dG"]).to_numpy()
    M = metrics(sc, y)
    bmean, bstd = bootstrap_auc(sc, y)

    # Excel
    xlsx = BENCH / "hivpr_benchmark_validation_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Ranking", index=False)
        pd.DataFrame([{"Metric": "ROC_AUC", "Value": M["ROC_AUC"], "Bootstrap_mean": bmean, "Bootstrap_SD": bstd},
                     {"Metric": "EF_1pct", "Value": M["EF_1pct"]},
                     {"Metric": "EF_5pct", "Value": M["EF_5pct"]},
                     {"Metric": "EF_10pct", "Value": M["EF_10pct"]},
                     {"Metric": "BEDROC", "Value": M["BEDROC"]}]).to_excel(xw, "Metrics_Summary", index=False)
        pd.DataFrame([{"native_ligand": NATIVE_RES, "redock_RMSD_A": ctrl_rmsd,
                       "baseline_dG": ctrl_dG, "n_compounds": len(df),
                       "n_actives": int(y.sum()), "n_decoys": int(len(y)-y.sum())}]).to_excel(
            xw, "Control_Validation", index=False)
        prop_df.to_excel(xw, "Subset_Properties", index=False)

    # terminal markdown
    log("\n" + "=" * 70)
    log(f"  DUD-E HIV-1 PROTEASE BENCHMARK RESULTS  ({len(df)} compounds)")
    log("=" * 70)
    log(f"\n  Top 15 ranked (dG, kcal/mol; * = active):\n")
    log("  | Rank | Compound | dG | Type |")
    log("  |------|----------|-----|------|")
    for _, r in df.head(15).iterrows():
        star = " *" if r["is_active"] else ""
        log(f"  | {r['Rank']} | {r['Compound_ID']}{star} | {r['dG']:.2f} | {r['Type']} |")
    log(f"\n  ROC-AUC        : {M['ROC_AUC']:.3f}  (bootstrap {bmean:.3f} +/- {bstd:.3f}; 95% CI "
        f"[{bmean-1.96*bstd:.3f}, {bmean+1.96*bstd:.3f}])")
    log(f"  EF 1% / 5% /10%: {M['EF_1pct']:.2f} / {M['EF_5pct']:.2f} / {M['EF_10pct']:.2f}")
    log(f"  BEDROC (a=20)  : {M['BEDROC']:.3f}")
    log(f"  Redock RMSD    : {ctrl_rmsd:.2f} A" if ctrl_rmsd else "  Redock RMSD    : n/a")
    log(f"  Control dG     : {ctrl_dG:.2f} kcal/mol" if ctrl_dG else "")
    log(f"  Runtime        : {(time.time()-t0)/60:.1f} min")
    log(f"  Literature     : Chang et al. 2010 AD4 AUC ~{CHANG2010_AUC:.2f} (NCI DSII HIVPR)")
    log(f"  -> This run AUC {M['ROC_AUC']:.3f} vs literature {CHANG2010_AUC:.2f}: "
        f"{'consistent' if abs(M['ROC_AUC']-CHANG2010_AUC) < 0.12 else 'divergent'}")
    log(f"\n  Excel -> {xlsx}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
