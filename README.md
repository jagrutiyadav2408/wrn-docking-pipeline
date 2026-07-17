# docksuite

A configuration-driven, **target-agnostic** molecular docking & virtual-screening
pipeline. The same code screens **any** protein target against **any** compound
library — you change a JSON file, not the source. AutoDock-GPU backend, with a
Gnina CNN scorer included.

Validated retrospectively on **DUD-E HIV-1 protease (ROC-AUC 0.806)** and
**DEKOIS 2.0 p38α**, using the modules in this repo. See [`validation/`](validation/).

---

## Why

No PDB IDs, residue names, chain lists, paths, or box sizes are hardcoded anywhere.
Every run is fully described by one JSON config, so results are reproducible and
the pipeline is auditable.

| Component | Supports |
|-----------|----------|
| Target | any PDB ID (auto-fetched) **or a local `.pdb`**, any native-ligand residue, any chains |
| Library | `.smi` · DUD-E · DEKOIS · **directory of 3D `.mol2`**, 10 → 10,000+ compounds |
| Search mode | `MATCH_INBUILT` (native-ligand-centred) or `BLIND_DOCK` (whole Cα backbone) |
| Backend | `AUTODOCK` (AutoDock-GPU) · `GNINA` (CNN scorer) |
| Ranking | single-run or **N-replicate consensus** with rank-stability |
| ADMET | `ALL` / `TOP_N` / `NONE` (RDKit + ADMET-AI, tiered risk) |

---

## Install

```bash
git clone https://github.com/<you>/docksuite.git && cd docksuite
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[chem,dev]"
pytest -q          # 50 tests, no GPU or external binaries needed
```

**External binaries** (not pip-installable; needed only at *run* time):

| Tool | Role | Notes |
|------|------|-------|
| AutoDock-GPU | docking | CUDA build |
| autogrid4 | grid maps | AutoDock 4.2.6 |
| Open Babel (`obabel`) | receptor/ligand prep | used *instead of* MGLTools → **Gasteiger, not Kollman, charges** |
| gnina *(optional)* | CNN scorer | Linux/CUDA only — see [`README_GNINA.md`](README_GNINA.md) |

Auto-discovered on `PATH` and conventional install dirs, or point at them explicitly
via `docking.autodock_gpu_path` / `$DOCKSUITE_AUTODOCK_GPU`, `$DOCKSUITE_OBABEL`, …

---

## Quick start

```bash
# Retrospective benchmark (auto-downloads the PDB + DUD-E set)
python scripts/run_benchmark.py --config config/benchmark_hivpr.json

# Prospective screen of your own library
cp config/screen_example.json config/local/my_screen.json   # edit paths
python scripts/run_screen.py  --config config/local/my_screen.json

# ADMET-profile the finished screen (uses the consensus ranking)
python scripts/run_admet.py   --config config/local/my_screen.json

# Override any value on the CLI (dotted paths)
python scripts/run_benchmark.py --config config/benchmark_hivpr.json \
    --target.pdb_id 1W82 --ligands.subset_size 40
```

```python
from src import BenchmarkRunner
m = BenchmarkRunner().run("config/benchmark_hivpr.json")
print(m.roc_auc, m.ef, m.bedroc)
```

---

## Configuration reference

```jsonc
{
  "target": {
    "pdb_id": "1HXB",               // any RCSB id
    "pdb_path": "./local.pdb",      // OPTIONAL: use a local structure instead of fetching
    "native_ligand_name": "ROC",    // residue name; null for BLIND_DOCK
    "chains_to_keep": ["A", "B"],   // [] = all chains
    "search_mode": "MATCH_INBUILT", // or "BLIND_DOCK"
    "box_buffer_angstroms": 4.0,
    "grid_spacing": 0.375
  },
  "ligands": {
    "source": "DUDE",               // DUDE | DEKOIS | SMI | CUSTOM | MOL2_DIR
    "target_name": "hivpr",         // DUD-E target      (source=DUDE)
    "path": "./lib.smi",            // file or mol2 dir  (other sources)
    "subset_size": 250,             // null = use all
    "active_decoy_ratio": null,     // e.g. 0.25 => 1 active : 3 decoys
    "stratify_by": "molecular_weight",
    "random_seed": 42
  },
  "docking": {
    "backend": "AUTODOCK",
    "runs_per_ligand": 100,
    "heuristics": true,
    "autostop": true,               // ignored when deterministic=true (see below)
    "pose_selection": "lowest_energy",  // or "largest_cluster"
    "cluster_rmsd_threshold": 2.0,
    "seed": 42,                     // falls back to ligands.random_seed
    "deterministic": true,          // seeded runs; disables autostop
    "consensus_runs": 3,            // replicates with seeds s, s+1, s+2 …
    "stability_top_n": 20
  },
  "quality": { "charge_cap": 1.0, "rg_factor": 1.5 },
  "admet":   { "mode": "NONE", "top_n": null },
  "output":  { "benchmark_dir": "./bench", "output_dir": "./out",
               "generate_figures": true, "save_intermediates": true },
  "hardware": { "gpu_device": 0, "n_workers": 4 }
}
```

`benchmark_dir` holds the **report** (`.xlsx`, figures); `output_dir` holds the
**raw artifacts** (`.dlg`, `_top1.pdb`, `.log`). They are different directories —
the runner logs both at startup.

---

## Reproducibility — read this before trusting a ranking

Measured on AutoDock-GPU v1.6 / RTX 4060, and the reason several defaults are what
they are:

- **`--seed` alone does not make AutoDock-GPU deterministic.** It takes *three*
  comma-separated seeds; supplying one leaves the rest seeded from time/PID. The
  backend always passes `--seed s,s,s`.
- **`--autostop` defeats seeding entirely** — it halts on convergence detected
  across GPU threads, so timing decides the stopping point (same seed measured at
  −16.66 / −17.62 / −16.94 kcal/mol). `deterministic: true` disables it, and it is
  also ~24% *faster*. Runs are then near-deterministic (±0.02 kcal/mol residual
  from floating-point non-associativity in parallel reductions) — not bit-exact.
- **`pose_selection` dominates reproducibility, not run count.** On a 100-compound
  screen (3 replicates):

  | Selection | mean per-compound σ | resolved (σ<0.3) |
  |---|---|---|
  | `largest_cluster` | 0.79–1.30 kcal/mol | 4–19 / 100 |
  | `lowest_energy` | **0.05 kcal/mol** | **99 / 100** |

  `largest_cluster` gets *worse* with more GA runs (more near-tied clusters, so
  "most populated" flips between seeds). Quadrupling `runs_per_ligand` 100→400 made
  σ worse (0.79→1.30), not better. `lowest_energy` is the default for ranking;
  `largest_cluster` remains available and is often preferred for predicting the
  bound *geometry*.
- **Use `consensus_runs` ≥ 3 and check `rank_stability`** before believing a hit.

---

## Architecture

```
PipelineConfig   → load / validate / dotted-path override
TargetPreparator → fetch or load local PDB, clean, single-altloc native, receptor.pdbqt
LigandLibrary    → load / subset / stratify / SMILES→3D or mol2→pdbqt
QualityGates     → torsion integrity · charge cap · conformer shape
GridEngine       → MATCH_INBUILT | BLIND_DOCK box + AutoGrid maps
DockingEngine    → AUTODOCK backend (+ GNINA hook), seeded, idempotent, crash-safe
PoseExtractor    → representative pose → DS-compatible PDB
ConsensusRanker  → mean/median/σ + rank-stability across replicates
MetricsEngine    → ROC-AUC · EF · BEDROC + bootstrap CI
AdmetProfiler    → RDKit + ADMET-AI, saturation-aware risk tiers
ReportGenerator  → Excel + figures + terminal Markdown
BenchmarkRunner  → orchestrates all of it (single entry: .run(config))
```

**Idempotency**: existing `.dlg` files are skipped (seed-aware — a changed seed
re-runs), so an interrupted screen resumes. **Crash recovery**: a failed ligand
never leaves a partial `.dlg`; it retries on the next run.

---

## Known limitations

Stated plainly, because they affect interpretation:

- **Open Babel replaces MGLTools** → Gasteiger charges rather than Kollman for the
  receptor. Affects absolute ΔG more than relative ranking, but it is a deviation
  from the classic AutoDock protocol.
- **AD4 scoring is target-dependent.** Excellent on buried hydrophobic sites
  (HIVPR 0.806); poor on polar kinase ATP pockets (p38α 0.40–0.47). Benchmark your
  target before trusting a prospective ranking.
- **The GNINA backend is not wired into `BenchmarkRunner`** — it is exposed via
  `src.gnina_interface` / `scripts/run_gnina_benchmark.py`. `DockingEngine`
  raises `NotImplementedError` for `backend="GNINA"`.
- **ADMET-AI predictions can saturate** on out-of-distribution chemotypes (e.g.
  DILI ≈ 1.0 for every compound in a beyond-Ro5 series). `AdmetProfiler` detects
  non-discriminating endpoints and flags them series-wide instead of letting them
  dominate tiers — but treat all ML ADMET values as *relative triage*, not truth.
- Metrics bootstrap CIs are computed for ROC-AUC only, not EF/BEDROC.

---

## Repo layout

```
src/          package (config, target, ligands, grid, docking, quality, poses,
               consensus, metrics, admet, report, benchmark, gnina_interface)
scripts/       CLIs: run_benchmark · run_screen · run_admet · run_gnina_benchmark
config/        JSON configs (config/local/ is gitignored for machine-specific ones)
tests/         50 pytest tests — no GPU/binaries required
validation/    standalone retrospective benchmarks + their measured results
examples/      working dirs, populated on first run
```

---

## Citation

If this pipeline supports a publication, please cite the underlying tools:

- **AutoDock-GPU** — Santos-Martins et al., *J. Chem. Theory Comput.* 2021.
- **AutoDock4** — Morris et al., *J. Comput. Chem.* 2009.
- **Open Babel** — O'Boyle et al., *J. Cheminform.* 2011.
- **RDKit** — https://www.rdkit.org
- **gnina** — Ragoza et al., *J. Chem. Inf. Model.* 2017; McNutt et al., *J. Cheminform.* 2021.
- **ADMET-AI** — Swanson et al., *Bioinformatics* 2024.
- Benchmarks: **DUD-E** (Mysinger et al., 2012), **DEKOIS 2.0** (Bauer et al., 2013).

MIT licensed — see [LICENSE](LICENSE).
