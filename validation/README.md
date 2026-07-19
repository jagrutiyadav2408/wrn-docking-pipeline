# Validation — retrospective benchmarks

Standalone scripts that produced the numbers quoted in the top-level README. They
are kept separate from the `docksuite` package deliberately: each is a
self-contained, auditable record of one benchmark run, not library code.

All of them read/write under a portable data root:

```bash
export DOCKSUITE_DATA=/path/to/working/dir     # default: ./validation_data
python validation/benchmark_hivpr.py
```

External binaries required: `autogrid4`, `AutoDock-GPU`, `obabel` (see main README).

| Script | Benchmark | Measured result |
|--------|-----------|-----------------|
| `benchmark_hivpr.py` | DUD-E HIV-1 protease (1HXB / native **ROC**), 62 actives + 188 decoys | **ROC-AUC 0.806** (95% CI 0.738–0.872), BEDROC 0.868, EF1% 4.03; redock control **0.63 Å PASS** |
| `benchmark_p38a.py` | DEKOIS 2.0 p38α (1W82 / native **L10**), MATCH_INBUILT box | **ROC-AUC 0.398** |
| `benchmark_p38a_blind.py` | Same set, BLIND_DOCK box (whole Cα, capped 126 pts = 47.2 Å) | **ROC-AUC 0.473** |
| `benchmark_p38a_enhanced.py` | p38α + score-bias corrections (size normalization, shape filter) | **ROC-AUC 0.240** — normalization made it *worse* |

## What these actually show

**The same AD4 scoring function succeeds on one target and fails on another:**

| Target | Active site | ROC-AUC |
|--------|-------------|---------|
| HIV-1 protease | hydrophobic, buried, C2-symmetric | **0.806** |
| p38α kinase | polar, solvent-exposed ATP pocket | **0.40–0.47** |

This is the expected behaviour of AD4's Lennard-Jones vdW term (cf. Chang et al.
2010, AD4 AUC ≈ 0.69 on NCI DSII HIVPR) — not a pipeline defect. The p38α
diagnosis was that the native-ligand box sits on the DFG pocket and under-covers
the hinge where the ATP-competitive actives bind; enlarging it to a blind box
recovered only +0.075 AUC (within bootstrap noise), because a 47 Å box also gives
decoys more off-target surface to score against.

**Negative results are kept on purpose.** `benchmark_p38a_enhanced.py` shows that
regressing ΔG on heavy-atom count removed the size correlation (r −0.640 → 0.000)
yet *lowered* ROC-AUC to 0.240 — evidence that the size bias was not the root
cause. Reproducing failures matters as much as reproducing the 0.806.

## Known deviations from the original specs

Recorded here so reviewers can audit them rather than discover them:

- **1HXB's native ligand is `ROC`** (Ro 31-8959 / saquinavir), not `MK1` — `MK1` is
  in 1HSG. It is also modelled in **two alternate conformations (altloc A/B)**;
  using both yields a bogus 98-atom "ligand" and a meaningless −29 kcal/mol
  control. The scripts take a single altloc (49 atoms).
- **DUD-E HIVPR ships 536 actives / 35,750 decoys**, not 62 / 3,100. The
  250-compound subset therefore *samples* 62 actives (seed 42) rather than
  retaining "all" of them; decoys are MW-quartile stratified when a plain random
  draw fails 80% property-range coverage.
- **MGLTools is not used.** `prepare_receptor4.py` / `prepare_ligand4.py` could not
  be installed non-interactively, so Open Babel is substituted throughout. This
  means **Gasteiger charges, not Kollman** — a real methodological difference.
- p38α native ligand is `L10`, not `095`.
