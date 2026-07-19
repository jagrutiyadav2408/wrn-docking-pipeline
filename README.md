# WRN Docking Pipeline

An automated, quality-gated framework for high-throughput molecular docking
and localized pharmacokinetic profiling. GPU-accelerated with AutoDock-GPU,
integrated with ADMET-AI for on-device ADMET prediction.

## Features

- **Algorithmic pocket extraction**: MATCH_INBUILT (native-ligand centered)
  or BLIND_DOCK (full protein) modes
- **Mandatory quality gates**: Torsional integrity + electrostatic capping
- **Deterministic GPU docking**: `--seed 42,42,42` with `--autostop 0` for
  near-bit-reproducible trajectories (±0.02 kcal/mol residual variation)
- **Lowest-energy pose selection**: Reproducible ranking (vs. largest-cluster
  coin-flip artifacts)
- **Local ADMET profiling**: ADMET-AI/Chemprop, no web servers, no data upload
- **Idempotent execution**: Skip completed runs, resume from crashes
- **Consumer hardware**: Validated on RTX 4060 laptop (8 GB VRAM), ~25 min
  for the 250-compound benchmark

## Installation

```bash
git clone https://github.com/yourname/wrn-docking-pipeline.git
cd wrn-docking-pipeline
pip install -r requirements.txt
pip install -e .
```

### External Dependencies

Not pip-installable; obtained separately and placed on `PATH`. The exact
versions used for the reported results are in the machine-generated provenance
record `data/hivpr/versions.txt`.

- AutoDock-GPU (CUDA build)
- AutoGrid 4.2.6
- Open Babel 3.1.0 (Gasteiger charges — MGLTools is not used)
- ADMET-AI (Chemprop backend)

## Quick Start: HIV-1 Protease Benchmark

```bash
python scripts/run_benchmark.py --config config/benchmark_hivpr.json
```

Expected output: ROC-AUC ≈ 0.806 on the 62-active / 188-decoy DUD-E HIVPR subset.

## Quick Start: WRN Prospective Screen

```bash
python scripts/run_screen.py --config config/screen_wrn.json
```

## Configuration

All behaviour is controlled via JSON config files — no PDB IDs, residue names,
chains, paths, or box sizes are hardcoded in the source. See `config/` for
examples and `config/template_custom.json` to start a new target.

Reproducibility knobs live under `docking`: `deterministic: true` passes
`--seed s,s,s` and disables `--autostop` (which otherwise defeats seeding);
`pose_selection: lowest_energy` is the default and is far more reproducible than
`largest_cluster`.

## Validation

Retrospective benchmarks and their measured results are in `validation/`. The
DUD-E HIV-1 protease benchmark (1HXB / native ROC) reaches ROC-AUC 0.806 with a
redocking control of 0.63 Å; DEKOIS 2.0 p38α scores 0.40–0.47, the expected
behaviour of the AD4 vdW term on a polar kinase pocket.

## Citation

[Your paper citation here]

## License

MIT License — see [LICENSE](LICENSE).
