# Changelog

## v1.2.0 (Current)
- Fixed: `lowest_energy` pose selection replaces `largest_cluster` as the default
  (eliminates reproducibility artifacts; measured per-compound σ 0.79–1.30 →
  0.05 kcal/mol on a 100-compound screen)
- Fixed: GPU device index off-by-one (CUDA 0-indexed → AutoDock-GPU 1-indexed)
- Fixed: Altloc extraction bug (PDB 1HXB single-altloc only)
- Fixed: Torsion gate false-positive (Open Babel legend text matched by the regex)
- Fixed: ADMET NaN-defaults-to-PASS on unparseable SMILES (now tiered UNPARSED)
- Added: Deterministic seeding (`--seed s,s,s` + `--autostop 0`)
- Added: Local PDB file support (proprietary structures via `target.pdb_path`)
- Added: MOL2 library support (pre-built 3D conformers via `MOL2_DIR`)
- HIVPR benchmark: AUC = 0.806, 15/15 top hits, redock control 0.63 Å

## v1.1.0
- Added: Consensus ranking module (N replicates, rank-stability scoring)
- Added: ADMET tiered stratification (EXCLUDE / HIGH RISK / MODERATE / PASS)

## v1.0.0
- Initial release
- AutoDock-GPU integration
- Quality gates (torsional, electrostatic)
- Algorithmic grid box extraction
- Idempotent execution
