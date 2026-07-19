# Troubleshooting

## GPU Device Index Off-by-One
AutoDock-GPU uses 1-based device indexing (`--devnum 1` for the first GPU),
while the configuration follows CUDA's 0-based convention (`gpu_device: 0`).
The package handles this conversion internally. If you see empty `.dlg`
files with no stderr, verify your device index with `nvidia-smi`.

## Autostop Defeats Seeding
`--autostop 1` (early termination) makes docking non-deterministic because
stopping depends on GPU thread convergence timing. Use `--autostop 0`
(`docking.deterministic: true`) for reproducible results. Measured on
AutoDock-GPU v1.6: the same seed with autostop on gave −16.66 / −17.62 / −16.94
kcal/mol; with autostop off, −17.58 / −17.59.

## Single Seed Is Not Enough
AutoDock-GPU's `--seed` accepts up to three comma-separated integers (pose init,
mutation, crossover). Passing one value leaves the other two seeded from
time/PID, so runs still vary. The package always passes `--seed s,s,s`.

## MGLTools vs. Open Babel
MGLTools `prepare_receptor4.py` / `prepare_ligand4.py` require interactive
installation and are not used. Open Babel 3.1.0 with Gasteiger charges is the
default for all charge assignment. This differs from the classic AutoDock
protocol, which uses Kollman charges.

## Altloc Bug (PDB 1HXB)
1HXB models ROC in two conformations (altloc A + B). The package extracts only
the first altloc to prevent a doubled 98-atom "ligand" and the resulting
meaningless ~ −29 kcal/mol control score.

## Empty .dlg Files
If all `.dlg` files are empty: check the GPU device index, CUDA version
compatibility, and that the `autodock_gpu` binary is on `PATH`.

## Local PDB Files
For structures not in RCSB (e.g., a proprietary 9S18), set `target.pdb_path`
to the local file. `target.pdb_id` is then used only for output naming.

## largest_cluster Ranking Is Unstable
If your ranking changes between runs, you are almost certainly using
`pose_selection: largest_cluster`. It gets *worse* with more GA runs (more
near-tied clusters flip between seeds). Use the default `lowest_energy`.

## ADMET-AI Values Look Saturated
Some ADMET-AI endpoints (notably DILI) saturate near 1.0 for large, beyond-Ro5
chemotypes that are out of the model's training distribution. `ADMETProfiler`
detects non-discriminating endpoints and flags them series-wide rather than
letting them dominate the risk tiers. Treat ML ADMET values as relative triage.
