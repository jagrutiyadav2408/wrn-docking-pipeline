# Examples

Working directories for the shipped configs. Each is populated on first run
(receptor/maps/ligand pdbqts/results are written here) and is safe to delete.

| Dir | Config | Notes |
|-----|--------|-------|
| `hivpr_benchmark/` | `config/benchmark_hivpr.json` | DUD-E HIV-protease; auto-downloads 1HXB + DUD-E set |
| `custom_target/` | `config/benchmark_custom.json` | template; edit `library.smi` and the config |
| `wrn_screen/` | your own | prospective WRN campaign (unlabelled `.smi`) |

Run, e.g.:

```bash
python scripts/run_benchmark.py --config config/benchmark_hivpr.json
```
