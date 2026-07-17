# GNINA Backend for the Virtual-Screening Pipeline

An **optional CNN scorer** that plugs into the existing AutoDock-GPU pipeline. Same
receptor prep, same ligand prep, same quality gates, same grid box — only the
**scoring backend** changes. GNINA ([Koes lab, U. Pittsburgh](https://github.com/gnina/gnina))
replaces the AutoDock-4 empirical function with a 3D-CNN trained on PDBbind (~22k
crystal structures), which typically discriminates actives from decoys far better on
retrospective benchmarks (DEKOIS/DUD-E ROC-AUC often > 0.70).

> **Two corrections vs. the original design note**
> 1. **GNINA does not use AutoGrid `.fld/.maps` files.** Like smina/vina it takes the
>    receptor, the ligand, and an explicit box (`--center_x/y/z`, `--size_x/y/z`). The
>    module supplies the box from the same center/size the pipeline already computes.
> 2. **There is no native Windows GNINA binary.** On Windows, run it through **WSL2** or
>    **Docker** (both with GPU passthrough). Set `GNINA_LAUNCHER` accordingly.

---

## Files

| File | Purpose |
|------|---------|
| `src/gnina_interface.py` | `GninaInterface` class + quality gates (dock / rescore / minimize, parsing, comparison) |
| `src/benchmark_gnina.py` | DEKOIS retrospective validation; reuses AutoDock dlgs, computes ROC/EF/BEDROC, honest gnina-absent fallback |
| `tests/test_gnina_interface.py` | pytest unit tests (no gnina/GPU needed) |
| `config/benchmark_gnina.json` | Config template with all new parameters |

---

## Installation

GNINA needs CUDA + Linux userland. Pick **one** launcher.

### Option A — Docker (simplest on Windows; needs Docker Desktop + WSL2 backend)
```powershell
docker pull gnina/gnina:latest
docker run --rm --gpus all gnina/gnina:latest gnina --version
```
Set in config: `"GNINA_LAUNCHER": "docker"`. Paths are auto-mapped into `/work`
(the module bind-mounts the working directory).

### Option B — WSL2 (native-speed, recommended for large screens)
```powershell
wsl --install -d Ubuntu        # first time only; reboot when prompted
```
Then **inside Ubuntu**:
```bash
# NVIDIA driver on Windows already exposes the GPU to WSL2 (nvidia-smi should work).
sudo apt-get update && sudo apt-get install -y libboost-all-dev libopenbabel-dev
wget https://github.com/gnina/gnina/releases/latest/download/gnina -O ~/gnina
chmod +x ~/gnina && sudo mv ~/gnina /usr/local/bin/gnina
gnina --version
```
Set in config: `"GNINA_LAUNCHER": "wsl"`. Windows paths (`C:\...`) are auto-translated
to `/mnt/c/...`.

### Option C — Native Linux / macOS
```bash
wget https://github.com/gnina/gnina/releases/latest/download/gnina
chmod +x gnina && sudo mv gnina /usr/local/bin/
gnina --version
```
Set `"GNINA_LAUNCHER": "native"`.

### CUDA requirements
- CUDA 11.8 or 12.x runtime (bundled in the release binary / Docker image)
- GPU compute capability ≥ 6.0 — **RTX 4060 = sm_86 ✓** (driver 591.74 present on this host)
- The release binary carries its own CUDA runtime, so a system CUDA toolkit is **not** required.

---

## Usage

### Run the DEKOIS benchmark (AutoDock baseline + gnina, honest reporting)
```bash
python src/benchmark_gnina.py config/benchmark_gnina.json
```
- If gnina is reachable → runs it, computes ROC/EF/BEDROC, writes
  `gnina_benchmark_results.xlsx`, `gnina_comparison.csv`, and three figures.
- If gnina is **not** reachable → prints the AutoDock-GPU baseline and the install
  hint, and **does not invent** gnina numbers.

### Programmatic
```python
from gnina_interface import GninaInterface
gi = GninaInterface(cfg)
if gi.validate_installation():
    res = gi.run_gnina_dock("receptor.pdbqt", "lig.pdbqt", "out/lig",
                            center=(2.19, 18.38, 38.37), box_size=(24, 24, 24))
    print(res["cnn_affinity"])
```

---

## Configuration parameters

| Key | Meaning |
|-----|---------|
| `SCORING_BACKEND` | `AUTODOCK` \| `GNINA` \| `BOTH` |
| `GNINA_MODE` | `dock` (full search) \| `rescore` (`--score_only` on AD4 pose) \| `minimize` (`--local_only`) |
| `GNINA_LAUNCHER` | `native` \| `wsl` \| `docker` |
| `GNINA_EXHAUSTIVENESS` | search thoroughness (1–4096; default 32). Auto-reduced 32→16→8 on CUDA OOM |
| `GNINA_NUM_MODES` / `GNINA_ENERGY_RANGE` | max poses / max ΔE (kcal/mol) |
| `GNINA_CNN_MODEL` | `default` \| `crossdock_default2018` \| `redock_default2018` \| `dense` \| `general_default2018` |
| `GNINA_CNN_SCORING` | `rescore` (default) \| `refinement` \| `all` \| `none` |
| `GNINA_MINIMIZE` / `GNINA_SEED` / `GNINA_DEVICE` | local minimization / RNG seed / GPU index |
| `COMPARE_BACKENDS` | run AutoDock + gnina and emit correlation + top-N overlap |

---

## Quality gates (applied to every ligand **before** gnina)

1. **Torsional integrity** — assert an active torsion tree (gnina reuses the AutoDock tree).
2. **Electrostatic capping** — proportionally scale so max|q| ≤ 1.0 e (preserves net charge shape).
3. **Conformer shape filter** — flag Rg > 1.5× (or SASA > 2.0×) native and log it.
4. **Grid-box validation** — MATCH_INBUILT: box encloses 100% of native heavy atoms;
   BLIND: box covers ≥ 90% of the Cα backbone.

---

## Reproducibility

Every run appends a block to `versions.txt` (gnina version, launcher, CUDA, GPU,
CNN model, seed, exhaustiveness, minimize). All gnina CLI arguments are logged at
`DEBUG` level.

---

## Platform support

| Platform | gnina | How |
|---|---|---|
| Linux + CUDA | ✅ | release binary on PATH, `GNINA_LAUNCHER="native"` |
| Windows | ⚠️ **no native build exists** | WSL2 (`"wsl"`) or Docker Desktop (`"docker"`) |
| macOS | ⚠️ | Docker (CPU only; slow) |

GPU requirements: NVIDIA driver ≥ 525 for CUDA 12.x (compute capability ≥ 6.0).
The release binary/Docker image bundles its own CUDA runtime — no system CUDA
toolkit needed.
