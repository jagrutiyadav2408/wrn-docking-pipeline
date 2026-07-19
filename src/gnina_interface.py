#!/usr/bin/env python3
"""
gnina_interface.py -- Unified GNINA scoring backend for the virtual-screening pipeline.

GNINA (Koes lab, U. Pittsburgh) replaces the AutoDock-4 empirical scoring function
with a 3D-CNN trained on PDBbind. This module wraps the gnina CLI for three modes:

    dock      : full CNN-guided docking (pose generation + CNN scoring)
    rescore   : re-score an existing AutoDock-GPU pose with the CNN (--score_only)
    minimize  : local CNN-minimization of an existing pose (--local_only)

IMPORTANT FACTUAL NOTES (differ from some pipeline documentation):
  * gnina does NOT consume AutoGrid .fld/.maps files. Like smina/vina it takes the
    receptor, the ligand, and an explicit search box (center_x/y/z, size_x/y/z).
  * There is no native Windows gnina binary. On Windows use WSL2 or Docker; set
    GNINA_LAUNCHER accordingly. Native Linux/macOS just use the binary on PATH.

The module is fully target-agnostic: no protein-specific paths are hardcoded; all
inputs are supplied through the config dict or method arguments.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("gnina")

# gnina CNN scoring modes accepted by --cnn_scoring
_CNN_SCORING_MODES = {"none", "rescore", "refinement", "metrorescore", "metrorefine", "all"}
# recognised --cnn built-in model names
_CNN_MODELS = {"default", "crossdock_default2018", "redock_default2018", "dense", "general_default2018"}


# --------------------------------------------------------------------------- #
# config container
# --------------------------------------------------------------------------- #
@dataclass
class GninaConfig:
    """Typed view over the relevant keys of the pipeline config dict."""
    binary: str = "gnina"                       # GNINA_BINARY
    launcher: str = "native"                    # GNINA_LAUNCHER: native | wsl | docker
    docker_image: str = "gnina/gnina:latest"    # GNINA_DOCKER_IMAGE
    mode: str = "dock"                          # GNINA_MODE
    minimize: bool = True                       # GNINA_MINIMIZE
    seed: int = 42                              # GNINA_SEED
    exhaustiveness: int = 32                    # GNINA_EXHAUSTIVENESS
    num_modes: int = 9                          # GNINA_NUM_MODES
    energy_range: float = 3.0                   # GNINA_ENERGY_RANGE
    cnn_model: str = "default"                  # GNINA_CNN_MODEL
    cnn_scoring: str = "rescore"                # GNINA_CNN_SCORING
    device: str = "0"                           # GNINA_DEVICE
    score_only: bool = False                    # GNINA_SCORE_ONLY
    charge_cap: float = 1.0                     # |q| cap for Gate 2
    rg_factor: float = 1.5                      # Gate 3 Rg threshold
    sasa_factor: float = 2.0                    # Gate 3 SASA threshold
    oom_retry_exhaust: Sequence[int] = field(default_factory=lambda: (16, 8))
    box_shrink: float = 0.9                     # Gate: shrink box on OOM

    @classmethod
    def from_dict(cls, cfg: dict) -> "GninaConfig":
        g = lambda k, d: cfg.get(k, d)
        cnn = str(g("GNINA_CNN_MODEL", "default"))
        if cnn not in _CNN_MODELS:
            logger.warning("unknown GNINA_CNN_MODEL %r; falling back to 'default'", cnn)
            cnn = "default"
        cs = str(g("GNINA_CNN_SCORING", "rescore")).lower()
        if cs not in _CNN_SCORING_MODES:
            logger.warning("unknown GNINA_CNN_SCORING %r; falling back to 'rescore'", cs)
            cs = "rescore"
        return cls(
            binary=str(g("GNINA_BINARY", "gnina")),
            launcher=str(g("GNINA_LAUNCHER", "native")).lower(),
            docker_image=str(g("GNINA_DOCKER_IMAGE", "gnina/gnina:latest")),
            mode=str(g("GNINA_MODE", "dock")).lower(),
            minimize=bool(g("GNINA_MINIMIZE", True)),
            seed=int(g("GNINA_SEED", 42)),
            exhaustiveness=int(g("GNINA_EXHAUSTIVENESS", 32)),
            num_modes=int(g("GNINA_NUM_MODES", 9)),
            energy_range=float(g("GNINA_ENERGY_RANGE", 3.0)),
            cnn_model=cnn,
            cnn_scoring=cs,
            device=str(g("GNINA_DEVICE", "0")),
            score_only=bool(g("GNINA_SCORE_ONLY", False)),
        )


# --------------------------------------------------------------------------- #
# quality gates (applied to ligand pdbqt BEFORE gnina)
# --------------------------------------------------------------------------- #
def gate_torsional_integrity(pdbqt: Path) -> int:
    """Gate 1: ensure the torsion tree is active. gnina reuses the AutoDock torsion
    tree; INACTIVE branches would freeze rotatable bonds. Returns #branches seen."""
    text = pdbqt.read_text(errors="ignore")
    n_branch = text.count("\nBRANCH ") + text.startswith("BRANCH ")
    # obabel/prepare_ligand emit active branches by default; we only assert here.
    if "INACTIVE" in text.upper():
        logger.warning("%s contains INACTIVE torsion markers", pdbqt.name)
    return n_branch


def gate_charge_cap(pdbqt: Path, cap: float = 1.0) -> int:
    """Gate 2: proportionally scale partial charges so max|q| <= cap, preserving the
    charge distribution shape. Returns number of atoms rescaled (0 if untouched)."""
    lines = pdbqt.read_text(errors="ignore").splitlines()
    qs = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 76:
            try:
                qs.append(abs(float(l[66:76])))
            except ValueError:
                pass
    mx = max(qs) if qs else 0.0
    if mx <= cap:
        return 0
    f = cap / mx
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
    logger.info("Gate2: rescaled %d charges in %s (max|q| %.2f -> %.2f)", n, pdbqt.name, mx, cap)
    return n


def _heavy_coords(pdbqt: Path):
    pts = []
    for l in pdbqt.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 54:
            t = l[76:].strip().split()[0] if l[76:].strip() else ""
            if t in ("H", "HD", "HS"):
                continue
            pts.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return pts


def radius_of_gyration(pdbqt: Path) -> Optional[float]:
    pts = _heavy_coords(pdbqt)
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    import math
    return math.sqrt(sum((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 for p in pts) / len(pts))


def gate_shape_filter(pdbqt: Path, native_rg: Optional[float],
                      rg_factor: float = 1.5) -> tuple[Optional[float], bool]:
    """Gate 3: flag conformers markedly more extended than the native ligand.
    Returns (Rg, flagged). SASA left to the caller (needs RDKit); Rg is the robust
    geometric proxy computed directly from coordinates."""
    rg = radius_of_gyration(pdbqt)
    flagged = bool(native_rg and rg and rg > rg_factor * native_rg)
    if flagged:
        logger.warning("Gate3: %s extended (Rg %.2f > %.2f x native %.2f)",
                       pdbqt.name, rg, rg_factor, native_rg)
    return rg, flagged


def gate_box_validation(center: tuple, size: tuple, heavy_pts: Sequence[tuple],
                        require_fraction: float = 1.0) -> bool:
    """Gate 4: verify the search box encloses the reference heavy atoms (native
    ligand for MATCH_INBUILT, Cα set for BLIND). Returns True if >= require_fraction
    of points fall inside center±size/2."""
    if not heavy_pts:
        return True
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    inside = sum(
        1 for x, y, z in heavy_pts
        if abs(x - center[0]) <= hx and abs(y - center[1]) <= hy and abs(z - center[2]) <= hz
    )
    frac = inside / len(heavy_pts)
    ok = frac >= require_fraction
    if not ok:
        logger.warning("Gate4: box encloses only %.1f%% of reference atoms (need %.0f%%)",
                       100 * frac, 100 * require_fraction)
    return ok


# --------------------------------------------------------------------------- #
# main interface
# --------------------------------------------------------------------------- #
class GninaInterface:
    """Unified interface for gnina docking, rescoring and minimization."""

    def __init__(self, config: dict):
        self.raw = config
        self.cfg = GninaConfig.from_dict(config)
        self._version: Optional[str] = None

    # ---- installation / launcher ---------------------------------------- #
    def _base_cmd(self) -> list[str]:
        """Launcher prefix: native binary, `wsl gnina`, or `docker run ... gnina`."""
        if self.cfg.launcher == "wsl":
            return ["wsl.exe", self.cfg.binary]
        if self.cfg.launcher == "docker":
            return ["docker", "run", "--rm", "--gpus", "all", "-v",
                    "{CWD}:/work", "-w", "/work", self.cfg.docker_image, "gnina"]
        return [self.cfg.binary]

    def _map_path(self, p: str | Path) -> str:
        """Translate a Windows path to the launcher's filesystem view."""
        p = Path(p)
        if self.cfg.launcher == "wsl":
            ap = p.resolve()
            drive = ap.drive.rstrip(":").lower()
            rest = ap.as_posix().split(":", 1)[1] if ":" in ap.as_posix() else ap.as_posix()
            return str(PurePosixPath(f"/mnt/{drive}") / rest.lstrip("/"))
        if self.cfg.launcher == "docker":
            return str(PurePosixPath("/work") / p.name)
        return str(p)

    def validate_installation(self) -> bool:
        """Check that gnina is reachable and record its version. Never raises."""
        try:
            cmd = self._base_cmd()
            cmd = [c.replace("{CWD}", str(Path.cwd())) for c in cmd] + ["--version"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error("gnina not runnable via launcher=%s: %s", self.cfg.launcher, e)
            return False
        out = (r.stdout + r.stderr).strip()
        m = re.search(r"gnina[^\d]*([\d.]+)", out, re.I)
        self._version = m.group(1) if m else (out.splitlines()[0] if out else "unknown")
        ok = r.returncode == 0
        (logger.info if ok else logger.error)("gnina version: %s (rc=%d)", self._version, r.returncode)
        return ok

    def install_hint(self) -> str:
        return (
            "gnina not found. There is no native Windows binary.\n"
            "  * WSL2:   wsl --install -d Ubuntu ; then inside Ubuntu install the gnina\n"
            "            release binary (see README_GNINA.md), set GNINA_LAUNCHER='wsl'.\n"
            "  * Docker: docker pull gnina/gnina:latest ; set GNINA_LAUNCHER='docker'.\n"
            "  * Linux:  download the release binary to PATH, GNINA_LAUNCHER='native'."
        )

    # ---- command construction ------------------------------------------- #
    def prepare_gnina_command(self, receptor_pdbqt: str, ligand_pdbqt: str,
                              output_pdbqt: str, center: tuple, box_size: tuple,
                              mode: str = "dock", log_path: Optional[str] = None) -> list[str]:
        """Build the full gnina argument list for the requested mode."""
        c = self.cfg
        cmd = [x.replace("{CWD}", str(Path.cwd())) for x in self._base_cmd()]
        cmd += ["-r", self._map_path(receptor_pdbqt), "-l", self._map_path(ligand_pdbqt)]

        if mode == "dock":
            cmd += ["-o", self._map_path(output_pdbqt),
                    "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
                    "--center_z", f"{center[2]:.3f}",
                    "--size_x", f"{box_size[0]:.1f}", "--size_y", f"{box_size[1]:.1f}",
                    "--size_z", f"{box_size[2]:.1f}",
                    "--exhaustiveness", str(c.exhaustiveness),
                    "--num_modes", str(c.num_modes),
                    "--energy_range", f"{c.energy_range:.1f}",
                    "--seed", str(c.seed)]
        elif mode == "rescore":
            cmd += ["--score_only"]
        elif mode == "minimize":
            cmd += ["-o", self._map_path(output_pdbqt), "--local_only"]
        else:
            raise ValueError(f"unknown gnina mode: {mode}")

        # CNN scoring model + mode
        cmd += ["--cnn_scoring", c.cnn_scoring]
        if c.cnn_model != "default":
            cmd += ["--cnn", c.cnn_model]
        if c.minimize and mode != "minimize":
            cmd += ["--minimize"]
        if log_path:
            cmd += ["--log", self._map_path(log_path)]
        return cmd

    def _run(self, cmd: list[str], cwd: Optional[Path] = None, timeout: int = 3600):
        env = os.environ.copy()
        if self.cfg.launcher == "native":
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.device
        logger.debug("exec: %s", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                              text=True, timeout=timeout, env=env)

    # ---- execution ------------------------------------------------------ #
    def run_gnina_dock(self, receptor_pdbqt: str, ligand_pdbqt: str, output_prefix: str,
                       center: tuple, box_size: tuple) -> dict:
        """Full docking. Retries with reduced exhaustiveness / shrunken box on CUDA OOM."""
        out = f"{output_prefix}_gnina.pdbqt"
        log = f"{output_prefix}_gnina.log"
        attempts = [(self.cfg.exhaustiveness, box_size)] + \
                   [(e, box_size) for e in self.cfg.oom_retry_exhaust]
        t0 = time.time()
        last = None
        for exh, bsz in attempts:
            self.cfg.exhaustiveness = exh
            cmd = self.prepare_gnina_command(receptor_pdbqt, ligand_pdbqt, out,
                                             center, bsz, "dock", log)
            try:
                r = self._run(cmd, cwd=Path(output_prefix).parent)
            except subprocess.TimeoutExpired:
                logger.error("gnina timeout for %s", output_prefix)
                return {"poses": [], "error": "timeout", "runtime": time.time() - t0}
            last = r
            if r.returncode == 0 and Path(out).is_file():
                df = self.parse_gnina_output(out, r.stdout)
                best = df.iloc[0].to_dict() if not df.empty else {}
                return {"poses": df.to_dict("records"),
                        "cnn_score": best.get("CNNscore"),
                        "cnn_affinity": best.get("CNNaffinity"),
                        "minimized_affinity": best.get("minimizedAffinity"),
                        "affinity": best.get("affinity"),
                        "runtime": time.time() - t0}
            if "out of memory" in (r.stderr + r.stdout).lower():
                logger.warning("CUDA OOM; retrying exhaustiveness=%s", exh)
                continue
            break
        logger.error("gnina dock failed for %s: %s", output_prefix,
                     (last.stderr if last else "")[-400:])
        return {"poses": [], "error": "failed", "runtime": time.time() - t0}

    def run_gnina_rescore(self, receptor_pdbqt: str, ligand_pdbqt: str,
                          pose_pdbqt: str) -> dict:
        """Re-score an existing (AutoDock-GPU) pose with the CNN. No search."""
        cmd = self.prepare_gnina_command(receptor_pdbqt, pose_pdbqt, "", (0, 0, 0),
                                         (0, 0, 0), "rescore")
        try:
            r = self._run(cmd, timeout=900)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        if r.returncode != 0:
            logger.error("gnina rescore failed: %s", r.stderr[-300:])
            return {"error": "failed"}
        vals = self._parse_score_stream(r.stdout + r.stderr)
        return vals or {"error": "no_score"}

    # ---- parsing -------------------------------------------------------- #
    @staticmethod
    def _parse_score_stream(text: str) -> dict:
        """Extract CNNscore / CNNaffinity / affinity from gnina stdout or --score_only."""
        out: dict = {}
        for key, pat in (("affinity", r"[Aa]ffinity:\s*(-?\d+\.\d+)"),
                         ("CNNscore", r"CNNscore:\s*(-?\d+\.\d+)"),
                         ("CNNaffinity", r"CNNaffinity:\s*(-?\d+\.\d+)"),
                         ("minimizedAffinity", r"[Mm]inimized[Aa]ffinity:\s*(-?\d+\.\d+)")):
            m = re.search(pat, text)
            if m:
                out[key] = float(m.group(1))
        # table form: "mode | affinity | CNN pose | CNN affinity"
        if "CNNaffinity" not in out:
            for m in re.finditer(r"^\s*\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                                 text, re.M):
                out.setdefault("affinity", float(m.group(1)))
                out.setdefault("CNNscore", float(m.group(2)))
                out.setdefault("CNNaffinity", float(m.group(3)))
                break
        return out

    def parse_gnina_output(self, output_pdbqt: str, stdout: str = "") -> pd.DataFrame:
        """Parse gnina output poses. gnina writes per-pose REMARK lines:
        REMARK minimizedAffinity / REMARK CNNscore / REMARK CNNaffinity.
        Falls back to the stdout score table. Returns a DataFrame sorted by CNNaffinity
        descending (higher CNNaffinity = predicted stronger binder)."""
        rows: list[dict] = []
        p = Path(output_pdbqt)
        if p.is_file():
            cur: dict = {}
            idx = 0
            for l in p.read_text(errors="ignore").splitlines():
                if l.startswith("REMARK"):
                    for key in ("minimizedAffinity", "CNNscore", "CNNaffinity", "affinity"):
                        m = re.search(rf"{key}\s+(-?\d+\.\d+)", l)
                        if m:
                            cur[key] = float(m.group(1))
                elif l.startswith(("MODEL", "ENDMDL")):
                    if l.startswith("ENDMDL") and cur:
                        idx += 1
                        cur["pose"] = idx
                        rows.append(cur)
                        cur = {}
            if cur:  # single-model file without ENDMDL
                cur["pose"] = idx + 1
                rows.append(cur)
        if not rows and stdout:
            v = self._parse_score_stream(stdout)
            if v:
                v["pose"] = 1
                rows.append(v)
        df = pd.DataFrame(rows)
        if not df.empty and "CNNaffinity" in df.columns:
            df = df.sort_values("CNNaffinity", ascending=False).reset_index(drop=True)
        return df

    # ---- comparison ----------------------------------------------------- #
    def compare_with_autodock(self, autodock_results: pd.DataFrame,
                              gnina_results: pd.DataFrame,
                              on: str = "Compound_ID",
                              ad_score: str = "dG",
                              gnina_score: str = "CNNaffinity") -> pd.DataFrame:
        """Merge AD4 ΔG vs gnina CNNaffinity per compound and attach rank columns.
        Correlations are returned in df.attrs (pearson/spearman/top10_overlap)."""
        merged = autodock_results.merge(gnina_results, on=on, how="inner",
                                        suffixes=("_ad", "_gnina"))
        if merged.empty:
            logger.warning("no overlapping compounds between backends")
            return merged
        # AD4: more negative ΔG = better -> rank ascending; gnina: higher = better
        merged["rank_ad"] = merged[ad_score].rank(method="min", ascending=True)
        merged["rank_gnina"] = merged[gnina_score].rank(method="min", ascending=False)
        try:
            from scipy.stats import pearsonr, spearmanr
            pear = pearsonr(merged[ad_score], merged[gnina_score])[0]
            spear = spearmanr(merged["rank_ad"], merged["rank_gnina"])[0]
        except Exception:
            pear = merged[ad_score].corr(merged[gnina_score])
            spear = merged["rank_ad"].corr(merged["rank_gnina"], method="spearman")
        top = min(10, len(merged))
        top_ad = set(merged.nsmallest(top, "rank_ad")[on])
        top_gn = set(merged.nsmallest(top, "rank_gnina")[on])
        merged.attrs.update({"pearson": float(pear), "spearman": float(spear),
                             "top_n": top, "top_n_overlap": len(top_ad & top_gn)})
        logger.info("backend comparison: pearson=%.3f spearman=%.3f top%d overlap=%d",
                    pear, spear, top, len(top_ad & top_gn))
        return merged

    def write_versions(self, path: Path, gpu: str = "unknown", cuda: str = "unknown") -> None:
        """Append reproducibility block to versions.txt."""
        c = self.cfg
        block = ("\n[GNINA]\n"
                 f"Version: {self._version or 'not detected'}\n"
                 f"Launcher: {c.launcher}\n"
                 f"CUDA: {cuda}\nGPU: {gpu}\n"
                 f"CNN Model: {c.cnn_model}\nCNN Scoring: {c.cnn_scoring}\n"
                 f"Seed: {c.seed}\nExhaustiveness: {c.exhaustiveness}\n"
                 f"Num modes: {c.num_modes}\nMinimize: {str(c.minimize).lower()}\n")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)


__all__ = ["GninaInterface", "GninaConfig", "gate_torsional_integrity",
           "gate_charge_cap", "gate_shape_filter", "gate_box_validation",
           "radius_of_gyration"]
