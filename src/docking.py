"""Backend-agnostic docking. AUTODOCK (AutoDock-GPU) implemented; GNINA hook ready."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Protocol

from .util import resolve_tool, run

logger = logging.getLogger("docksuite.docking")


class _Backend(Protocol):
    """Docking-backend interface: produce a result file for one ligand."""
    def dock(self, ligand: Path, maps: Path, output_name: str, out_dir: Path,
             seed: int | None = None) -> Path: ...


class AutoDockBackend:
    """AutoDock-GPU backend (AD4 empirical scoring), idempotent + crash-safe.

    Reproducibility notes (measured on AutoDock-GPU v1.6, RTX 4060):
      * ``--seed`` accepts up to THREE comma-separated integers. Passing a single
        value leaves the other two seeded from time/PID, so runs still vary. We
        always pass ``--seed s,s,s``.
      * ``--autostop`` terminates on convergence detected across GPU threads, so
        its stopping point depends on thread timing and defeats seeding entirely
        (same seed measured at -16.66/-17.62/-16.94). ``docking.deterministic``
        therefore disables autostop; it also measured ~24% FASTER.
      * Even then runs are near- (not bit-) deterministic: ~0.02 kcal/mol residual
        from floating-point non-associativity in parallel GPU reductions.

    Args:
        config: A :class:`~vspipeline.config.PipelineConfig`.
    """

    def __init__(self, config) -> None:
        self._bin = resolve_tool("autodock_gpu", config.get("docking.autodock_gpu_path"))
        self.nrun = int(config.get("docking.runs_per_ligand", 100))
        self.heuristics = 1 if config.get("docking.heuristics", True) else 0
        self.deterministic = bool(config.get("docking.deterministic", True))
        autostop_cfg = bool(config.get("docking.autostop", True))
        if self.deterministic and autostop_cfg:
            logger.warning("docking.deterministic=true -> disabling --autostop "
                           "(it defeats seeding; measured ~24 pct faster too)")
        self.autostop = 0 if self.deterministic else (1 if autostop_cfg else 0)
        # docking.seed falls back to ligands.random_seed
        self.seed = int(config.get("docking.seed", config.get("ligands.random_seed", 42)))
        # config gpu_device is 0-indexed (CUDA/nvidia-smi convention); AutoDock-GPU
        # --devnum is 1-indexed (must be >= 1), so add 1. Honour gpu_device under
        # either docking (manuscript schema) or hardware.
        gpu_dev = config.get("docking.gpu_device", config.get("hardware.gpu_device", 0))
        self.device = str(int(gpu_dev) + 1)
        self.save_intermediates = bool(config.get("output.save_intermediates", True))

    def dock(self, ligand: Path, maps: Path, output_name: str, out_dir: Path,
             seed: int | None = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = self.seed if seed is None else int(seed)
        dlg = out_dir / f"{output_name}.dlg"
        # Seed-aware idempotency: the caller encodes the seed in output_name
        # (``<id>__s<seed>``), so a changed seed yields a different .dlg and re-runs.
        if self._done(dlg):
            logger.debug("skip %s (already docked with this seed)", output_name)
            return dlg
        if Path(ligand).resolve() != (out_dir / ligand.name).resolve():
            shutil.copy(ligand, out_dir / ligand.name)
        logger.info("Docking seed = %d for %s", seed, ligand.stem)
        proc = run([self._bin, "--ffile", maps.name, "--lfile", ligand.name,
                    "--resnam", output_name, "--nrun", str(self.nrun),
                    "--heuristics", str(self.heuristics), "--autostop", str(self.autostop),
                    "--seed", f"{seed},{seed},{seed}",     # all three or it stays random
                    "--devnum", self.device], cwd=out_dir, check=False)

        # persist the engine's stdout/stderr (written even on failure - it is the
        # only record of why a ligand died once the partial .dlg is removed)
        if self.save_intermediates:
            log_path = out_dir / f"{output_name}.log"
            logger.info("Saving run log to %s", log_path)
            log_path.write_text((proc.stdout or "")
                                + (f"\n[STDERR]\n{proc.stderr}" if proc.stderr else ""),
                                encoding="utf-8", errors="replace")

        if proc.returncode != 0 or not dlg.is_file():
            dlg.unlink(missing_ok=True)                 # never leave a partial .dlg
            raise RuntimeError(f"AutoDock-GPU failed for {output_name}: {proc.stderr[-300:]}")
        logger.info("Saving .dlg to %s", dlg)
        return dlg

    @staticmethod
    def _done(dlg: Path) -> bool:
        return dlg.is_file() and dlg.stat().st_size > 0 and \
            "CLUSTERING HISTOGRAM" in dlg.read_text(errors="ignore")


class GninaBackend:
    """Placeholder for the Gnina CNN backend (interface reserved).

    The full implementation lives in the companion ``gnina_interface`` module;
    this hook keeps :class:`DockingEngine` backend-agnostic today.
    """

    def __init__(self, config) -> None:
        self._config = config

    def dock(self, ligand: Path, maps: Path, output_name: str, out_dir: Path,
             seed: int | None = None) -> Path:
        raise NotImplementedError(
            "GNINA backend not wired in this build; use backend='AUTODOCK'. "
            "See gnina_interface.GninaInterface for the CNN scorer.")


_BACKENDS = {"AUTODOCK": AutoDockBackend, "GNINA": GninaBackend}


class DockingEngine:
    """Dispatch docking to the configured backend.

    Args:
        backend: ``"AUTODOCK"`` or ``"GNINA"``.
        config: A :class:`~vspipeline.config.PipelineConfig`.

    Raises:
        ValueError: On an unknown backend name.
        RuntimeError: If the selected backend's binary is unavailable.
    """

    def __init__(self, backend: str, config) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {sorted(_BACKENDS)}")
        self.backend_name = backend
        self._backend: _Backend = _BACKENDS[backend](config)
        if backend == "AUTODOCK" and not getattr(self._backend, "_bin", None):
            raise RuntimeError("AutoDock-GPU binary not found; set docking.autodock_gpu_path")

    def dock(self, ligand: Path, maps: Path, output_name: str, out_dir: Path,
             seed: int | None = None) -> Path:
        """Dock one ligand and return the result file path.

        Args:
            ligand: Prepared ligand ``.pdbqt``.
            maps: Grid field (``receptor.maps.fld``) in ``out_dir``.
            output_name: Result basename (``<name>.dlg``); encode the seed here
                (e.g. ``<id>__s42``) so idempotency is seed-aware.
            out_dir: Output directory (must contain the maps).
            seed: RNG seed for this run; ``None`` uses the configured seed.

        Returns:
            Path to the produced ``.dlg`` (or backend-specific result).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._backend.dock(ligand, maps, output_name, out_dir, seed)
