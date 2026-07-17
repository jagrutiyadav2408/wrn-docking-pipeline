"""End-to-end orchestration: prepare -> load -> gate -> grid -> dock -> score -> report."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig
from .consensus import ConsensusRanker, ConsensusResult
from .docking import DockingEngine
from .grid import GridEngine
from .ligands import LigandLibrary
from .metrics import BenchmarkMetrics, MetricsEngine
from .poses import PoseExtractor
from .quality import QualityGates, radius_of_gyration
from .report import ReportGenerator
from .target import TargetPreparator
from .util import configure_logging, progress

logger = logging.getLogger("docksuite.benchmark")


class BenchmarkRunner:
    """Single-entry orchestration for benchmarks and screening campaigns.

    Args:
        log_level: Logging level applied when :meth:`run` starts.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    def run(self, config_path: str, overrides: Optional[dict] = None) -> BenchmarkMetrics:
        """Run the full pipeline described by ``config_path``.

        Steps: receptor prep -> library load/subset -> ligand prep + quality
        gates -> grid box + AutoGrid maps -> idempotent batch docking -> pose
        extraction -> metrics (labelled sets) -> report bundle.

        Args:
            config_path: Path to the JSON config.
            overrides: Optional dotted-path overrides (from the CLI).

        Returns:
            The computed :class:`BenchmarkMetrics` for labelled libraries, or a
            metrics object with ``roc_auc=nan`` for unlabelled screens.
        """
        configure_logging(self._log_level)
        cfg = PipelineConfig(config_path, overrides)

        bench_dir = Path(cfg.get("output.benchmark_dir"))
        out_dir = Path(cfg.get("output.output_dir"))
        bench_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        lig_dir = bench_dir / "ligands_pdbqt"
        lig_dir.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.get("ligands.random_seed", 42))
        save_intermediates = bool(cfg.get("output.save_intermediates", True))
        if not save_intermediates:
            logger.warning("output.save_intermediates=false -> .dlg/_top1.pdb/.log are "
                           "discarded after scoring; this DISABLES resume/crash-recovery")
        logger.info("intermediates (.dlg/_top1.pdb/.log) -> %s | report -> %s",
                    out_dir, bench_dir)

        # 1. receptor -------------------------------------------------------
        target = TargetPreparator().prepare(cfg)

        # 2. library --------------------------------------------------------
        library = LigandLibrary(seed=seed)
        df = library.load(cfg)

        # 3. ligand prep + quality gates -----------------------------------
        native_rg = radius_of_gyration_of(target.native_ligand_pdb)
        gates = QualityGates(
            charge_cap=float(cfg.get("quality.charge_cap", 1.0)),
            rg_factor=float(cfg.get("quality.rg_factor", 1.5)),
            native_rg=native_rg)
        prepared, ligand_types = [], []
        for row in progress(df.to_dict("records"), "prep"):
            pdbqt = library.prepare_ligand(row, lig_dir, seed)
            if pdbqt is None:
                logger.warning("prep failed: %s", row["id"])
                continue
            gates.process(pdbqt)
            prepared.append((pdbqt, row))
            for t in _atom_types(pdbqt):
                if t not in ligand_types:
                    ligand_types.append(t)
        logger.info("prepared %d/%d ligands", len(prepared), len(df))

        # 4. grid + maps ----------------------------------------------------
        grid = GridEngine(spacing=float(cfg.get("target.grid_spacing", 0.375)))
        box = grid.calculate_box(target, cfg.get("target.search_mode"),
                                 float(cfg.get("target.box_buffer_angstroms", 4.0)))
        fld = grid.generate_maps(target.receptor_pdbqt, box, ligand_types, bench_dir)
        for mp in list(bench_dir.glob("receptor.*.map")) + [fld, bench_dir / "receptor.maps.xyz"]:
            if mp.is_file():
                _copy(mp, out_dir / mp.name)

        # 5. dock N replicates (seeded, idempotent, crash-recovering) -------
        engine = DockingEngine(cfg.get("docking.backend"), cfg)
        extractor = PoseExtractor(
            "largest_cluster" if cfg.get("docking.pose_selection", "largest_cluster")
            == "largest_cluster" else "lowest_energy")
        base_seed = int(cfg.get("docking.seed", seed))
        n_reps = max(1, int(cfg.get("docking.consensus_runs", 3)))
        logger.info("consensus: %d replicate(s), seeds %s", n_reps,
                    ", ".join(str(base_seed + r) for r in range(n_reps)))

        long_rows: list[dict] = []
        best_pose: dict[str, object] = {}
        for r in range(n_reps):
            rep_seed = base_seed + r
            for pdbqt, row in progress(prepared, f"dock r{r+1}/{n_reps} (seed {rep_seed})"):
                name = f"{row['id']}__s{rep_seed}"      # seed in name => seed-aware idempotency
                try:
                    engine.dock(pdbqt, out_dir / "receptor.maps.fld", name, out_dir, seed=rep_seed)
                except RuntimeError as e:               # crash recovery: skip, retry next run
                    logger.error("docking failed for %s (seed %d): %s", row["id"], rep_seed, e)
                    continue
                pose = extractor.extract(out_dir / f"{name}.dlg")
                if pose is None:
                    continue
                long_rows.append({"id": row["id"], "replicate": r + 1, "dG": pose.delta_g,
                                  "cluster_size": pose.cluster_size})
                prev = best_pose.get(row["id"])
                if prev is None or pose.delta_g < prev.delta_g:
                    best_pose[row["id"]] = pose         # keep the best pose across replicates

        long_df = pd.DataFrame(long_rows)
        if long_df.empty:
            raise RuntimeError("no docking results produced")

        # 6. consensus ranking ---------------------------------------------
        consensus = ConsensusRanker(int(cfg.get("docking.stability_top_n", 20))).rank(long_df)
        cons = consensus.ranking

        # representative pose (best replicate) + optional cleanup
        for cid, pose in best_pose.items():
            if save_intermediates:
                pose.to_pdb(out_dir / f"{cid}_top1.pdb")
        if not save_intermediates:
            for f in list(out_dir.glob("*__s*.dlg")) + list(out_dir.glob("*__s*.log")):
                f.unlink(missing_ok=True)

        # rankings frame for metrics/report: consensus mean dG is the score
        active_map = {row["id"]: int(row["is_active"]) for _, row in prepared
                      if "is_active" in row}
        rankings = pd.DataFrame({
            "id": cons["Compound_ID"], "dG": cons["mean_dG"], "score": -cons["mean_dG"],
            "std_dG": cons["std_dG"], "rank_stability": cons["rank_stability"],
            "rank": cons["consensus_rank"],
        })
        if active_map:
            rankings["is_active"] = rankings["id"].map(active_map)

        # 7. metrics + report ----------------------------------------------
        metrics = None
        if "is_active" in rankings and rankings["is_active"].nunique() == 2:
            metrics = MetricsEngine(seed=seed).calculate(rankings)
        report = ReportGenerator(bench_dir, bool(cfg.get("output.generate_figures", True)))
        report.generate(rankings, metrics, cfg, extra_sheets={
            "Consensus_Ranking": cons,
            "Single_Run_Replicates": consensus.replicates,
        })
        self._log_consensus_summary(consensus, n_reps)

        # 8. verify what actually landed on disk ---------------------------
        self._verify_outputs(out_dir, expected_runs=len(prepared) * n_reps,
                             expected_poses=len(prepared), saved=save_intermediates)

        if metrics is None:                               # unlabelled screen
            metrics = BenchmarkMetrics(float("nan"), {}, float("nan"),
                                       (float("nan"), float("nan")), 0, len(rankings))
        return metrics

    @staticmethod
    def _log_consensus_summary(result: ConsensusResult, n_reps: int) -> None:
        """Print the consensus top-10, stability count, and reproducibility metric."""
        cons = result.ranking
        lines = ["", "=" * 72,
                 f"  CONSENSUS RANKING  ({n_reps} replicate runs)", "=" * 72,
                 f"  {'#':>2}  {'Compound':<16}{'mean dG':>10} {'+/- std':>8}"
                 f"{'stability':>11}{'1-run rank':>12}"]
        for _, r in cons.head(10).iterrows():
            lines.append(f"  {int(r.consensus_rank):>2}  {r.Compound_ID:<16}"
                         f"{r.mean_dG:>10.2f} {r.std_dG:>8.2f}"
                         f"{r.rank_stability:>11.2f}{int(r.single_run_rank):>12}")
        lines += ["-" * 72,
                  f"  stable in top-{result.top_n} across ALL {n_reps} replicates : "
                  f"{result.n_stable}/{len(cons)}",
                  f"  reproducibility: mean per-compound std = {result.mean_std:.2f} kcal/mol",
                  f"                   mean per-compound spread (max-min) = {result.mean_spread:.2f}",
                  "=" * 72]
        logger.info("\n".join(lines))

    @staticmethod
    def _verify_outputs(out_dir: Path, expected_runs: int, expected_poses: int,
                        saved: bool) -> dict[str, int]:
        """Log a count of the artifacts in ``out_dir`` by kind.

        Args:
            out_dir: Directory holding the docking artifacts.
            expected_runs: ligands x replicates (one .dlg/.log per run).
            expected_poses: ligands (one representative _top1.pdb each).
            saved: Whether intermediates were meant to be kept.

        Returns:
            Mapping of artifact kind -> count.
        """
        counts = {
            ".dlg": len(list(out_dir.glob("*.dlg"))),
            "_top1.pdb": len(list(out_dir.glob("*_top1.pdb"))),
            ".log": len(list(out_dir.glob("*.log"))),
        }
        logger.info("output_dir %s contains: %s", out_dir,
                    ", ".join(f"{k}={v}" for k, v in counts.items()))
        if saved:
            for kind, want in ((".dlg", expected_runs), (".log", expected_runs),
                               ("_top1.pdb", expected_poses)):
                if counts[kind] < want:
                    logger.warning("only %d/%d %s files present in %s",
                                   counts[kind], want, kind, out_dir)
        return counts


# --------------------------------------------------------------------------- #
def radius_of_gyration_of(native_pdb) -> Optional[float]:
    """Rg of the native ligand pdb, or None if absent (BLIND_DOCK)."""
    if not native_pdb or not Path(native_pdb).is_file():
        return None
    return radius_of_gyration(Path(native_pdb))


def _atom_types(pdbqt: Path) -> list[str]:
    seen: list[str] = []
    for l in pdbqt.read_text(errors="ignore").splitlines():
        if l.startswith(("ATOM", "HETATM")) and len(l) > 77:
            t = l[77:].strip().split()[0]
            if t and t not in seen:
                seen.append(t)
    return seen


def _copy(src: Path, dst: Path) -> None:
    import shutil
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
