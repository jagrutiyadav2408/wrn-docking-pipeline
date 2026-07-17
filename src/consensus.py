"""Consensus ranking across replicate docking runs.

A single AutoDock-GPU run is stochastic, so "top hit" from one replicate is not
reliable. This module aggregates N replicate runs (different seeds) per compound
into mean/median/std statistics plus a rank-stability score, and ranks by mean dG.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("docksuite.consensus")


@dataclass
class ConsensusResult:
    """Output of :meth:`ConsensusRanker.rank`.

    Attributes:
        ranking: Consensus table (one row per compound), sorted by ``mean_dG``.
        replicates: Wide table of per-replicate dG (one column per replicate).
        mean_std: Mean of per-compound ``std_dG`` (kcal/mol) - reproducibility.
        mean_spread: Mean of per-compound (max - min) dG (kcal/mol).
        n_stable: Compounds present in the top-N of *every* replicate.
        top_n: The N used for the stability window.
    """
    ranking: pd.DataFrame
    replicates: pd.DataFrame
    mean_std: float
    mean_spread: float
    n_stable: int
    top_n: int


class ConsensusRanker:
    """Aggregate replicate docking runs into a consensus ranking.

    Args:
        stability_top_n: Window used for ``rank_stability`` (default 20).
    """

    def __init__(self, stability_top_n: int = 20) -> None:
        self.top_n = max(1, int(stability_top_n))

    def rank(self, long_df: pd.DataFrame, id_col: str = "id",
             replicate_col: str = "replicate", score_col: str = "dG") -> ConsensusResult:
        """Compute consensus statistics from long-form replicate results.

        Args:
            long_df: Rows of ``(id, replicate, dG)``; dG is the AutoDock free
                energy (more negative = better).
            id_col: Compound id column.
            replicate_col: Replicate index column.
            score_col: Score column.

        Returns:
            A populated :class:`ConsensusResult`.

        Raises:
            ValueError: If ``long_df`` is empty.
        """
        if long_df.empty:
            raise ValueError("no replicate results to rank")

        wide = long_df.pivot_table(index=id_col, columns=replicate_col, values=score_col)
        n_reps = wide.shape[1]

        stats = pd.DataFrame({
            "Compound_ID": wide.index,
            "mean_dG": wide.mean(axis=1).to_numpy(),
            # sample std; undefined for a single replicate -> 0.0
            "std_dG": (wide.std(axis=1, ddof=1).fillna(0.0).to_numpy()
                       if n_reps > 1 else [0.0] * len(wide)),
            "median_dG": wide.median(axis=1).to_numpy(),
        })

        # rank_stability: fraction of replicates in which the compound is top-N
        top_sets = [set(wide[c].sort_values().index[:self.top_n]) for c in wide.columns]
        stats["rank_stability"] = [
            sum(cid in s for s in top_sets) / len(top_sets) for cid in wide.index
        ]

        # single-run rank from the first replicate, for comparison
        first = wide[wide.columns[0]]
        first_rank = first.rank(method="min").astype(int)
        stats["single_run_rank"] = [int(first_rank[cid]) for cid in wide.index]

        stats = stats.sort_values("mean_dG").reset_index(drop=True)
        stats["consensus_rank"] = range(1, len(stats) + 1)
        stats = stats[["Compound_ID", "mean_dG", "std_dG", "median_dG",
                       "consensus_rank", "rank_stability", "single_run_rank"]]

        spread = (wide.max(axis=1) - wide.min(axis=1))
        result = ConsensusResult(
            ranking=stats,
            replicates=wide.rename(columns=lambda c: f"replicate_{c}").reset_index(),
            mean_std=float(stats["std_dG"].mean()),
            mean_spread=float(spread.mean()),
            n_stable=int((stats["rank_stability"] >= 1.0).sum()),
            top_n=self.top_n,
        )
        logger.info("consensus over %d replicates: mean std=%.2f, mean spread=%.2f kcal/mol, "
                    "%d compounds stable in top-%d",
                    n_reps, result.mean_std, result.mean_spread, result.n_stable, self.top_n)
        return result


__all__ = ["ConsensusRanker", "ConsensusResult"]
