"""Retrospective benchmark metrics: ROC-AUC, enrichment factors, BEDROC.

All functions are generic over active/decoy counts and ratios — nothing assumes
a particular library size. Scores follow the convention *higher = better ranked*
(pass ``-dG`` for AutoDock free energies, or CNN affinity for Gnina).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger("docksuite.metrics")


@dataclass
class BenchmarkMetrics:
    """Container for computed benchmark metrics.

    Attributes:
        roc_auc: Area under the ROC curve.
        ef: Enrichment factors keyed by fraction (e.g. ``{0.01: 4.0}``).
        bedroc: BEDROC score at the configured alpha.
        auc_ci95: 95% bootstrap confidence interval for ROC-AUC.
        n_actives: Number of actives in the ranking.
        n_decoys: Number of decoys in the ranking.
        extra: Any additional scalar metrics.
    """
    roc_auc: float
    ef: dict[float, float]
    bedroc: float
    auc_ci95: tuple[float, float]
    n_actives: int
    n_decoys: int
    extra: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        """Flatten to a single dict suitable for a DataFrame/report row."""
        row = {"ROC_AUC": self.roc_auc, "BEDROC": self.bedroc,
               "AUC_CI95_low": self.auc_ci95[0], "AUC_CI95_high": self.auc_ci95[1],
               "n_actives": self.n_actives, "n_decoys": self.n_decoys}
        row.update({f"EF_{int(k*100)}pct": v for k, v in self.ef.items()})
        row.update(self.extra)
        return row


def enrichment_factor(scores: Sequence[float], labels: Sequence[int], frac: float) -> float:
    """Enrichment factor at the top ``frac`` of the ranked library.

    Args:
        scores: Ranking scores (higher = better).
        labels: Binary labels (1=active, 0=decoy).
        frac: Top fraction in (0, 1], e.g. ``0.01`` for EF 1%.

    Returns:
        EF = (hit-rate in top fraction) / (hit-rate overall); 0 if no actives.
    """
    y = np.asarray(labels)[np.argsort(-np.asarray(scores, float))]
    n_top = max(1, int(round(frac * len(y))))
    base = y.sum() / len(y)
    return float((y[:n_top].sum() / n_top) / base) if base else 0.0


def bedroc(scores: Sequence[float], labels: Sequence[int], alpha: float = 20.0) -> float:
    """Boltzmann-enhanced discrimination of ROC (Truchon & Bayly, 2007).

    Args:
        scores: Ranking scores (higher = better).
        labels: Binary labels (1=active, 0=decoy).
        alpha: Early-recognition emphasis (20.0 ~ first 8% of the list).

    Returns:
        BEDROC in [0, 1]; ``nan`` if all-active or all-decoy.
    """
    y = np.asarray(labels)[np.argsort(-np.asarray(scores, float))]
    N, n = len(y), int(np.sum(y))
    if n == 0 or n == N:
        return float("nan")
    ra = n / N
    ranks = np.where(y == 1)[0] + 1
    rie = np.sum(np.exp(-alpha * ranks / N)) / (
        n / N * (1 - math.exp(-alpha)) / (math.exp(alpha / N) - 1))
    factor = ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
    value = rie * factor + 1 / (1 - math.exp(alpha * (1 - ra)))
    return float(min(1.0, max(0.0, value)))         # BEDROC is defined on [0, 1]; clip FP epsilon


class MetricsEngine:
    """Compute a full :class:`BenchmarkMetrics` from a ranking table.

    Args:
        ef_fractions: Fractions at which to compute enrichment factors.
        bedroc_alpha: Alpha for BEDROC.
        bootstrap: Number of bootstrap resamples for the ROC-AUC CI.
        seed: RNG seed for reproducible bootstrapping.
    """

    def __init__(self, ef_fractions: Sequence[float] = (0.01, 0.05, 0.10),
                 bedroc_alpha: float = 20.0, bootstrap: int = 1000, seed: int = 42) -> None:
        self.ef_fractions = tuple(ef_fractions)
        self.bedroc_alpha = bedroc_alpha
        self.bootstrap = bootstrap
        self.seed = seed

    def calculate(self, rankings, score_col: str = "score",
                  label_col: str = "is_active") -> BenchmarkMetrics:
        """Compute metrics from a DataFrame with score and label columns.

        Args:
            rankings: DataFrame containing ``score_col`` (higher=better) and
                ``label_col`` (1/0). For AutoDock pass ``score = -dG``.
            score_col: Name of the score column.
            label_col: Name of the binary-label column.

        Returns:
            Populated :class:`BenchmarkMetrics`.

        Raises:
            ValueError: If the ranking lacks both classes.
        """
        from sklearn.metrics import roc_auc_score

        scores = rankings[score_col].to_numpy(float)
        labels = rankings[label_col].to_numpy(int)
        if len(set(labels)) < 2:
            raise ValueError("ranking must contain both actives and decoys")

        auc = float(roc_auc_score(labels, scores))
        ef = {f: enrichment_factor(scores, labels, f) for f in self.ef_fractions}
        bed = bedroc(scores, labels, self.bedroc_alpha)
        ci = self._bootstrap_ci(scores, labels)
        m = BenchmarkMetrics(roc_auc=auc, ef=ef, bedroc=bed, auc_ci95=ci,
                             n_actives=int(labels.sum()), n_decoys=int((labels == 0).sum()))
        logger.info("ROC-AUC=%.3f CI95=[%.3f, %.3f] BEDROC=%.3f EF=%s",
                    auc, ci[0], ci[1], bed, {int(k*100): round(v, 2) for k, v in ef.items()})
        return m

    def _bootstrap_ci(self, scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(labels))
        vals = []
        for _ in range(self.bootstrap):
            b = rng.choice(idx, len(idx), replace=True)
            if len(set(labels[b])) == 2:
                vals.append(roc_auc_score(labels[b], scores[b]))
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
