"""Tests for MetricsEngine, enrichment factor, and BEDROC."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import MetricsEngine, bedroc, enrichment_factor


def _ranking(scores, labels):
    return pd.DataFrame({"score": scores, "is_active": labels})


def test_perfect_ranking_auc_is_one():
    # actives score highest -> AUC = 1.0
    df = _ranking([9, 8, 7, 3, 2, 1], [1, 1, 1, 0, 0, 0])
    m = MetricsEngine(bootstrap=200).calculate(df)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.n_actives == 3 and m.n_decoys == 3


def test_inverted_ranking_auc_is_zero():
    df = _ranking([1, 2, 3, 7, 8, 9], [1, 1, 1, 0, 0, 0])
    m = MetricsEngine(bootstrap=100).calculate(df)
    assert m.roc_auc == pytest.approx(0.0)


def test_enrichment_factor_perfect_top_fraction():
    scores = list(range(100, 0, -1))
    labels = [1] * 10 + [0] * 90            # all actives at the very top
    ef10 = enrichment_factor(scores, labels, 0.10)
    assert ef10 == pytest.approx(10.0)      # 100% actives found in top 10% -> 10x


def test_enrichment_factor_no_actives_is_zero():
    assert enrichment_factor([3, 2, 1], [0, 0, 0], 0.5) == 0.0


def test_bedroc_bounds_and_ordering():
    good = bedroc([9, 8, 7, 3, 2, 1], [1, 1, 1, 0, 0, 0])
    bad = bedroc([1, 2, 3, 7, 8, 9], [1, 1, 1, 0, 0, 0])
    assert 0.0 <= bad < good <= 1.0


def test_bedroc_nan_when_single_class():
    assert np.isnan(bedroc([1, 2, 3], [1, 1, 1]))


def test_calculate_requires_two_classes():
    df = _ranking([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError):
        MetricsEngine().calculate(df)


def test_metrics_as_row_is_flat():
    df = _ranking([9, 8, 7, 3, 2, 1], [1, 1, 1, 0, 0, 0])
    row = MetricsEngine(bootstrap=100).calculate(df).as_row()
    assert "ROC_AUC" in row and "EF_1pct" in row and "BEDROC" in row
    assert all(not isinstance(v, dict) for v in row.values())
