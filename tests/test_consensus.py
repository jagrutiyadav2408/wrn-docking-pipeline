"""Tests for ConsensusRanker (pure logic, no binaries)."""
import pandas as pd
import pytest

from src.consensus import ConsensusRanker


def _long(rows):
    """rows: {id: [dG_r1, dG_r2, ...]} -> long-form frame."""
    out = []
    for cid, vals in rows.items():
        for i, v in enumerate(vals, 1):
            out.append({"id": cid, "replicate": i, "dG": v})
    return pd.DataFrame(out)


def test_ranks_by_mean_dg():
    df = _long({"a": [-10.0, -10.0, -10.0], "b": [-8.0, -8.0, -8.0], "c": [-12.0, -12.0, -12.0]})
    res = ConsensusRanker(stability_top_n=2).rank(df)
    assert res.ranking["Compound_ID"].tolist() == ["c", "a", "b"]   # most negative first
    assert res.ranking.iloc[0]["consensus_rank"] == 1


def test_statistics_are_correct():
    df = _long({"a": [-10.0, -12.0, -14.0]})
    r = ConsensusRanker().rank(df).ranking.iloc[0]
    assert r.mean_dG == pytest.approx(-12.0)
    assert r.median_dG == pytest.approx(-12.0)
    assert r.std_dG == pytest.approx(2.0)          # sample std of (-10,-12,-14)


def test_zero_std_for_identical_replicates():
    df = _long({"a": [-9.0, -9.0, -9.0]})
    assert ConsensusRanker().rank(df).ranking.iloc[0].std_dG == pytest.approx(0.0)


def test_single_replicate_std_is_zero_not_nan():
    df = _long({"a": [-9.0], "b": [-7.0]})
    res = ConsensusRanker().rank(df)
    assert res.ranking["std_dG"].tolist() == [0.0, 0.0]
    assert res.mean_std == pytest.approx(0.0)


def test_rank_stability_full_and_partial():
    # 'a' is top-1 in every replicate; 'b' only in one of three
    df = _long({"a": [-10.0, -10.0, -10.0],
                "b": [-11.0, -1.0, -1.0],
                "c": [-2.0, -2.0, -2.0]})
    res = ConsensusRanker(stability_top_n=1).rank(df)
    stab = dict(zip(res.ranking["Compound_ID"], res.ranking["rank_stability"]))
    assert stab["a"] == pytest.approx(2 / 3)       # top-1 in replicates 2 and 3
    assert stab["b"] == pytest.approx(1 / 3)       # top-1 only in replicate 1
    assert stab["c"] == pytest.approx(0.0)


def test_n_stable_counts_only_fully_stable():
    df = _long({"a": [-10.0, -10.0], "b": [-9.0, -1.0], "c": [-1.0, -1.0]})
    res = ConsensusRanker(stability_top_n=1).rank(df)
    assert res.n_stable == 1                        # only 'a' is top-1 in both


def test_single_run_rank_reflects_first_replicate():
    # 'b' wins replicate 1 but loses on the mean -> exposes single-run instability
    df = _long({"a": [-9.0, -20.0], "b": [-10.0, -1.0]})
    res = ConsensusRanker().rank(df)
    row_a = res.ranking[res.ranking.Compound_ID == "a"].iloc[0]
    row_b = res.ranking[res.ranking.Compound_ID == "b"].iloc[0]
    assert row_a.consensus_rank == 1 and row_a.single_run_rank == 2
    assert row_b.consensus_rank == 2 and row_b.single_run_rank == 1


def test_spread_metric():
    df = _long({"a": [-10.0, -14.0], "b": [-5.0, -5.0]})
    res = ConsensusRanker().rank(df)
    assert res.mean_spread == pytest.approx(2.0)    # (4 + 0) / 2


def test_empty_raises():
    with pytest.raises(ValueError):
        ConsensusRanker().rank(pd.DataFrame(columns=["id", "replicate", "dG"]))


def test_replicates_wide_table_shape():
    df = _long({"a": [-1.0, -2.0, -3.0], "b": [-4.0, -5.0, -6.0]})
    res = ConsensusRanker().rank(df)
    assert len(res.replicates) == 2
    assert sum(c.startswith("replicate_") for c in res.replicates.columns) == 3
