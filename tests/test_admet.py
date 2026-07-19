"""Tests for ADMETProfiler physchem, SMILES repair, and risk tiering.

ADMET-AI is not invoked here (use_admet_ai=False) so these run without the model
or a network; they exercise the RDKit descriptors and the tier logic.
"""
import pandas as pd
import pytest

from src.admet import AdmetProfiler


def test_sanitize_nitro_smiles():
    # Open Babel writes neutral 5-valent nitro that RDKit rejects
    fixed = AdmetProfiler._sanitize_smiles("c1ccccc1[N](=O)[O-]")
    assert "[N+](=O)[O-]" in fixed
    from rdkit import Chem
    assert Chem.MolFromSmiles(fixed) is not None


def test_physchem_marks_parse_ok():
    df = AdmetProfiler._physchem(["CCO", "not_a_smiles_$$$"])
    assert bool(df.iloc[0]["parse_ok"]) is True
    assert "MW" in df.columns and df.iloc[0]["MW"] > 0
    assert bool(df.iloc[1]["parse_ok"]) is False         # unparseable row flagged


def test_unparseable_never_passes(monkeypatch):
    # a molecule RDKit cannot parse must be tiered UNPARSED, not PASS
    recs = pd.DataFrame({"id": ["good", "bad"], "dG": [-9.0, -8.0],
                         "smiles": ["CCO", "%%%broken%%%"]})
    prof = AdmetProfiler(mode="ALL", use_admet_ai=False).profile(recs)
    tiers = dict(zip(prof["id"], prof["Risk_Tier"]))
    assert tiers["bad"] == "UNPARSED"
    assert tiers["good"] != "UNPARSED"


def test_saturated_endpoint_detection():
    # an endpoint uniformly high across the set carries no per-compound signal
    out = pd.DataFrame({"DILI": [0.99, 1.00, 0.98, 0.99, 1.00],
                        "hERG": [0.30, 0.55, 0.70, 0.45, 0.60]})
    sat = AdmetProfiler._saturated_endpoints(out)
    assert "DILI" in sat                                 # flagged (no spread, high)
    assert "hERG" not in sat                             # varies -> discriminating


def test_classify_excludes_high_herg_but_ignores_saturated():
    row = pd.Series({"PAINS_alert": 0, "hERG": 0.8, "DILI": 1.0,
                     "Lipinski_violations": 1, "HIA_Hou": 0.9, "parse_ok": True, "MW": 400})
    # hERG >= 0.7 -> EXCLUDE
    assert AdmetProfiler._classify(row, frozenset()) == "EXCLUDE"
    # if hERG were low and only DILI high, and DILI is saturated (ignored) -> not EXCLUDE
    row2 = pd.Series({"PAINS_alert": 0, "hERG": 0.4, "DILI": 1.0,
                      "Lipinski_violations": 1, "HIA_Hou": 0.9, "parse_ok": True, "MW": 400})
    assert AdmetProfiler._classify(row2, frozenset({"DILI"})) in ("MODERATE", "PASS", "HIGH RISK")
    assert AdmetProfiler._classify(row2, frozenset({"DILI"})) != "EXCLUDE"


def test_mode_none_returns_empty():
    recs = pd.DataFrame({"id": ["a"], "dG": [-9.0], "smiles": ["CCO"]})
    assert AdmetProfiler(mode="NONE").profile(recs).empty
