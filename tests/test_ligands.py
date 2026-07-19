"""Tests for LigandLibrary loading, parsing, and subsetting (no binaries)."""
import pandas as pd
import pytest

from src.ligands import LigandLibrary


def test_read_ism_labels_and_columns(tmp_path):
    ism = tmp_path / "actives_final.ism"
    ism.write_text("CCO 111 CHEMBL1\nCCN 222 CHEMBL2\n")
    df = LigandLibrary._read_ism(ism, is_active=1)
    assert list(df.columns) == ["smiles", "id", "is_active"]
    assert df["is_active"].tolist() == [1, 1]
    assert df["id"].tolist() == ["active_111", "active_222"]


def test_load_smi_unlabelled(tmp_path):
    smi = tmp_path / "lib.smi"
    smi.write_text("CCO mol_a\nCCN mol_b\nc1ccccc1\n")
    df = LigandLibrary._load_smi(smi, labelled=False)
    assert len(df) == 3
    assert "is_active" not in df.columns
    assert df.iloc[2]["id"].startswith("lig")     # auto-named when no id column


def test_subset_preserves_actives_and_size():
    df = pd.DataFrame({
        "smiles": ["C"] * 24,
        "id": [f"active_{i}" for i in range(4)] + [f"decoy_{i}" for i in range(20)],
        "is_active": [1] * 4 + [0] * 20,
        "MW": list(range(300, 300 + 24)),
    })
    sub = LigandLibrary(seed=42).subset(df, n=12, stratify_by="molecular_weight", seed=42)
    assert len(sub) == 12
    assert int(sub["is_active"].sum()) == 3        # 12 * 0.25 default ratio
    assert (sub["is_active"] == 0).sum() == 9


def test_subset_unlabelled_random():
    df = pd.DataFrame({"smiles": ["C"] * 50, "id": [f"m{i}" for i in range(50)]})
    sub = LigandLibrary().subset(df, n=10, stratify_by=None, seed=1)
    assert len(sub) == 10


def test_subset_reproducible_with_seed():
    df = pd.DataFrame({"smiles": ["C"] * 50, "id": [f"m{i}" for i in range(50)]})
    a = LigandLibrary().subset(df, 10, None, seed=7)["id"].tolist()
    b = LigandLibrary().subset(df, 10, None, seed=7)["id"].tolist()
    assert a == b
