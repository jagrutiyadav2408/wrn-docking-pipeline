"""Tests for TargetPreparator cleaning/altloc/chain logic (binaries mocked)."""
import json

import pytest

import src.target as tmod
from src.config import PipelineConfig
from src.target import TargetPreparator


def _make_config(tmp_path, cfg_dict, native="LIG"):
    cfg_dict["target"]["pdb_id"] = "MINI"
    cfg_dict["target"]["native_ligand_name"] = native
    cfg_dict["target"]["search_mode"] = "MATCH_INBUILT"
    cfg_dict["ligands"]["source"] = "SMI"
    cfg_dict["ligands"].pop("target_name", None)
    cfg_dict["output"]["benchmark_dir"] = str(tmp_path / "bench")
    cfg_dict["output"]["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg_dict))
    return PipelineConfig(str(p))


@pytest.fixture
def mocked_preparator(monkeypatch, mini_pdb):
    """A TargetPreparator whose fetch returns mini_pdb and whose obabel is mocked."""
    prep = TargetPreparator.__new__(TargetPreparator)   # bypass tool resolution
    prep._obabel = "obabel"                             # truthy dummy

    def fake_fetch(pdb_id, out_dir):
        dest = out_dir / f"{pdb_id}.pdb"
        dest.write_text(mini_pdb.read_text())
        return dest

    def fake_run(cmd, cwd=None, check=True, **kw):
        args = [str(c) for c in cmd]
        out = args[args.index("-O") + 1]
        from pathlib import Path
        Path(out).write_text("REMARK mocked receptor\n")
        class _P:
            returncode, stdout, stderr = 0, "", ""
        return _P()

    monkeypatch.setattr(prep, "_fetch", fake_fetch)
    monkeypatch.setattr(tmod, "run", fake_run)
    return prep


def test_prepare_extracts_single_altloc_native(mocked_preparator, tmp_path, valid_config_dict):
    cfg = _make_config(tmp_path, valid_config_dict)
    target = mocked_preparator.prepare(cfg)
    native_txt = target.native_ligand_pdb.read_text()
    # LIG has altloc A (2 atoms) + altloc B (1 atom); only altloc A retained
    assert native_txt.count("HETATM") == 2
    assert "50.000" not in native_txt                  # the altloc-B atom is excluded


def test_prepare_keeps_requested_chains_and_drops_water(mocked_preparator, tmp_path, valid_config_dict):
    cfg = _make_config(tmp_path, valid_config_dict)
    target = mocked_preparator.prepare(cfg)
    assert target.chains == ["A", "B"]
    assert target.receptor_pdbqt.is_file()
    # water (HOH) must not appear in the cleaned receptor source
    assert all("HOH" not in l for l in target.receptor_source_lines)


def test_prepare_missing_native_raises(mocked_preparator, tmp_path, valid_config_dict):
    cfg = _make_config(tmp_path, valid_config_dict, native="ZZZ")
    with pytest.raises(ValueError, match="ZZZ"):
        mocked_preparator.prepare(cfg)
