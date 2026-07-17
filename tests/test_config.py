"""Tests for PipelineConfig loading, validation, and dotted overrides."""
import json

import pytest

from src.config import ConfigError, PipelineConfig


def test_loads_valid_config(config_file):
    cfg = PipelineConfig(str(config_file))
    assert cfg.get("target.pdb_id") == "1HXB"
    assert cfg.get("docking.runs_per_ligand") == 100


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        PipelineConfig(str(tmp_path / "nope.json"))


def test_missing_required_key_raises(tmp_path, valid_config_dict):
    del valid_config_dict["target"]["pdb_id"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(valid_config_dict))
    with pytest.raises(ConfigError, match="pdb_id"):
        PipelineConfig(str(p))


def test_invalid_search_mode_raises(tmp_path, valid_config_dict):
    valid_config_dict["target"]["search_mode"] = "TELEPORT"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(valid_config_dict))
    with pytest.raises(ConfigError, match="search_mode"):
        PipelineConfig(str(p))


def test_match_inbuilt_requires_native_ligand(tmp_path, valid_config_dict):
    valid_config_dict["target"]["native_ligand_name"] = None
    p = tmp_path / "c.json"
    p.write_text(json.dumps(valid_config_dict))
    with pytest.raises(ConfigError, match="native_ligand_name"):
        PipelineConfig(str(p))


def test_cli_override_and_coercion(config_file):
    cfg = PipelineConfig(str(config_file),
                         overrides={"target.pdb_id": "1W82", "ligands.subset_size": "40"})
    assert cfg.get("target.pdb_id") == "1W82"
    assert cfg.get("ligands.subset_size") == 40           # coerced str -> int


def test_override_bool_and_null_coercion(config_file):
    cfg = PipelineConfig(str(config_file),
                         overrides={"output.generate_figures": "false",
                                    "ligands.active_decoy_ratio": "null"})
    assert cfg.get("output.generate_figures") is False
    assert cfg.get("ligands.active_decoy_ratio") is None


def test_get_default_for_missing(config_file):
    cfg = PipelineConfig(str(config_file))
    assert cfg.get("nonexistent.key", "fallback") == "fallback"
