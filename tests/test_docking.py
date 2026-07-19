"""Tests for DockingEngine / AutoDockBackend (binary mocked, no GPU)."""
import json

import pytest

import src.docking as dmod
from src.config import PipelineConfig
from src.docking import AutoDockBackend


def _cfg(tmp_path, cfg_dict, **docking_over):
    cfg_dict["docking"].update(docking_over)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg_dict))
    return PipelineConfig(str(p))


def test_gpu_device_index_conversion(tmp_path, valid_config_dict):
    # config gpu_device is 0-indexed; AutoDock-GPU --devnum must be 1-indexed
    b = AutoDockBackend(_cfg(tmp_path, valid_config_dict, gpu_device=0))
    assert b.device == "1"
    b2 = AutoDockBackend(_cfg(tmp_path, valid_config_dict, gpu_device=2))
    assert b2.device == "3"


def test_deterministic_disables_autostop(tmp_path, valid_config_dict):
    # deterministic=true must force --autostop 0 even if autostop=true in config
    b = AutoDockBackend(_cfg(tmp_path, valid_config_dict, autostop=True, deterministic=True))
    assert b.autostop == 0
    b2 = AutoDockBackend(_cfg(tmp_path, valid_config_dict, autostop=True, deterministic=False))
    assert b2.autostop == 1


def test_seed_falls_back_to_random_seed(tmp_path, valid_config_dict):
    b = AutoDockBackend(_cfg(tmp_path, valid_config_dict))       # ligands.random_seed == 42
    assert b.seed == 42
    b2 = AutoDockBackend(_cfg(tmp_path, valid_config_dict, seed=7))
    assert b2.seed == 7


def test_dock_command_has_triple_seed_and_devnum(tmp_path, valid_config_dict, monkeypatch):
    b = AutoDockBackend(_cfg(tmp_path, valid_config_dict, gpu_device=0, seed=42))
    b._bin = "autodock_gpu"
    captured = {}

    def fake_run(cmd, cwd=None, check=True, **kw):
        captured["cmd"] = [str(c) for c in cmd]
        (cwd / "L__s42.dlg").write_text("CLUSTERING HISTOGRAM\n 1 | -1.0 | 1 | -1.0 | 1 |\n")
        class _P:
            returncode, stdout, stderr = 0, "", ""
        return _P()

    monkeypatch.setattr(dmod, "run", fake_run)
    lig = tmp_path / "L.pdbqt"; lig.write_text("ROOT\nENDROOT\nTORSDOF 0\n")
    b.dock(lig, tmp_path / "receptor.maps.fld", "L__s42", tmp_path, seed=42)
    s = " ".join(captured["cmd"])
    assert "--seed" in s and "42,42,42" in captured["cmd"]      # all three RNGs pinned
    assert "--devnum" in s and "1" in captured["cmd"]
    assert "--autostop" in s and "0" in captured["cmd"]         # deterministic default


def test_dock_idempotent_skips_completed(tmp_path, valid_config_dict, monkeypatch):
    b = AutoDockBackend(_cfg(tmp_path, valid_config_dict)); b._bin = "autodock_gpu"
    dlg = tmp_path / "L__s42.dlg"
    dlg.write_text("CLUSTERING HISTOGRAM\n 1 | -1.0 | 1 | -1.0 | 1 |\n")   # already done
    called = {"n": 0}
    monkeypatch.setattr(dmod, "run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    lig = tmp_path / "L.pdbqt"; lig.write_text("ROOT\n")
    out = b.dock(lig, tmp_path / "m.fld", "L__s42", tmp_path, seed=42)
    assert out == dlg and called["n"] == 0                     # run() never invoked
