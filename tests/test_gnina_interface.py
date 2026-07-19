#!/usr/bin/env python3
"""
Unit tests for gnina_interface. These run WITHOUT a gnina binary or GPU: they
exercise config parsing, command construction, quality gates, output parsing, and
the backend comparison on synthetic fixtures.

    pytest tests/test_gnina_interface.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gnina_interface import (GninaConfig, GninaInterface, gate_box_validation,
                                       gate_charge_cap, gate_torsional_integrity,
                                       radius_of_gyration)

BASE_CFG = {
    "GNINA_BINARY": "gnina", "GNINA_MODE": "dock", "GNINA_EXHAUSTIVENESS": 32,
    "GNINA_NUM_MODES": 9, "GNINA_ENERGY_RANGE": 3.0, "GNINA_SEED": 42,
    "GNINA_MINIMIZE": True, "GNINA_CNN_MODEL": "default", "GNINA_DEVICE": "0",
}


# ---- config ------------------------------------------------------------ #
def test_config_defaults():
    c = GninaConfig.from_dict({})
    assert c.exhaustiveness == 32 and c.num_modes == 9 and c.seed == 42
    assert c.cnn_model == "default" and c.cnn_scoring == "rescore"


def test_config_bad_model_falls_back():
    c = GninaConfig.from_dict({"GNINA_CNN_MODEL": "not_a_model"})
    assert c.cnn_model == "default"


def test_config_bad_cnn_scoring_falls_back():
    c = GninaConfig.from_dict({"GNINA_CNN_SCORING": "bogus"})
    assert c.cnn_scoring == "rescore"


# ---- command construction --------------------------------------------- #
def test_dock_command_has_box_and_seed():
    gi = GninaInterface(BASE_CFG)
    cmd = gi.prepare_gnina_command("rec.pdbqt", "lig.pdbqt", "out.pdbqt",
                                   (1.0, 2.0, 3.0), (20.0, 20.0, 20.0), "dock", "out.log")
    s = " ".join(cmd)
    assert "-r" in cmd and "-l" in cmd and "-o" in cmd
    assert "--center_x" in s and "1.000" in s
    assert "--size_x" in s and "20.0" in s
    assert "--exhaustiveness" in s and "32" in cmd
    assert "--seed" in s and "42" in cmd
    assert "--cnn_scoring" in s


def test_rescore_command_is_score_only():
    gi = GninaInterface(BASE_CFG)
    cmd = gi.prepare_gnina_command("rec.pdbqt", "pose.pdbqt", "", (0, 0, 0), (0, 0, 0), "rescore")
    assert "--score_only" in cmd
    assert "--center_x" not in " ".join(cmd)


def test_minimize_command_is_local_only():
    gi = GninaInterface(BASE_CFG)
    cmd = gi.prepare_gnina_command("rec.pdbqt", "pose.pdbqt", "o.pdbqt", (0, 0, 0), (0, 0, 0), "minimize")
    assert "--local_only" in cmd


def test_bad_mode_raises():
    gi = GninaInterface(BASE_CFG)
    with pytest.raises(ValueError):
        gi.prepare_gnina_command("r", "l", "o", (0, 0, 0), (1, 1, 1), "teleport")


def test_wsl_launcher_translates_paths():
    gi = GninaInterface({**BASE_CFG, "GNINA_LAUNCHER": "wsl"})
    cmd = gi.prepare_gnina_command(r"C:\data\rec.pdbqt", r"C:\data\lig.pdbqt",
                                   r"C:\data\out.pdbqt", (0, 0, 0), (10, 10, 10), "dock")
    assert cmd[0] == "wsl.exe"
    assert any(a.startswith("/mnt/c/") for a in cmd)


# ---- quality gates ----------------------------------------------------- #
def _write_pdbqt(tmp_path, charges):
    lines = ["ROOT"]
    for i, q in enumerate(charges):
        lines.append(f"ATOM  {i+1:>5}  C   LIG A   1    "
                     f"{i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00    {q:+.3f} C")
    lines += ["ENDROOT", "TORSDOF 0"]
    p = tmp_path / "lig.pdbqt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_charge_cap_scales_when_over(tmp_path):
    p = _write_pdbqt(tmp_path, [2.0, -1.0, 0.5])
    n = gate_charge_cap(p, cap=1.0)
    assert n == 3
    vals = [float(l[66:76]) for l in p.read_text().splitlines() if l.startswith("ATOM")]
    assert max(abs(v) for v in vals) <= 1.0001
    assert vals[0] == pytest.approx(1.0, abs=1e-3)      # 2.0 * (1/2)


def test_charge_cap_noop_when_under(tmp_path):
    p = _write_pdbqt(tmp_path, [0.4, -0.3, 0.1])
    assert gate_charge_cap(p, cap=1.0) == 0


def test_torsional_integrity_counts_branches(tmp_path):
    p = tmp_path / "t.pdbqt"
    p.write_text("ROOT\nATOM 1\nENDROOT\nBRANCH 1 2\nATOM 2\nENDBRANCH 1 2\nTORSDOF 1\n")
    assert gate_torsional_integrity(p) >= 1


def test_box_validation_inside_and_outside():
    pts = [(0, 0, 0), (1, 1, 1), (-1, -1, -1)]
    assert gate_box_validation((0, 0, 0), (10, 10, 10), pts, 1.0) is True
    assert gate_box_validation((0, 0, 0), (2, 2, 2), pts + [(50, 0, 0)], 1.0) is False


def test_radius_of_gyration(tmp_path):
    p = _write_pdbqt(tmp_path, [0.0, 0.0, 0.0])   # atoms at x=0,1,2
    rg = radius_of_gyration(p)
    assert rg == pytest.approx(0.8165, abs=1e-3)  # sqrt(mean of (1,0,1))


# ---- output parsing ---------------------------------------------------- #
def test_parse_score_stream_table():
    txt = ("mode |  affinity | CNN pose | CNN affinity\n"
           "-----+-----------+----------+-------------\n"
           "    1     -7.20      0.8534       5.123\n"
           "    2     -6.10      0.7000       4.500\n")
    v = GninaInterface._parse_score_stream(txt)
    assert v["affinity"] == pytest.approx(-7.20)
    assert v["CNNscore"] == pytest.approx(0.8534)
    assert v["CNNaffinity"] == pytest.approx(5.123)


def test_parse_gnina_output_pdbqt(tmp_path):
    out = tmp_path / "o_gnina.pdbqt"
    out.write_text(
        "MODEL 1\nREMARK minimizedAffinity -7.2\nREMARK CNNscore 0.85\n"
        "REMARK CNNaffinity 5.1\nATOM 1\nENDMDL\n"
        "MODEL 2\nREMARK minimizedAffinity -6.0\nREMARK CNNscore 0.70\n"
        "REMARK CNNaffinity 4.3\nATOM 1\nENDMDL\n")
    gi = GninaInterface(BASE_CFG)
    df = gi.parse_gnina_output(str(out))
    assert len(df) == 2
    # sorted by CNNaffinity descending -> pose with 5.1 first
    assert df.iloc[0]["CNNaffinity"] == pytest.approx(5.1)


# ---- comparison -------------------------------------------------------- #
def test_compare_with_autodock_overlap():
    ad = pd.DataFrame({"Compound_ID": list("ABCDE"),
                       "dG": [-10, -9, -8, -7, -6]})
    gn = pd.DataFrame({"Compound_ID": list("ABCDE"),
                       "CNNaffinity": [5.0, 4.0, 3.0, 2.0, 1.0]})
    gi = GninaInterface(BASE_CFG)
    m = gi.compare_with_autodock(ad, gn)
    assert len(m) == 5
    assert m.attrs["spearman"] == pytest.approx(1.0)   # perfectly concordant
    assert m.attrs["top_n_overlap"] == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
