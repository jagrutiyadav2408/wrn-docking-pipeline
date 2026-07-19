"""Tests for QualityGates (torsional, electrostatic, optional shape)."""
import pytest

from src.quality import QualityGates


def _pdbqt(tmp_path, charges, remark_lines=""):
    lines = [remark_lines, "ROOT"] if remark_lines else ["ROOT"]
    for i, q in enumerate(charges):
        lines.append(f"ATOM  {i+1:>5}  C   LIG A   1    "
                     f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00    {q:+.3f} C")
    lines += ["ENDROOT", "TORSDOF 0"]
    p = tmp_path / "lig.pdbqt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_electrostatic_cap_scales_over_limit(tmp_path):
    p = _pdbqt(tmp_path, [2.0, -1.0, 0.5])
    n = QualityGates(charge_cap=1.0).gate_charge_cap(p)
    assert n == 3
    vals = [float(l[66:76]) for l in p.read_text().splitlines() if l.startswith("ATOM")]
    assert max(abs(v) for v in vals) <= 1.0001
    assert vals[0] == pytest.approx(1.0, abs=1e-3)          # 2.0 * (1/2)


def test_electrostatic_cap_noop_under_limit(tmp_path):
    p = _pdbqt(tmp_path, [0.4, -0.3, 0.1])
    assert QualityGates(charge_cap=1.0).gate_charge_cap(p) == 0


def test_torsion_gate_ignores_legend_false_positive(tmp_path):
    # Open Babel always writes this legend; it must NOT be read as a frozen bond
    legend = ("REMARK  2 active torsions:\n"
              "REMARK  status: ('A' for Active; 'I' for Inactive)\n"
              "REMARK    1  A    between atoms: C_1  and  C_2\n"
              "REMARK    2  A    between atoms: C_2  and  C_3")
    p = _pdbqt(tmp_path, [0.1], legend)
    # all-active -> returns the active-torsion count, no frozen warning raised
    assert QualityGates.gate_torsional_integrity(p) == 2


def test_torsion_gate_detects_real_frozen(tmp_path):
    txt = ("REMARK  status: ('A' for Active; 'I' for Inactive)\n"
           "REMARK    1  A    between atoms: C_1  and  C_2\n"
           "REMARK    2  I    between atoms: C_2  and  C_3\nROOT\nENDROOT\n")
    p = tmp_path / "f.pdbqt"; p.write_text(txt)
    # one genuine inactive torsion present; active count is 1
    assert QualityGates.gate_torsional_integrity(p) == 1


def test_shape_gate_off_by_default(tmp_path):
    # process() with default settings must NOT run the shape gate (manuscript protocol)
    calls = {"n": 0}
    g = QualityGates(native_rg=3.0)
    assert g.enable_shape_gate is False
    g.gate_conformer_shape = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
    g.process(_pdbqt(tmp_path, [0.5, -0.5]))
    assert calls["n"] == 0


def test_shape_gate_runs_when_enabled(tmp_path):
    calls = {"n": 0}
    g = QualityGates(native_rg=3.0, enable_shape_gate=True)
    g.gate_conformer_shape = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
    g.process(_pdbqt(tmp_path, [0.5, -0.5]))
    assert calls["n"] == 1
