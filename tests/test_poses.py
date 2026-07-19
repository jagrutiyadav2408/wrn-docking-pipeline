"""Tests for PoseExtractor (lowest_energy vs largest_cluster) and Pose.to_pdb."""
import pytest

from src.poses import Pose, PoseExtractor

# Cluster histogram: rank | dG | size | mean | run |
#   run 5 = global-minimum energy (-10.50), small cluster (size 3)
#   run 2 = most populated cluster (size 10), shallower (-9.00)
DLG = """\
CLUSTERING HISTOGRAM
    1 |   -10.50 |     3 |   -10.40 |    5 |####
    2 |    -9.00 |    10 |    -8.90 |    2 |##########
DOCKED: MODEL        1
DOCKED: USER    Run = 5
DOCKED: ATOM      1  C   LIG A   1      1.000   1.000   1.000  1.00  0.00     0.00 C
DOCKED: ATOM      2  C   LIG A   1      2.000   1.000   1.000  1.00  0.00     0.00 C
DOCKED: ENDMDL
DOCKED: MODEL        2
DOCKED: USER    Run = 2
DOCKED: ATOM      1  C   LIG A   1      5.000   5.000   5.000  1.00  0.00     0.00 C
DOCKED: ENDMDL
"""


def _write(tmp_path):
    p = tmp_path / "lig.dlg"; p.write_text(DLG)
    return p


def test_lowest_energy_selects_global_minimum(tmp_path):
    pose = PoseExtractor("lowest_energy").extract(_write(tmp_path))
    assert pose.delta_g == pytest.approx(-10.50)
    assert pose.run == 5
    assert len(pose.atom_lines) == 2               # run-5 model has 2 atoms


def test_largest_cluster_selects_most_populated(tmp_path):
    pose = PoseExtractor("largest_cluster").extract(_write(tmp_path))
    assert pose.cluster_size == 10
    assert pose.delta_g == pytest.approx(-9.00)
    assert pose.run == 2


def test_lowest_energy_and_largest_cluster_differ_here(tmp_path):
    le = PoseExtractor("lowest_energy").extract(_write(tmp_path))
    lc = PoseExtractor("largest_cluster").extract(_write(tmp_path))
    assert le.run != lc.run                        # the whole point of the choice


def test_unknown_selection_raises():
    with pytest.raises(ValueError):
        PoseExtractor("teleport")


def test_extract_none_when_no_histogram(tmp_path):
    p = tmp_path / "empty.dlg"; p.write_text("no clusters here\n")
    assert PoseExtractor("lowest_energy").extract(p) is None


def test_pose_to_pdb_writes_atoms_and_creates_dir(tmp_path):
    pose = Pose(ligand="L", delta_g=-10.5, cluster_size=3, run=5,
                atom_lines=["ATOM      1  C   LIG A   1      1.000   1.000   1.000  1.00  0.00     0.00 C"])
    out = tmp_path / "sub" / "L_top1.pdb"
    pose.to_pdb(out)
    assert out.is_file()
    txt = out.read_text()
    assert txt.startswith("ATOM") and txt.strip().endswith("END")
