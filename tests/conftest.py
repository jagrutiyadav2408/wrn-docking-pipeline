"""Shared pytest fixtures. No external binaries or network are required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def valid_config_dict() -> dict:
    """A minimal, schema-valid configuration dictionary."""
    return {
        "target": {"pdb_id": "1HXB", "native_ligand_name": "ROC",
                   "chains_to_keep": ["A", "B"], "search_mode": "MATCH_INBUILT",
                   "box_buffer_angstroms": 4.0, "grid_spacing": 0.375},
        "ligands": {"source": "DUDE", "target_name": "hivpr", "subset_size": 250,
                    "active_decoy_ratio": None, "stratify_by": "molecular_weight",
                    "random_seed": 42},
        "docking": {"backend": "AUTODOCK", "runs_per_ligand": 100, "heuristics": True,
                    "autostop": True, "cluster_rmsd_threshold": 2.0},
        "admet": {"mode": "NONE", "top_n": None},
        "output": {"benchmark_dir": "./bench", "output_dir": "./out", "generate_figures": True},
        "hardware": {"gpu_device": 0, "n_workers": 4},
    }


@pytest.fixture
def config_file(tmp_path, valid_config_dict) -> Path:
    """Write ``valid_config_dict`` to a temp JSON file and return its path."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(valid_config_dict))
    return p


def _pdb_line(rec, serial, name, alt, res, chain, resseq, x, y, z, elem):
    """Build a column-exact PDB ATOM/HETATM record."""
    return (f"{rec:<6}{serial:>5} {name:<4}{alt:1}{res:>3} {chain:1}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}")


@pytest.fixture
def mini_pdb(tmp_path) -> Path:
    """A tiny two-chain PDB with a dual-altloc ligand and a water (exact columns)."""
    lines = [
        _pdb_line("ATOM", 1, "CA", " ", "ALA", "A", 1, 10.0, 10.0, 10.0, "C"),
        _pdb_line("ATOM", 2, "CA", " ", "ALA", "A", 2, 12.0, 10.0, 10.0, "C"),
        _pdb_line("ATOM", 3, "CA", " ", "GLY", "B", 1, 11.0, 12.0, 10.0, "C"),
        _pdb_line("HETATM", 4, "C1", "A", "LIG", "A", 900, 11.0, 11.0, 11.0, "C"),
        _pdb_line("HETATM", 5, "C2", "A", "LIG", "A", 900, 12.0, 11.0, 11.0, "C"),
        _pdb_line("HETATM", 6, "C1", "B", "LIG", "A", 900, 50.0, 50.0, 50.0, "C"),
        _pdb_line("HETATM", 7, "O", " ", "HOH", "A", 950, 99.0, 99.0, 99.0, "O"),
        "END",
    ]
    p = tmp_path / "MINI.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p
