"""Parse AutoDock ``.dlg`` output and extract the representative binding pose."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("docksuite.poses")

# Cluster histogram row: rank | dG | #inclust | ... | run | ...
_CLUSTER_ROW = re.compile(
    r"^\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(\d+)\s*\|", re.M)


@dataclass
class Pose:
    """A single extracted docking pose.

    Attributes:
        ligand: Ligand/output name.
        delta_g: Binding free energy of the pose (kcal/mol).
        cluster_size: Number of runs in the selected cluster.
        run: GA run number that produced the representative pose.
        atom_lines: Raw PDBQT ATOM/HETATM lines of the pose (may be empty).
    """
    ligand: str
    delta_g: float
    cluster_size: int
    run: int
    atom_lines: list[str] = field(default_factory=list)

    def to_pdb(self, out_path: Path) -> Path:
        """Write the pose as a minimal PDB (Discovery-Studio friendly).

        Creates the parent directory if needed.

        Args:
            out_path: Destination ``.pdb`` path.

        Returns:
            The written path.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        body = []
        for l in self.atom_lines:
            rec = "ATOM  " + l[6:] if l.startswith("HETATM") else l
            body.append(rec[:66].rstrip())
        logger.info("Saving top1 pose to %s", out_path)
        out_path.write_text("\n".join(body) + "\nEND\n")
        return out_path


class PoseExtractor:
    """Extract the top pose from the largest cluster of a ``.dlg`` file.

    Args:
        selection: ``"largest_cluster"`` (most populated, then lowest energy) or
            ``"lowest_energy"`` (global minimum regardless of cluster size).
    """

    def __init__(self, selection: str = "largest_cluster") -> None:
        if selection not in ("largest_cluster", "lowest_energy"):
            raise ValueError(f"unknown selection: {selection!r}")
        self.selection = selection

    def extract(self, dlg_path: Path) -> Optional[Pose]:
        """Parse ``dlg_path`` and return the representative :class:`Pose`.

        Args:
            dlg_path: Path to an AutoDock ``.dlg`` file.

        Returns:
            The chosen pose, or ``None`` if the file has no cluster table.
        """
        if not dlg_path.is_file():
            return None
        text = dlg_path.read_text(errors="ignore")
        rows = [(int(m[1]), float(m[2]), int(m[3]), int(m[5]))
                for m in _CLUSTER_ROW.finditer(text)]           # (rank, dG, size, run)
        if not rows:
            logger.warning("%s: no cluster histogram found", dlg_path.name)
            return None
        if self.selection == "largest_cluster":
            rank, dg, size, run = max(rows, key=lambda r: (r[2], -r[1]))
        else:
            rank, dg, size, run = min(rows, key=lambda r: r[1])
        atoms = self._atoms_for_run(text, run)
        return Pose(ligand=dlg_path.stem, delta_g=dg, cluster_size=size, run=run, atom_lines=atoms)

    @staticmethod
    def _atoms_for_run(text: str, run_no: int) -> list[str]:
        """Return the PDBQT atom lines of the docked model for ``run_no``."""
        by_run: dict[int, list[str]] = {}
        cur: list[str] = []
        cur_run: Optional[int] = None
        for line in text.splitlines():
            if not line.startswith("DOCKED:"):
                continue
            body = line[8:]
            if body.strip().startswith("MODEL"):
                cur, cur_run = [], None
            elif "Run =" in body and (m := re.search(r"Run\s*=\s*(\d+)", body)):
                cur_run = int(m.group(1))
            elif body[:6].strip() in ("ATOM", "HETATM"):
                cur.append(body.rstrip())
            elif body.strip() == "ENDMDL" and cur_run is not None:
                by_run[cur_run] = cur
        return by_run.get(run_no, [])
