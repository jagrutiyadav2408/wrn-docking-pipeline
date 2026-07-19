"""Grid-box calculation and AutoGrid map generation."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .util import resolve_tool, run

logger = logging.getLogger("docksuite.grid")

#: AutoGrid hard limit on points per axis.
NPTS_CAP = 126


@dataclass
class GridBox:
    """A cubic docking search box.

    Attributes:
        center: (x, y, z) grid centre in Angstrom.
        npts: Points per axis (same for x/y/z; cubic box).
        spacing: Grid spacing in Angstrom.
    """
    center: tuple[float, float, float]
    npts: int
    spacing: float

    @property
    def size(self) -> float:
        """Edge length of the box in Angstrom."""
        return self.npts * self.spacing


def _heavy_coords(lines: Sequence[str]) -> list[tuple[float, float, float]]:
    pts = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 54:
            typ = l[76:].strip().split()[0] if l[76:].strip() else ""
            if typ == "H":
                continue
            pts.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return pts


class GridEngine:
    """Compute grid boxes and generate AutoGrid maps for any target/mode.

    Args:
        spacing: Grid spacing (Angstrom).
        obabel: Optional explicit path to Open Babel.
        autogrid: Optional explicit path to ``autogrid4``.
    """

    def __init__(self, spacing: float = 0.375, obabel: str | None = None,
                 autogrid: str | None = None) -> None:
        self.spacing = spacing
        self._obabel = resolve_tool("obabel", obabel)
        self._autogrid = resolve_tool("autogrid4", autogrid)

    # ------------------------------------------------------------------ #
    def calculate_box(self, target, mode: str, buffer: float) -> GridBox:
        """Compute the search box for ``target`` in the given ``mode``.

        Args:
            target: A :class:`~vspipeline.target.PreparedTarget`.
            mode: ``"MATCH_INBUILT"`` (native-ligand-centred) or
                ``"BLIND_DOCK"`` (whole Ca backbone, capped at 126 pts).
            buffer: Safety buffer added on each side (Angstrom).

        Returns:
            The computed :class:`GridBox`.

        Raises:
            ValueError: On unknown mode or missing reference atoms.
        """
        if mode == "MATCH_INBUILT":
            pts = _heavy_coords(target.native_ligand_pdb.read_text(errors="ignore").splitlines())
            if not pts:
                raise ValueError("MATCH_INBUILT: no native-ligand heavy atoms found")
        elif mode == "BLIND_DOCK":
            pts = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                   for l in target.receptor_source_lines
                   if l.startswith("ATOM") and l[12:16].strip() == "CA"]
            if not pts:
                raise ValueError("BLIND_DOCK: no Ca atoms found")
        else:
            raise ValueError(f"unknown search mode: {mode!r}")

        xs, ys, zs = zip(*pts)
        if mode == "MATCH_INBUILT":
            center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
        else:  # blind: geometric centre of the bounding box
            center = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)
        span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) + 2*buffer
        npts = int(math.ceil(span / self.spacing))
        npts += npts % 2
        capped = min(npts, NPTS_CAP)
        if capped < npts:
            logger.warning("box %d pts exceeds AutoGrid cap; capped to %d (%.1f A)",
                           npts, capped, capped*self.spacing)
        box = GridBox(center=center, npts=capped, spacing=self.spacing)
        logger.info("%s box: centre (%.2f, %.2f, %.2f) size %.1f A (%d pts)",
                    mode, *center, box.size, box.npts)
        return box

    def generate_maps(self, receptor_pdbqt: Path, box: GridBox,
                      ligand_types: Sequence[str], work_dir: Path) -> Path:
        """Write a GPF and run AutoGrid, producing ``receptor.maps.fld``.

        Args:
            receptor_pdbqt: Prepared receptor.
            box: The search box.
            ligand_types: Union of AutoDock atom types across the library.
            work_dir: Directory to run AutoGrid in (maps written here).

        Returns:
            Path to the generated ``.fld`` grid field file.

        Raises:
            RuntimeError: If AutoGrid is unavailable or fails.
        """
        if not self._autogrid:
            raise RuntimeError("autogrid4 not found; set docking tool paths in config")
        rec_types = self._atom_types(receptor_pdbqt)
        maps = "\n".join(f"map receptor.{t}.map" for t in ligand_types)
        gpf = work_dir / "receptor.gpf"
        gpf.write_text(
            f"npts {box.npts} {box.npts} {box.npts}\ngridfld receptor.maps.fld\n"
            f"spacing {box.spacing}\nreceptor_types {' '.join(rec_types)}\n"
            f"ligand_types {' '.join(ligand_types)}\nreceptor {receptor_pdbqt.name}\n"
            f"gridcenter {box.center[0]:.3f} {box.center[1]:.3f} {box.center[2]:.3f}\n"
            f"smooth 0.5\n{maps}\nelecmap receptor.e.map\ndsolvmap receptor.d.map\n"
            f"dielectric -0.1465\n")
        logger.info("running autogrid4 (%d pts)...", box.npts)
        run([self._autogrid, "-p", "receptor.gpf", "-l", "receptor.glg"], cwd=work_dir)
        fld = work_dir / "receptor.maps.fld"
        if not fld.is_file():
            raise RuntimeError("autogrid4 completed but receptor.maps.fld is missing")
        return fld

    @staticmethod
    def _atom_types(pdbqt: Path) -> list[str]:
        seen: list[str] = []
        for l in pdbqt.read_text(errors="ignore").splitlines():
            if l.startswith(("ATOM", "HETATM")) and len(l) > 77:
                t = l[77:].strip().split()[0]
                if t and t not in seen:
                    seen.append(t)
        return seen
