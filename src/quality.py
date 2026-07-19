"""Ligand quality gates applied to every ``.pdbqt`` before docking.

Gates are target-agnostic and operate purely on the ligand file:
  * torsional integrity - warn on frozen/inactive torsion branches
  * electrostatic capping - proportionally scale so max|q| <= cap (net-charge shape preserved)
  * conformer validation - radius-of-gyration sanity vs an optional native reference
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("docksuite.quality")


def radius_of_gyration(pdbqt: Path) -> Optional[float]:
    """Heavy-atom radius of gyration of a ``.pdbqt`` conformer (Angstrom)."""
    pts = []
    for line in pdbqt.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            typ = line[76:].strip().split()[0] if line[76:].strip() else ""
            if typ in ("H", "HD", "HS"):
                continue
            pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    return math.sqrt(sum((x-cx)**2 + (y-cy)**2 + (z-cz)**2 for x, y, z in pts) / len(pts))


class QualityGates:
    """Apply in-place quality gates to ligand ``.pdbqt`` files.

    Args:
        charge_cap: Maximum permitted absolute partial charge (e).
        rg_factor: Flag conformers with Rg above ``rg_factor * native_rg``.
        native_rg: Radius of gyration of the native ligand, or ``None`` to skip
            the shape gate (e.g. in BLIND_DOCK mode).
    """

    def __init__(self, charge_cap: float = 1.0, rg_factor: float = 1.5,
                 native_rg: Optional[float] = None, enable_shape_gate: bool = False) -> None:
        self.charge_cap = charge_cap
        self.rg_factor = rg_factor
        self.native_rg = native_rg
        # The published protocol uses only the torsional + electrostatic gates.
        # Conformer-shape filtering is available but OFF by default.
        self.enable_shape_gate = enable_shape_gate

    def process(self, ligand_pdbqt: Path) -> Path:
        """Run the mandatory gates against ``ligand_pdbqt`` (modified in place).

        Applies Gate 1 (torsional integrity) and Gate 2 (electrostatic capping).
        The optional conformer-shape gate runs only when ``enable_shape_gate``.

        Args:
            ligand_pdbqt: Path to a prepared ligand ``.pdbqt``.

        Returns:
            The same path, after gating.
        """
        self.gate_torsional_integrity(ligand_pdbqt)
        self.gate_charge_cap(ligand_pdbqt)
        if self.enable_shape_gate:
            self.gate_conformer_shape(ligand_pdbqt)
        return ligand_pdbqt

    # ------------------------------------------------------------------ #
    @staticmethod
    def gate_torsional_integrity(pdbqt: Path) -> int:
        """Warn only on *genuinely* frozen torsions; return active-torsion count.

        Detects real inactive torsions via their status line
        ``REMARK  <n>  I  between atoms: ...`` rather than the legend text
        ``('A' for Active; 'I' for Inactive)``, which is always present and must
        not be treated as a frozen bond.
        """
        text = pdbqt.read_text(errors="ignore")
        frozen = re.findall(r"^REMARK\s+\d+\s+I\s+between atoms", text, re.M)
        active = re.findall(r"^REMARK\s+\d+\s+A\s+between atoms", text, re.M)
        if frozen:
            logger.warning("%s: %d frozen torsion(s) present (expected all-active)",
                           pdbqt.name, len(frozen))
        return len(active)

    def gate_charge_cap(self, pdbqt: Path) -> int:
        """Proportionally scale partial charges so max|q| <= ``charge_cap``.

        Returns:
            Number of atom charges rescaled (0 if already within the cap).
        """
        lines = pdbqt.read_text(errors="ignore").splitlines()
        charges = []
        for l in lines:
            if l.startswith(("ATOM", "HETATM")) and len(l) >= 76 and l[66:76].strip():
                try:
                    charges.append(abs(float(l[66:76])))
                except ValueError:
                    pass
        mx = max(charges) if charges else 0.0
        if mx <= self.charge_cap:
            return 0
        factor = self.charge_cap / mx
        out, n = [], 0
        for l in lines:
            if l.startswith(("ATOM", "HETATM")) and len(l) >= 76:
                try:
                    q = float(l[66:76]) * factor
                    l = l[:66] + f"{q:+.3f}".rjust(10) + l[76:]
                    n += 1
                except ValueError:
                    pass
            out.append(l)
        pdbqt.write_text("\n".join(out) + "\n")
        logger.info("%s: capped %d charges (max|q| %.2f -> %.2f)", pdbqt.name, n, mx, self.charge_cap)
        return n

    def gate_conformer_shape(self, pdbqt: Path) -> tuple[Optional[float], bool]:
        """Flag conformers markedly more extended than the native reference.

        Returns:
            ``(Rg, flagged)`` where ``flagged`` is True if Rg exceeds the limit.
        """
        rg = radius_of_gyration(pdbqt)
        flagged = bool(self.native_rg and rg and rg > self.rg_factor * self.native_rg)
        if flagged:
            logger.warning("%s: extended conformer (Rg %.2f > %.2f x native %.2f)",
                           pdbqt.name, rg, self.rg_factor, self.native_rg)
        return rg, flagged
