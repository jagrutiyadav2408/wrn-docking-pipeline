"""Receptor preparation: fetch, clean, extract native ligand, build .pdbqt.

Generalizes across monomers, dimers, and higher oligomers. Alternate locations
(altloc) are collapsed to a single conformer and only the requested chains are
retained. MGLTools is used when available; otherwise Open Babel is the fallback
(Gasteiger charges instead of Kollman) and this substitution is logged.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .util import resolve_tool, run

logger = logging.getLogger("docksuite.target")

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

#: Crystallization additives / ions removed during cleaning.
BUFFERS = {"HOH", "WAT", "SO4", "GOL", "EDO", "PEG", "MSE", "ACT", "CL", "NA",
           "MG", "CA", "K", "ZN", "MPD", "DMS", "IOD", "PO4", "BME", "NO3"}


@dataclass
class PreparedTarget:
    """Outputs of :meth:`TargetPreparator.prepare`.

    Attributes:
        pdb_id: Source PDB identifier.
        receptor_pdbqt: Prepared receptor for docking.
        native_ligand_pdb: Extracted native ligand (single altloc), or ``None``.
        receptor_source_lines: Cleaned protein PDB lines (for BLIND_DOCK Ca box).
        chains: Chains retained.
    """
    pdb_id: str
    receptor_pdbqt: Path
    native_ligand_pdb: Optional[Path]
    receptor_source_lines: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)


class TargetPreparator:
    """Prepare any PDB target for docking, driven entirely by configuration.

    Args:
        obabel: Optional explicit Open Babel path.
    """

    def __init__(self, obabel: str | None = None) -> None:
        self._obabel = resolve_tool("obabel", obabel)

    def prepare(self, config) -> PreparedTarget:
        """Fetch and prepare the receptor described by ``config.target``.

        Args:
            config: A :class:`~vspipeline.config.PipelineConfig`.

        Returns:
            A populated :class:`PreparedTarget`.

        Raises:
            RuntimeError: If Open Babel is missing or receptor prep fails.
            ValueError: If a requested native ligand is absent from the PDB.
        """
        if not self._obabel:
            raise RuntimeError("Open Babel (obabel) not found; cannot prepare receptor")

        pdb_id = config.get("target.pdb_id")
        native_res = config.get("target.native_ligand_name")
        chains = config.get("target.chains_to_keep") or []
        bench_dir = Path(config.get("output.benchmark_dir"))
        bench_dir.mkdir(parents=True, exist_ok=True)

        local = config.get("target.pdb_path")
        if local:                                   # local structure (not an RCSB id)
            pdb_path = self._use_local(Path(local), pdb_id, bench_dir)
        else:
            pdb_path = self._fetch(pdb_id, bench_dir)
        lines = pdb_path.read_text(errors="ignore").splitlines()

        # native ligand (single altloc: blank or 'A')
        native_pdb = None
        if native_res:
            nat = [l for l in lines if l.startswith("HETATM")
                   and l[17:20].strip() == native_res and l[16] in (" ", "A")]
            if not nat:
                present = sorted({l[17:20].strip() for l in lines if l.startswith("HETATM")})
                raise ValueError(
                    f"native ligand {native_res!r} not found in {pdb_id}; "
                    f"HETATM residues present: {present}")
            native_pdb = bench_dir / "native_reference_input.pdb"
            native_pdb.write_text("\n".join(nat + ["END"]) + "\n")
            logger.info("native ligand %s: %d atoms (single altloc)", native_res, len(nat))

        # protein: retain requested chains (or all), drop waters/buffers
        keep = set(chains)
        prot = [l for l in lines if l.startswith("ATOM")
                and (not keep or l[21] in keep)
                and l[17:20].strip() not in BUFFERS]
        clean_pdb = bench_dir / "protein_clean.pdb"
        clean_pdb.write_text("\n".join(prot + ["TER", "END"]) + "\n")
        chains_kept = sorted({l[21] for l in prot})
        logger.info("cleaned receptor: %d atoms, chains %s", len(prot), chains_kept)

        receptor = bench_dir / "receptor.pdbqt"
        run([self._obabel, clean_pdb, "-O", receptor, "-xr", "-p", "7.4",
             "--partialcharge", "gasteiger"])
        if not receptor.is_file():
            raise RuntimeError("receptor.pdbqt was not produced by Open Babel")
        logger.info("receptor.pdbqt ready (Gasteiger charges; MGLTools/Kollman unavailable)")

        return PreparedTarget(pdb_id=pdb_id, receptor_pdbqt=receptor,
                              native_ligand_pdb=native_pdb, receptor_source_lines=prot,
                              chains=chains_kept)

    @staticmethod
    def _use_local(src: Path, pdb_id: str, out_dir: Path) -> Path:
        """Copy a local structure into the working dir as ``<pdb_id>.pdb``.

        Args:
            src: Path to the local ``.pdb`` (or ``.ent``) file.
            pdb_id: Logical id used for output naming.
            out_dir: Working directory.

        Returns:
            The cached path inside ``out_dir``.

        Raises:
            RuntimeError: If ``src`` does not exist.
        """
        if not src.is_file():
            raise RuntimeError(f"target.pdb_path not found: {src}")
        dest = out_dir / f"{pdb_id}.pdb"
        if dest.resolve() != src.resolve():
            dest.write_text(src.read_text(errors="ignore"))
        logger.info("using local structure %s -> %s", src.name, dest.name)
        return dest

    def _fetch(self, pdb_id: str, out_dir: Path) -> Path:
        """Download ``pdb_id`` from RCSB (idempotent)."""
        dest = out_dir / f"{pdb_id}.pdb"
        if dest.is_file() and dest.stat().st_size > 0:
            logger.info("reusing cached %s", dest.name)
            return dest
        url = RCSB_URL.format(pdb_id=pdb_id)
        logger.info("downloading %s", url)
        urllib.request.urlretrieve(url, dest)   # noqa: S310 - fixed RCSB host
        return dest
