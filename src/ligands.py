"""Ligand-library loading, subsetting, stratification, and 3D preparation.

Sources: plain ``.smi``, DUD-E (``actives_final.ism`` / ``decoys_final.ism``),
DEKOIS (``*.smi``), or a custom ``.smi``. Benchmark sources carry an
``is_active`` label; screening sources do not.
"""
from __future__ import annotations

import logging
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from .util import resolve_tool, run

logger = logging.getLogger("docksuite.ligands")

DUDE_URL = "http://dude.docking.org/targets/{name}/{name}.tar.gz"


def _natural_key(text: str) -> list:
    """Sort key so ``ligand2`` precedes ``ligand10`` (natural numeric order)."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


class LigandLibrary:
    """Load, subset, and prepare compound libraries for any target.

    Args:
        obabel: Optional explicit Open Babel path (for 3D prep).
        seed: Default RNG seed (overridable per call).
    """

    def __init__(self, obabel: str | None = None, seed: int = 42) -> None:
        self._obabel = resolve_tool("obabel", obabel)
        self.seed = seed

    # ------------------------------------------------------------------ #
    def load(self, config) -> pd.DataFrame:
        """Load a library into a DataFrame with ``smiles``, ``id`` columns.

        Benchmark sources additionally include ``is_active`` (1/0). The frame is
        subset/stratified per ``config.ligands`` when ``subset_size`` is set.

        Args:
            config: A :class:`~vspipeline.config.PipelineConfig`.

        Returns:
            The (possibly subset) library DataFrame.
        """
        source = config.get("ligands.source")
        bench_dir = Path(config.get("output.benchmark_dir"))
        bench_dir.mkdir(parents=True, exist_ok=True)
        seed = int(config.get("ligands.random_seed", self.seed))

        if source == "DUDE":
            df = self._load_dude(config.get("ligands.target_name"), bench_dir)
        elif source == "DEKOIS":
            df = self._load_smi(Path(config.get("ligands.path")), labelled=True)
        elif source in ("SMI", "CUSTOM"):
            df = self._load_smi(Path(config.get("ligands.path")), labelled=False)
        elif source == "MOL2_DIR":
            df = self._load_mol2_dir(Path(config.get("ligands.path")))
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"unknown ligand source: {source!r}")

        n = config.get("ligands.subset_size")
        if n and n < len(df):
            df = self.subset(df, n, config.get("ligands.stratify_by"), seed,
                             config.get("ligands.active_decoy_ratio"))
        logger.info("library ready: %d compounds%s", len(df),
                    f" ({int(df['is_active'].sum())} actives)" if "is_active" in df else "")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def subset(self, df: pd.DataFrame, n: int, stratify_by: Optional[str], seed: int,
               active_decoy_ratio: Optional[float] = None) -> pd.DataFrame:
        """Return an ``n``-compound subset, preserving actives where labelled.

        For labelled libraries, actives are sampled (up to ``ratio``-implied count
        or a 1:3 default) and decoys fill the remainder. Decoys may be stratified
        by a numeric property to preserve coverage.

        Args:
            df: Full library.
            n: Target subset size.
            stratify_by: ``"molecular_weight"`` (quartile-stratified decoys) or
                ``None`` for a plain random draw.
            seed: RNG seed.
            active_decoy_ratio: Optional actives:decoys ratio (e.g. 0.33 -> 1:3);
                ``None`` uses a 1:3 default for labelled sets.

        Returns:
            The subset DataFrame.
        """
        rng = random.Random(seed)
        if "is_active" not in df:
            idx = rng.sample(list(df.index), n)
            return df.loc[idx]

        act = df[df.is_active == 1]
        dec = df[df.is_active == 0]
        ratio = active_decoy_ratio if active_decoy_ratio else 0.25   # 1 active : 3 decoys
        n_act = min(len(act), max(1, int(round(n * ratio))))
        n_dec = min(len(dec), n - n_act)
        act_idx = rng.sample(list(act.index), n_act)

        if stratify_by in ("molecular_weight", "MW") and "MW" in dec:
            dec_sub = self._stratified_mw(dec, n_dec, rng)
        else:
            dec_sub = dec.loc[rng.sample(list(dec.index), n_dec)]
        out = pd.concat([act.loc[act_idx], dec_sub])
        logger.info("subset: %d actives + %d decoys = %d (stratify=%s)",
                    n_act, len(dec_sub), len(out), stratify_by)
        return out

    @staticmethod
    def _stratified_mw(dec: pd.DataFrame, n_dec: int, rng: random.Random) -> pd.DataFrame:
        dec = dec.copy()
        dec["_q"] = pd.qcut(dec["MW"], 4, labels=False, duplicates="drop")
        per = max(1, n_dec // 4)
        picks: list = []
        for q in sorted(dec["_q"].dropna().unique()):
            pool = list(dec[dec["_q"] == q].index)
            picks += rng.sample(pool, min(per, len(pool)))
        while len(picks) < n_dec and len(picks) < len(dec):
            extra = rng.choice(list(dec.index))
            if extra not in picks:
                picks.append(extra)
        return dec.loc[picks[:n_dec]].drop(columns="_q")

    # ------------------------------------------------------------------ #
    def prepare_ligand(self, row: dict, out_dir: Path, seed: int = 42) -> Optional[Path]:
        """Prepare one ligand ``.pdbqt`` from whichever input the row carries.

        Dispatches to :meth:`prepare_pdbqt_from_mol2` for pre-built 3D mol2
        conformers, or :meth:`prepare_pdbqt` for SMILES.

        Args:
            row: A library record (dict) with ``id`` and either ``mol2_path``
                or ``smiles``.
            out_dir: Destination directory.
            seed: RDKit embedding seed (SMILES path only).

        Returns:
            The ``.pdbqt`` path, or ``None`` on failure.
        """
        if row.get("mol2_path"):
            return self.prepare_pdbqt_from_mol2(Path(row["mol2_path"]), row["id"], out_dir)
        return self.prepare_pdbqt(row["smiles"], row["id"], out_dir, seed)

    def prepare_pdbqt_from_mol2(self, mol2_path: Path, name: str, out_dir: Path) -> Optional[Path]:
        """Convert an existing 3D ``.mol2`` conformer to ``.pdbqt`` (idempotent).

        Preserves the supplied conformer (no re-embedding); merges non-polar
        hydrogens (united-atom) and assigns Gasteiger charges.

        Args:
            mol2_path: Source ``.mol2`` file.
            name: Output basename.
            out_dir: Destination directory.

        Returns:
            The ``.pdbqt`` path, or ``None`` on failure.
        """
        pdbqt = out_dir / f"{name}.pdbqt"
        if pdbqt.is_file() and pdbqt.stat().st_size > 0:
            return pdbqt
        run([self._obabel, str(mol2_path), "-O", str(pdbqt), "--partialcharge", "gasteiger"],
            check=False)
        return pdbqt if pdbqt.is_file() and pdbqt.stat().st_size > 0 else None

    def prepare_pdbqt(self, smiles: str, name: str, out_dir: Path,
                      seed: int = 42) -> Optional[Path]:
        """Embed a SMILES to 3D and write a Gasteiger ``.pdbqt`` (idempotent).

        Args:
            smiles: Input SMILES.
            name: Output basename.
            out_dir: Destination directory.
            seed: RDKit embedding seed.

        Returns:
            The ``.pdbqt`` path, or ``None`` on RDKit/embedding failure.
        """
        pdbqt = out_dir / f"{name}.pdbqt"
        if pdbqt.is_file() and pdbqt.stat().st_size > 0:
            return pdbqt
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:  # pragma: no cover - occasional MMFF params gap
            pass
        sdf = out_dir / f"{name}.sdf"
        Chem.SDWriter(str(sdf)).write(mol)
        mol2 = out_dir / f"{name}.mol2"
        run([self._obabel, sdf, "-O", mol2], check=False)
        run([self._obabel, mol2, "-O", pdbqt, "--partialcharge", "gasteiger"], check=False)
        sdf.unlink(missing_ok=True)
        mol2.unlink(missing_ok=True)
        return pdbqt if pdbqt.is_file() else None

    # ------------------------------------------------------------------ #
    def _load_dude(self, name: str, bench_dir: Path) -> pd.DataFrame:
        root = bench_dir / "dude" / name
        if not (root / "actives_final.ism").is_file():
            self._download_dude(name, bench_dir)
        act = self._read_ism(root / "actives_final.ism", 1)
        dec = self._read_ism(root / "decoys_final.ism", 0)
        df = pd.concat([act, dec], ignore_index=True)
        self._attach_mw(df)
        logger.info("DUD-E %s: %d actives, %d decoys", name, len(act), len(dec))
        return df

    def _download_dude(self, name: str, bench_dir: Path) -> None:
        tgz = bench_dir / f"{name}.tar.gz"
        logger.info("downloading DUD-E %s", name)
        urllib.request.urlretrieve(DUDE_URL.format(name=name), tgz)   # noqa: S310
        with tarfile.open(tgz) as tf:
            tf.extractall(bench_dir / "dude")   # noqa: S202 - trusted DUD-E archive
        # normalise layout: dude/<name>/actives_final.ism
        extracted = bench_dir / "dude" / name
        if not extracted.is_dir():
            for p in (bench_dir / "dude").iterdir():
                if p.is_dir() and (p / "actives_final.ism").is_file():
                    p.rename(extracted)
                    break

    @staticmethod
    def _read_ism(path: Path, is_active: int) -> pd.DataFrame:
        rows = []
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                tag = "active" if is_active else "decoy"
                rows.append({"smiles": parts[0], "id": f"{tag}_{parts[1]}", "is_active": is_active})
        return pd.DataFrame(rows)

    @staticmethod
    def _load_mol2_dir(path: Path) -> pd.DataFrame:
        """Load a directory of pre-built 3D ``.mol2`` conformers (unlabelled)."""
        if not path.is_dir():
            raise ValueError(f"MOL2_DIR path is not a directory: {path}")
        files = sorted(path.glob("*.mol2"), key=lambda p: _natural_key(p.stem))
        if not files:
            raise ValueError(f"no .mol2 files in {path}")
        logger.info("MOL2_DIR: %d conformers in %s", len(files), path.name)
        return pd.DataFrame([{"id": p.stem, "mol2_path": str(p)} for p in files])

    @staticmethod
    def _load_smi(path: Path, labelled: bool) -> pd.DataFrame:
        rows = []
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 1:
                continue
            smi = parts[0]
            cid = parts[1] if len(parts) > 1 else f"lig{len(rows)+1}"
            row = {"smiles": smi, "id": cid}
            if labelled:
                row["is_active"] = int(cid.lower().startswith(("active", "bdb", "chembl")))
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _attach_mw(df: pd.DataFrame) -> None:
        """Add an ``MW`` column via RDKit (needed for MW stratification)."""
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Descriptors
            RDLogger.DisableLog("rdApp.*")
        except Exception:  # pragma: no cover
            return
        mws = []
        for smi in df["smiles"]:
            m = Chem.MolFromSmiles(smi)
            mws.append(Descriptors.MolWt(m) if m else float("nan"))
        df["MW"] = mws
