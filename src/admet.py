"""ADMET profiling and tiered risk stratification.

Combines RDKit physicochemical descriptors + drug-likeness rules (Lipinski,
Veber, bRo5) with ADMET-AI (Chemprop, 104 TDC endpoints) and PAINS/Brenk alerts,
then assigns each compound an EXCLUDE / HIGH RISK / MODERATE / PASS tier.

Target-agnostic: operates on ``(id, smiles[, dG])`` records from any source.
ADMET-AI is optional — if unavailable, physchem rules + FilterCatalog alerts are
still produced and this substitution is logged.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger("docksuite.admet")

#: Curated ADMET-AI endpoints surfaced in the main report (full set kept separately).
KEY_ENDPOINTS: tuple[str, ...] = (
    "HIA_Hou", "Caco2_Wang", "Pgp_Broccatelli", "Bioavailability_Ma", "PAMPA_NCATS",
    "BBB_Martins", "PPBR_AZ", "VDss_Lombardo",
    "CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
    "Clearance_Hepatocyte_AZ", "Half_Life_Obach",
    "AMES", "hERG", "DILI", "ClinTox", "Carcinogens_Lagunin", "LD50_Zhu",
)


class AdmetProfiler:
    """Profile compounds and stratify them by developability risk.

    Args:
        mode: ``"ALL"`` (profile everything), ``"TOP_N"`` (best ``top_n`` by dG),
            or ``"NONE"`` (skip; returns an empty frame).
        top_n: Number of compounds for ``TOP_N`` mode.
        use_admet_ai: Run ADMET-AI if importable (falls back to physchem-only).
    """

    def __init__(self, mode: str = "ALL", top_n: Optional[int] = None,
                 use_admet_ai: bool = True) -> None:
        self.mode = mode.upper()
        self.top_n = top_n
        self.use_admet_ai = use_admet_ai

    # ------------------------------------------------------------------ #
    def profile(self, records: pd.DataFrame) -> pd.DataFrame:
        """Profile a table of compounds.

        Args:
            records: DataFrame with columns ``id`` and ``smiles`` (optionally
                ``dG``/``score`` from docking, used for TOP_N selection and merge).

        Returns:
            One row per profiled compound: descriptors, drug-likeness flags,
            selected ADMET-AI endpoints, and a ``Risk_Tier``. Empty if mode NONE.
        """
        if self.mode == "NONE" or records.empty:
            logger.info("ADMET stage skipped (mode=%s)", self.mode)
            return pd.DataFrame()

        df = records.copy()
        if self.mode == "TOP_N" and self.top_n and "dG" in df:
            df = df.sort_values("dG").head(int(self.top_n))
            logger.info("ADMET TOP_N: profiling best %d by dG", len(df))
        else:
            logger.info("ADMET ALL: profiling %d compounds", len(df))

        physchem = self._physchem(df["smiles"].tolist())
        out = pd.concat([df.reset_index(drop=True), physchem], axis=1)

        if self.use_admet_ai:
            ai = self._admet_ai(df["smiles"].tolist())
            if ai is not None:
                out = pd.concat([out, ai.reset_index(drop=True)], axis=1)

        # Series-aware tiering: an endpoint that is uniformly high across the whole
        # set carries no per-compound signal (often an out-of-distribution artifact,
        # e.g. DILI saturating for large beyond-Ro5 analogs). Flag it series-wide and
        # exclude it from per-compound EXCLUDE logic so real discriminators drive tiers.
        saturated = self._saturated_endpoints(out)
        if saturated:
            logger.warning("non-discriminating saturated endpoint(s) flagged series-wide "
                           "(excluded from per-compound tiering): %s", saturated)
        out["series_flags"] = ";".join(f"{k}~{v}" for k, v in saturated.items())
        ignore = frozenset(saturated)
        out["Risk_Tier"] = out.apply(lambda r: self._classify(r, ignore), axis=1)
        if "dG" in out:
            out = out.sort_values("dG").reset_index(drop=True)
        logger.info("risk tiers: %s", out["Risk_Tier"].value_counts().to_dict())
        return out

    @staticmethod
    def _saturated_endpoints(out: pd.DataFrame) -> dict[str, float]:
        """Toxicity endpoints that are uniformly high (near-zero spread)."""
        sat = {}
        for ep in ("hERG", "AMES", "DILI", "Carcinogens_Lagunin", "ClinTox"):
            if ep in out and len(out) >= 5 and out[ep].std() < 0.05 and out[ep].median() >= 0.7:
                sat[ep] = round(float(out[ep].median()), 2)
        return sat

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_smiles(smi: str) -> str:
        """Repair writer quirks that RDKit rejects.

        OpenBabel emits nitro groups as ``[N](=O)[O-]`` / ``[N](=O)=O`` (neutral
        5-valent N), which RDKit refuses. The charge-separated ``[N+](=O)[O-]``
        form is the valid equivalent.
        """
        return (smi.replace("[N](=O)[O-]", "[N+](=O)[O-]")
                   .replace("[N](=O)=O", "[N+](=O)[O-]")
                   .replace("[n](=O)[O-]", "[n+](=O)[O-]"))

    @classmethod
    def _physchem(cls, smiles: Sequence[str]) -> pd.DataFrame:
        """RDKit descriptors + Lipinski/Veber/bRo5 flags.

        Compounds RDKit cannot parse (even after :meth:`_sanitize_smiles`) get
        ``parse_ok=False`` so they are tiered UNPARSED rather than silently
        defaulting to a clean profile.
        """
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
        RDLogger.DisableLog("rdApp.*")
        rows = []
        for smi in smiles:
            m = Chem.MolFromSmiles(smi)
            if m is None:                                  # try the quirk repair
                fixed = cls._sanitize_smiles(smi)
                m = Chem.MolFromSmiles(fixed)
                if m is not None:
                    logger.info("recovered unparseable SMILES via writer-quirk repair")
            if m is None:
                logger.warning("RDKit cannot parse SMILES (tiered UNPARSED): %s", smi)
                rows.append({"parse_ok": False})
                continue
            mw = Descriptors.MolWt(m)
            logp = Crippen.MolLogP(m)
            hbd = Lipinski.NumHDonors(m)
            hba = Lipinski.NumHAcceptors(m)
            tpsa = Descriptors.TPSA(m)
            rotb = Descriptors.NumRotatableBonds(m)
            lip_viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
            rows.append({
                "MW": round(mw, 1), "cLogP": round(logp, 2), "HBD": hbd, "HBA": hba,
                "TPSA": round(tpsa, 1), "RotB": rotb, "QED": round(QED.qed(m), 3),
                "Lipinski_violations": lip_viol,
                "Veber_pass": bool(rotb <= 10 and tpsa <= 140),
                "beyond_Ro5": bool(mw > 500 or logp > 5),
                "parse_ok": True,
            })
        return pd.DataFrame(rows)

    def _admet_ai(self, smiles: Sequence[str]) -> Optional[pd.DataFrame]:
        """Run ADMET-AI; return curated endpoints + alert columns, or None."""
        try:
            from admet_ai import ADMETModel
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning("ADMET-AI unavailable (%s); physchem-only profile", e)
            return None
        logger.info("running ADMET-AI on %d compounds...", len(smiles))
        preds = ADMETModel().predict(smiles=list(smiles)).reset_index(drop=True)
        keep = [c for c in (*KEY_ENDPOINTS, "PAINS_alert", "BRENK_alert", "NIH_alert")
                if c in preds.columns]
        return preds[keep].round(3)

    @staticmethod
    def _classify(row, ignore: frozenset = frozenset()) -> str:
        """Assign a developability tier from alerts, tox, and drug-likeness.

        Args:
            row: One profiled compound.
            ignore: Endpoints to skip in tox logic (series-wide saturated ones).
        """
        def g(k, default=0.0):
            v = row.get(k, default)
            return default if pd.isna(v) else v

        def tox(k, thr):        # honour the ignore set for saturated endpoints
            return k not in ignore and g(k) >= thr

        # Guard first: a compound whose structure could not be parsed has no
        # descriptors. Never let those NaNs default through to a clean tier.
        if not row.get("parse_ok", True) or pd.isna(row.get("MW", float("nan"))):
            return "UNPARSED"

        # EXCLUDE: structural liability or high predicted toxicity
        if g("PAINS_alert") > 0 or tox("hERG", 0.7) or tox("AMES", 0.7) \
                or tox("DILI", 0.7) or tox("Carcinogens_Lagunin", 0.7):
            return "EXCLUDE"
        # HIGH RISK: moderate toxicity, poor absorption, or multiple Ro5 breaks
        if tox("hERG", 0.5) or tox("AMES", 0.5) or tox("DILI", 0.5) \
                or tox("ClinTox", 0.5) or g("HIA_Hou") < 0.3 \
                or g("Lipinski_violations") >= 2:
            return "HIGH RISK"
        # MODERATE: Brenk alert, one Ro5 break, or broad CYP inhibition
        cyp_hits = sum(g(f"CYP{i}") >= 0.5 for i in
                       ("1A2_Veith", "2C19_Veith", "2C9_Veith", "2D6_Veith", "3A4_Veith"))
        if g("BRENK_alert") > 0 or g("Lipinski_violations") == 1 or cyp_hits >= 3 \
                or g("Bioavailability_Ma") < 0.3:
            return "MODERATE"
        return "PASS"


__all__ = ["AdmetProfiler", "KEY_ENDPOINTS"]
