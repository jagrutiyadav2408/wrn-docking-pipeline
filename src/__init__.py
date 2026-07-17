"""
Generalized, configuration-driven virtual-screening & retrospective-benchmark
pipeline (AutoDock-GPU backend, Gnina-ready).

Every behaviour is controlled by a single JSON config — no target-specific logic,
PDB IDs, residue names, or paths are hardcoded anywhere in this package.

Public API
----------
    PipelineConfig      - load / validate / override configuration
    TargetPreparator    - fetch + clean + prepare any PDB receptor
    LigandLibrary       - load / subset / stratify any compound library
    GridEngine          - MATCH_INBUILT or BLIND_DOCK grid boxes + maps
    DockingEngine       - backend-agnostic docking (AUTODOCK now, GNINA hook)
    QualityGates        - torsion / charge / conformer ligand gates
    PoseExtractor       - parse .dlg, extract top pose of largest cluster
    MetricsEngine       - ROC-AUC / EF / BEDROC with bootstrap CIs
    ReportGenerator     - Excel + figures + terminal Markdown
    BenchmarkRunner     - single-entry orchestration
"""
from __future__ import annotations

import logging

from .config import PipelineConfig
from .consensus import ConsensusRanker, ConsensusResult
from .grid import GridBox, GridEngine
from .ligands import LigandLibrary
from .metrics import BenchmarkMetrics, MetricsEngine
from .poses import Pose, PoseExtractor
from .quality import QualityGates
from .report import ReportGenerator
from .target import PreparedTarget, TargetPreparator
from .docking import DockingEngine
from .benchmark import BenchmarkRunner

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig", "ConsensusRanker", "ConsensusResult", "TargetPreparator", "PreparedTarget", "LigandLibrary",
    "GridEngine", "GridBox", "DockingEngine", "QualityGates", "PoseExtractor",
    "Pose", "MetricsEngine", "BenchmarkMetrics", "ReportGenerator",
    "BenchmarkRunner", "__version__",
]

# library best-practice: attach a NullHandler; apps/CLI configure real handlers
logging.getLogger(__name__).addHandler(logging.NullHandler())
