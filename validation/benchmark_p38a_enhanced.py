#!/usr/bin/env python3
"""
Enhanced p38a benchmark: applies size-normalization + shape-filtering corrections
POST-HOC to the already-docked 40-compound subset (reuses existing .dlg files, no
re-docking / no GPU). Reports raw vs corrected ROC-AUC / EF / BEDROC honestly.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import os
# Portable data root: set $DOCKSUITE_DATA to your working dir
_BASE = Path(os.environ.get("DOCKSUITE_DATA", "./validation_data")).resolve()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Descriptors3D, Lipinski, MolSurf, rdFreeSASA
from sklearn.metrics import roc_auc_score, roc_curve
RDLogger.DisableLog("rdApp.*")

BENCH  = _BASE / "p38a_benchmark"
OUTPUT = _BASE / "p38a_output"
FIG    = _BASE / "figures"
SMI    = BENCH / "DEKOIS2" / "DEKOIS2" / "p38-alpha" / "active_decoys.smi"
OBABEL = __import__("shutil").which("obabel") or "obabel"
