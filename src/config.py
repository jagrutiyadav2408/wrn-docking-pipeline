"""Configuration loading, validation, and dotted-path access/override."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("docksuite.config")

_SEARCH_MODES = {"MATCH_INBUILT", "BLIND_DOCK"}
_BACKENDS = {"AUTODOCK", "GNINA"}
_ADMET_MODES = {"ALL", "TOP_N", "NONE"}
_LIGAND_SOURCES = {"DUDE", "DEKOIS", "SMI", "CUSTOM", "MOL2_DIR"}

# (dotted_key, type_or_none, required)
_SCHEMA: tuple[tuple[str, Any, bool], ...] = (
    ("target.pdb_id", str, True),
    ("target.native_ligand_name", (str, type(None)), False),
    ("target.chains_to_keep", list, False),
    ("target.search_mode", str, True),
    ("target.box_buffer_angstroms", (int, float), True),
    ("target.grid_spacing", (int, float), True),
    ("ligands.source", str, True),
    ("ligands.subset_size", (int, type(None)), False),
    ("ligands.random_seed", int, True),
    ("docking.backend", str, True),
    ("docking.runs_per_ligand", int, True),
    ("docking.cluster_rmsd_threshold", (int, float), True),
    ("admet.mode", str, True),
    ("output.benchmark_dir", str, True),
    ("output.output_dir", str, True),
)


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or has invalid values."""


class PipelineConfig:
    """Load, validate, and expose a JSON pipeline configuration.

    Access values by dotted path via :meth:`get`, or the raw nested dict via
    :attr:`data`. No target-specific defaults are baked in; only structural
    defaults (e.g. an empty ``chains_to_keep`` meaning "all chains").

    Args:
        config_path: Path to a JSON config file.
        overrides: Optional mapping of dotted-path -> value applied after load
            (typically from CLI flags such as ``--target.pdb_id 1W82``).

    Raises:
        ConfigError: If the file is unreadable or fails validation.
    """

    def __init__(self, config_path: str, overrides: dict[str, Any] | None = None) -> None:
        self.path = Path(config_path)
        if not self.path.is_file():
            raise ConfigError(f"config file not found: {self.path}")
        try:
            self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"invalid JSON in {self.path}: {e}") from e
        if overrides:
            for dotted, value in overrides.items():
                self.set(dotted, value)
        self.validate()
        logger.info("loaded config %s (target=%s, mode=%s, backend=%s)",
                    self.path.name, self.get("target.pdb_id"),
                    self.get("target.search_mode"), self.get("docking.backend"))

    # ------------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        """Return the value at a dotted path, or ``default`` if absent.

        Args:
            dotted: Path such as ``"target.pdb_id"``.
            default: Returned when any path segment is missing.
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Set (creating intermediate dicts) the value at a dotted path.

        String values that look like ints/floats/bools/null are coerced, so CLI
        overrides like ``--ligands.subset_size 40`` arrive with the right type.
        """
        value = _coerce(value)
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        node[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration dict."""
        return copy.deepcopy(self.data)

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate presence, types, and enum membership of known keys.

        Raises:
            ConfigError: On the first structural problem encountered.
        """
        for dotted, typ, required in _SCHEMA:
            val = self.get(dotted, _MISSING)
            if val is _MISSING:
                if required:
                    raise ConfigError(f"missing required config key: {dotted!r}")
                continue
            if typ is not None and not isinstance(val, typ):
                raise ConfigError(
                    f"config key {dotted!r} must be {typ}, got {type(val).__name__}")

        self._require_enum("target.search_mode", _SEARCH_MODES)
        self._require_enum("docking.backend", _BACKENDS)
        self._require_enum("admet.mode", _ADMET_MODES)
        self._require_enum("ligands.source", _LIGAND_SOURCES, upper=True)

        if self.get("target.search_mode") == "MATCH_INBUILT" and not self.get("target.native_ligand_name"):
            raise ConfigError("MATCH_INBUILT requires target.native_ligand_name")
        if self.get("admet.mode") == "TOP_N" and not self.get("admet.top_n"):
            raise ConfigError("admet.mode=TOP_N requires admet.top_n")
        if self.get("target.grid_spacing") <= 0:
            raise ConfigError("target.grid_spacing must be > 0")

    def _require_enum(self, dotted: str, allowed: Iterable[str], upper: bool = False) -> None:
        val = self.get(dotted)
        if upper and isinstance(val, str):
            val = val.upper()
            self.set(dotted, val)
        if val not in allowed:
            raise ConfigError(f"{dotted!r}={val!r} not in {sorted(allowed)}")


_MISSING = object()


def _coerce(value: Any) -> Any:
    """Coerce CLI/string scalars to int/float/bool/None where unambiguous."""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value
