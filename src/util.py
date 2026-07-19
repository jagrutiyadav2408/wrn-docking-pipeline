"""Internal helpers: binary resolution, subprocess wrapper, logging setup.

Not target-specific. Binary locations may be supplied via config or environment;
otherwise the PATH and a set of conventional install directories are searched.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("docksuite.util")

#: Conventional install locations searched when a tool is not on PATH.
CONVENTIONAL_DIRS: list[Path] = [
    Path(r"C:\Program Files (x86)\The Scripps Research Institute\Autodock\4.2.6"),
    Path("/usr/local/bin"),
    Path("/opt/autodock"),
]

#: Logical tool name -> candidate executable basenames.
TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "obabel": ("obabel",),
    "autogrid4": ("autogrid4",),
    "autodock_gpu": ("AutoDock-GPU", "autodock_gpu", "adgpu"),
    "gnina": ("gnina",),
}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root package logging with a concise, timestamped format.

    Args:
        level: Logging level (e.g. ``logging.INFO``).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger("docksuite")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def resolve_tool(name: str, override: Optional[str] = None,
                 extra_dirs: Optional[Sequence[Path]] = None) -> Optional[str]:
    """Resolve an executable path for a logical tool name.

    Search order: explicit ``override`` -> ``$DOCKSUITE_<NAME>`` env var ->
    ``PATH`` -> conventional install dirs (+ ``extra_dirs``).

    Args:
        name: Logical tool key from :data:`TOOL_ALIASES`.
        override: Explicit path from config; returned verbatim if it exists.
        extra_dirs: Additional directories to search.

    Returns:
        Absolute path string, or ``None`` if the tool cannot be found.
    """
    if override and Path(override).exists():
        return str(override)
    env = os.environ.get(f"DOCKSUITE_{name.upper()}")
    if env and Path(env).exists():
        return env
    dirs = list(CONVENTIONAL_DIRS) + list(extra_dirs or [])
    for basename in TOOL_ALIASES.get(name, (name,)):
        found = shutil.which(basename)
        if found:
            return found
        for d in dirs:
            for cand in (d / basename, d / f"{basename}.exe"):
                if cand.is_file():
                    return str(cand)
    return None


def run(cmd: Sequence[object], cwd: Optional[Path] = None, check: bool = True,
        timeout: Optional[int] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output.

    Args:
        cmd: Command and arguments (each coerced to ``str``).
        cwd: Working directory.
        check: Raise :class:`RuntimeError` on non-zero exit if ``True``.
        timeout: Seconds before killing the process.
        env: Environment overrides merged onto ``os.environ``.

    Returns:
        The completed process.

    Raises:
        RuntimeError: If ``check`` and the process exits non-zero.
    """
    full_env = {**os.environ, **env} if env else None
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout, env=full_env)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT: {proc.stdout[-600:]}\nSTDERR: {proc.stderr[-600:]}")
    return proc


def progress(iterable, desc: str, disable: bool = False):
    """Wrap an iterable in a tqdm progress bar, degrading gracefully if absent."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, disable=disable)
    except Exception:  # pragma: no cover - tqdm optional
        return iterable
