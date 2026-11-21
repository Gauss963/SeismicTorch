"""Workspace and packaged-resource locations, independent of the current directory."""

import os
from pathlib import Path


def project_root():
    configured = os.environ.get("SEISMICTORCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src/seismictorch").is_dir():
                return candidate
    return Path.cwd()


ROOT = project_root()
DATA = ROOT / "data"
RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_CANDIDATE = ROOT / "models/qsis/current"
