"""Run historical scripts with the exact archived Python source and original layout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

from seismictorch.paths import ROOT


def historical_workspace():
    migration = json.loads((ROOT / "docs/reorganization.json").read_text())
    archive = ROOT / migration["frozen_source_archive"]
    identity = hashlib.sha256(archive.read_bytes()).hexdigest()[:16]
    workspace = ROOT / ".cache/legacy-runtime" / identity
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name, expected in migration["original_source_sha256"].items():
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Archived source changed: {name}")
            target = workspace / name
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Cached historical source changed: {target}")
            else:
                target.write_bytes(data)
    directories = {row["old"]: ROOT / row["new"] for row in migration["directory_moves"]}
    directories.update({p.name: p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")})
    for name, destination in directories.items():
        if name == "__pycache__":
            continue
        link = workspace / name
        if link.is_symlink():
            if link.resolve() != destination.resolve():
                raise ValueError(f"Historical directory alias changed: {link}")
        elif not link.exists():
            link.symlink_to(destination.resolve(), target_is_directory=True)
    return workspace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", action="store_true", help="Print the persistent historical workspace")
    parser.add_argument("script", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.workspace and not args.script:
        parser.error("provide a Python script or --workspace")
    workspace = historical_workspace()
    if args.workspace:
        print(workspace)
        return
    script = Path(args.script)
    if not script.is_absolute():
        script = workspace / script
    if not script.is_file() or script.suffix != ".py":
        parser.error("script must name an existing Python file")
    previous_directory, previous_path, previous_argv = Path.cwd(), sys.path[:], sys.argv
    try:
        os.chdir(workspace)
        sys.path[:0] = [str(script.parent), str(workspace)]
        sys.argv = [str(script), *args.arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path
        sys.argv = previous_argv
