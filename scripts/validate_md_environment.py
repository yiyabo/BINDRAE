#!/usr/bin/env python3
"""Validate the isolated OpenMM environment used by the BINDRAE MD pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional


REQUIRED_MODULES = {
    "openmm": ("openmm",),
    "pdbfixer": ("pdbfixer",),
    "openff.toolkit": ("openff-toolkit", "openff-toolkit-base"),
    "openff.interchange": ("openff-interchange", "openff-interchange-base"),
    "openmmforcefields": ("openmmforcefields",),
    "rdkit": ("rdkit",),
    "parmed": ("parmed",),
    "mdtraj": ("mdtraj",),
    "MDAnalysis": ("MDAnalysis",),
}
REQUIRED_EXECUTABLES = ("antechamber", "tleap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless OpenMM exposes its CUDA platform",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser.parse_args()


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def distribution_version(distribution_names: tuple[str, ...]) -> Optional[str]:
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def executable_path(executable: str) -> Optional[str]:
    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved
    environment_candidate = Path(sys.executable).resolve().parent / executable
    if environment_candidate.is_file() and os.access(environment_candidate, os.X_OK):
        return str(environment_candidate)
    return None


def openmm_platforms() -> Dict[str, object]:
    if not module_available("openmm"):
        return {"names": [], "load_error": "openmm is not installed"}
    try:
        from openmm import Platform

        names = [
            Platform.getPlatform(index).getName()
            for index in range(Platform.getNumPlatforms())
        ]
        return {"names": names, "load_error": None}
    except Exception as exc:  # OpenMM plugin errors are runtime-specific.
        return {"names": [], "load_error": f"{type(exc).__name__}: {exc}"}


def build_report(require_cuda: bool) -> Dict[str, object]:
    modules = {
        module_name: {
            "available": module_available(module_name),
            "version": distribution_version(distribution_names),
        }
        for module_name, distribution_names in REQUIRED_MODULES.items()
    }
    executables = {
        executable: executable_path(executable)
        for executable in REQUIRED_EXECUTABLES
    }
    platforms = openmm_platforms()
    platform_names = platforms["names"]
    missing_modules = sorted(
        module_name
        for module_name, metadata in modules.items()
        if not metadata["available"]
    )
    cuda_available = "CUDA" in platform_names
    errors = []
    if missing_modules:
        errors.append(f"missing required modules: {', '.join(missing_modules)}")
    missing_executables = sorted(
        executable for executable, path in executables.items() if path is None
    )
    if missing_executables:
        errors.append(
            f"missing required executables: {', '.join(missing_executables)}"
        )
    if platforms["load_error"]:
        errors.append(f"OpenMM platform load failed: {platforms['load_error']}")
    if require_cuda and not cuda_available:
        errors.append("OpenMM CUDA platform is unavailable")

    return {
        "ok": not errors,
        "require_cuda": bool(require_cuda),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "modules": modules,
        "executables": executables,
        "openmm_platforms": platforms,
        "cuda_available": cuda_available,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    report = build_report(require_cuda=bool(args.require_cuda))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
