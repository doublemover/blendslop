#!/usr/bin/env python3
"""Blender ``--python`` entry point for refinement-lab CLI commands."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "blender_blocking"))


def _script_args() -> list[str]:
    if "--" in sys.argv:
        return list(sys.argv[sys.argv.index("--") + 1 :])
    return list(sys.argv[1:])


def main() -> int:
    from blender_blocking.verify_setup import configure_dependency_paths

    configure_dependency_paths()

    from blender_blocking.refinement_lab.cli import main as refinement_main

    return refinement_main(_script_args())


if __name__ == "__main__":
    raise SystemExit(main())
