"""CLI compatibility entrypoint for the benchmark harness."""

from __future__ import annotations

try:
    from .harness import *  # noqa: F401,F403
    from .harness import main
except ImportError:  # pragma: no cover - direct script path
    import sys
    from pathlib import Path

    blender_blocking_root = Path(__file__).resolve().parents[1]
    repo_root = blender_blocking_root.parent
    for path in (blender_blocking_root, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from benchmarks.harness import *  # type: ignore # noqa: F401,F403
    from benchmarks.harness import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
