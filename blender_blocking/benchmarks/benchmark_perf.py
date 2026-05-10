"""CLI compatibility entrypoint for the benchmark harness."""

from __future__ import annotations

try:
    from .harness import *  # noqa: F401,F403
    from .harness import main
except ImportError:  # pragma: no cover - direct script path
    from harness import *  # type: ignore # noqa: F401,F403
    from harness import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
