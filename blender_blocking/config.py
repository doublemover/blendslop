"""Public configuration surface for the Blender automated blocking tool."""

from __future__ import annotations

try:
    from blender_blocking.config_models import *  # type: ignore[F403]
    from blender_blocking.config_models import __all__
except ImportError:  # pragma: no cover - legacy script import path
    from config_models import *  # type: ignore[F403]
    from config_models import __all__  # type: ignore[no-redef]
