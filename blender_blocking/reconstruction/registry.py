"""Global reconstruction backend registry."""

from __future__ import annotations

from typing import Dict, Iterable

from .backend import BackendInfo, ReconstructionBackend

_BACKENDS: Dict[str, ReconstructionBackend] = {}


def register_backend(backend: ReconstructionBackend, *, replace: bool = False) -> None:
    """Register a backend by name."""
    if not replace and backend.name in _BACKENDS:
        raise ValueError(f"backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> ReconstructionBackend:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise KeyError(f"unknown backend {name!r}; available: {available}") from exc


def list_backends() -> list[BackendInfo]:
    return [
        BackendInfo(
            name=backend.name,
            version=backend.version,
            capabilities=backend.capabilities,
        )
        for backend in _BACKENDS.values()
    ]


def iter_backends() -> Iterable[ReconstructionBackend]:
    return tuple(_BACKENDS.values())


def clear_backends_for_tests() -> None:
    _BACKENDS.clear()


def register_builtin_backends() -> None:
    """Register built-ins. Imports are local to keep optional deps lazy."""
    from .backends.legacy_slice import LegacySliceBackend
    from .backends.profile_loft import ProfileLoftBackend
    from .backends.silhouette_intersection import SilhouetteIntersectionBackend
    from .backends.visual_hull import VisualHullBackend
    from .backends.hybrid_loft_hull import HybridLoftHullBackend
    from .backends.primitive_fit import PrimitiveFitBackend
    from .backends.gaussian_ellipsoid import GaussianEllipsoidBackend
    from .backends.differentiable_refine import DifferentiableRefinementBackend

    for backend in (
        LegacySliceBackend(),
        ProfileLoftBackend(),
        SilhouetteIntersectionBackend(),
        VisualHullBackend(),
        HybridLoftHullBackend(),
        PrimitiveFitBackend(),
        GaussianEllipsoidBackend(),
        DifferentiableRefinementBackend(),
    ):
        register_backend(backend, replace=True)
