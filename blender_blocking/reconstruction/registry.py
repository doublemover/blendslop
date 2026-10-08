"""Global reconstruction backend registry."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

from .backend import BackendInfo, ReconstructionBackend

_BACKENDS: Dict[str, ReconstructionBackend] = {}
_ALIASES: Dict[str, str] = {}


def register_backend(
    backend: ReconstructionBackend,
    *,
    replace: bool = False,
    aliases: Sequence[str] = (),
) -> None:
    """Register a backend by name."""
    if not replace and (backend.name in _BACKENDS or backend.name in _ALIASES):
        raise ValueError(f"backend already registered: {backend.name}")
    if replace and backend.name in _ALIASES:
        del _ALIASES[backend.name]
    for alias in aliases:
        if alias == backend.name:
            raise ValueError(f"backend alias duplicates canonical name: {alias}")
        if alias in _BACKENDS:
            raise ValueError(f"backend alias conflicts with registered backend: {alias}")
        if not replace and alias in _ALIASES:
            raise ValueError(f"backend alias already registered: {alias}")
    _BACKENDS[backend.name] = backend
    for alias, target in tuple(_ALIASES.items()):
        if target == backend.name:
            del _ALIASES[alias]
    for alias in aliases:
        _ALIASES[alias] = backend.name


def get_backend(name: str) -> ReconstructionBackend:
    canonical = _ALIASES.get(name, name)
    try:
        return _BACKENDS[canonical]
    except KeyError as exc:
        names = set(_BACKENDS) | set(_ALIASES)
        available = ", ".join(sorted(names)) or "<none>"
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


def list_backend_aliases() -> dict[str, str]:
    return dict(sorted(_ALIASES.items()))


def clear_backends_for_tests() -> None:
    _BACKENDS.clear()
    _ALIASES.clear()


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
    from .backends.shape_program import ShapeProgramBackend
    from .backends.implicit_residual import ImplicitResidualBackend

    for backend, aliases in (
        (LegacySliceBackend(), ()),
        (ProfileLoftBackend(), ("loft_profile",)),
        (SilhouetteIntersectionBackend(), ()),
        (VisualHullBackend(), ()),
        (HybridLoftHullBackend(), ()),
        (PrimitiveFitBackend(), ()),
        (GaussianEllipsoidBackend(), ()),
        (DifferentiableRefinementBackend(), ()),
        (ShapeProgramBackend(), ()),
        (ImplicitResidualBackend(), ()),
    ):
        register_backend(backend, replace=True, aliases=aliases)
