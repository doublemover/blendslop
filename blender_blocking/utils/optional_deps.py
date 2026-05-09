"""Lazy optional dependency detection helpers.

Ambitious reconstruction modes should report precise skip reasons when heavy
libraries are absent.  This module centralizes those probes so backends do not
hide optional import failures behind broad exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class OptionalDependency:
    """Availability record for a lazily imported dependency."""

    name: str
    available: bool
    module: Optional[ModuleType] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def skip_reason(self) -> str:
        if self.available:
            return ""
        return f"optional dependency {self.name!r} is unavailable: {self.error}"

    def require(self) -> ModuleType:
        """Return the imported module or raise with a clear message."""
        if self.module is None:
            raise RuntimeError(self.skip_reason)
        return self.module

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe availability record."""
        return {
            "module_name": self.name,
            "available": self.available,
            "module_version": getattr(self.module, "__version__", None)
            if self.module is not None
            else None,
            "module_file": getattr(self.module, "__file__", None)
            if self.module is not None
            else None,
            "error_type": self.error_type,
            "error": self.error,
        }


_CACHE: Dict[str, OptionalDependency] = {}


def probe_dependency(import_name: str, *, cache: bool = True) -> OptionalDependency:
    """Probe one import path without making it a hard runtime dependency."""
    if cache and import_name in _CACHE:
        return _CACHE[import_name]
    try:
        module = importlib.import_module(import_name)
        result = OptionalDependency(import_name, True, module=module)
    except Exception as exc:
        result = OptionalDependency(
            import_name,
            False,
            error=str(exc),
            error_type=type(exc).__name__,
        )
    if cache:
        _CACHE[import_name] = result
    return result


def dependency_report(import_names: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """Return JSON-safe availability records for multiple dependencies."""
    report: Dict[str, Dict[str, object]] = {}
    for name in import_names:
        dep = probe_dependency(name)
        report[name] = dep.to_dict()
    return report


def require_dependency(import_name: str) -> ModuleType:
    """Import an optional dependency or raise a skip-ready RuntimeError."""
    return probe_dependency(import_name).require()


def clear_dependency_cache() -> None:
    """Reset cached probes for tests."""
    _CACHE.clear()
