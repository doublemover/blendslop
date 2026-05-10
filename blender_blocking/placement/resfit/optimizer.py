"""Optimizer orchestration exports for residual fitting."""

from __future__ import annotations

from .pipeline import fit_residual_primitives, fit_residual_primitives_multistart

__all__ = ["fit_residual_primitives", "fit_residual_primitives_multistart"]
