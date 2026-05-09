"""Helpers for building reconstruction targets from preprocessed inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .artifacts import hash_json
from .types import (
    Bounds2D,
    Bounds3D,
    OrthographicCameraSpec,
    ProfileBand,
    ReconstructionTarget,
    ViewConstraint,
)


def make_axis_camera(view: str, resolution: tuple[int, int]) -> OrthographicCameraSpec:
    axis = {"front": "y", "side": "x", "top": "z"}.get(view, "custom")
    azimuth = {"front": 0.0, "side": 90.0, "top": 0.0}.get(view, 0.0)
    elevation = 90.0 if view == "top" else 0.0
    return OrthographicCameraSpec(
        view_name=view,
        axis=axis,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        resolution=resolution,
    )


def make_view_constraints(
    masks_by_view: Mapping[str, Any],
    *,
    bboxes_by_view: Optional[Mapping[str, Bounds2D]] = None,
    uncertainties_by_view: Optional[Mapping[str, Any]] = None,
    diagnostics_by_view: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> tuple[ViewConstraint, ...]:
    constraints: list[ViewConstraint] = []
    for view, mask in sorted(masks_by_view.items()):
        uncertainty = (uncertainties_by_view or {}).get(view)
        mask_value = getattr(mask, "hard_mask", mask)
        if uncertainty is None and hasattr(mask, "foreground_prob"):
            uncertainty = mask
        shape = getattr(mask_value, "shape", None)
        if shape is not None and len(shape) >= 2:
            resolution = (int(shape[1]), int(shape[0]))
        else:
            resolution = (512, 512)
        constraints.append(
            ViewConstraint(
                view=view,
                mask=mask_value,
                camera=make_axis_camera(view, resolution),
                bbox=(bboxes_by_view or {}).get(view),
                uncertainty=uncertainty,
                diagnostics=(diagnostics_by_view or {}).get(view, {}),
            )
        )
    return tuple(constraints)


def make_reconstruction_target(
    *,
    masks_by_view: Mapping[str, Any],
    profile_bands: Optional[Mapping[str, Sequence[ProfileBand]]] = None,
    bboxes_by_view: Optional[Mapping[str, Bounds2D]] = None,
    uncertainties_by_view: Optional[Mapping[str, Any]] = None,
    diagnostics_by_view: Optional[Mapping[str, Mapping[str, Any]]] = None,
    bounds: Optional[Bounds3D] = None,
    config: Optional[Mapping[str, Any]] = None,
    constraints_payload: Optional[Mapping[str, Any]] = None,
    artifact_root: str | Path | None = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> ReconstructionTarget:
    config_hash = hash_json(config or {})
    constraint_hash = hash_json(constraints_payload or {})
    return ReconstructionTarget(
        constraints=make_view_constraints(
            masks_by_view,
            bboxes_by_view=bboxes_by_view,
            uncertainties_by_view=uncertainties_by_view,
            diagnostics_by_view=diagnostics_by_view,
        ),
        profile_bands={
            view: tuple(bands) for view, bands in (profile_bands or {}).items()
        },
        bounds=bounds,
        config_hash=config_hash,
        constraint_hash=constraint_hash,
        artifact_root=Path(artifact_root) if artifact_root is not None else None,
        extras=dict(extras or {}),
    )
