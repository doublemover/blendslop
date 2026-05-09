"""View job construction shared by pure and Blender synthetic builders."""

from __future__ import annotations

from .specs import DEFAULT_VIEWS, SyntheticShapeSpec, SyntheticViewSpec


def canonical_view_jobs(
    spec: SyntheticShapeSpec,
    resolution: tuple[int, int] = (256, 256),
    include_orbit: bool = False,
) -> tuple[SyntheticViewSpec, ...]:
    views = [
        SyntheticViewSpec(
            view.view_name,
            view.camera_type,
            view.azimuth_deg,
            view.elevation_deg,
            view.roll_deg,
            resolution,
            view.padding,
            view.occluders,
            _expected_features(spec, view.view_name),
        )
        for view in DEFAULT_VIEWS
    ]
    if include_orbit:
        for angle in range(0, 360, 30):
            views.append(
                SyntheticViewSpec(
                    f"orbit_{angle:03d}",
                    "orthographic",
                    azimuth_deg=float(angle),
                    elevation_deg=0.0,
                    resolution=resolution,
                    expected_visible_features=_expected_features(spec, "orbit"),
                )
            )
    return tuple(views)


def _expected_features(spec: SyntheticShapeSpec, view_name: str) -> tuple[str, ...]:
    if spec.family == "adversarial_silhouette":
        return (str(spec.parameters.get("mask_kind", "mask")),)
    if "thin_supports" in spec.intended_challenges:
        return ("thin_supports", "primary_mass")
    if "holes" in spec.intended_challenges:
        return ("outer_silhouette", "holes_or_gaps")
    if view_name == "top":
        return ("top_extent",)
    return ("outer_silhouette",)
