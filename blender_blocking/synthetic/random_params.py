"""Deterministic parameter generation for synthetic fixture specs."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .specs import (
    ChallengeTag,
    FailureModeTag,
    ShapeFamily,
    SyntheticShapeSpec,
    challenge_values,
    failure_values,
    stable_shape_id,
)


@dataclass(frozen=True)
class SeededGenerator:
    seed: int
    namespace: str = "synthetic"

    def rng(self, *parts: object) -> random.Random:
        text = "|".join([self.namespace, str(self.seed), *(str(part) for part in parts)])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))


def analytic_primitive_spec(kind: str, seed: int, variant: str | None = None) -> SyntheticShapeSpec:
    rng = SeededGenerator(seed, "analytic").rng(kind, variant or "")
    params: dict[str, Any]
    dims: dict[str, float]
    symmetry = ("x", "y", "z")
    challenges = [ChallengeTag.ANALYTIC_EXACT]
    failures: list[FailureModeTag] = []

    if kind == "box":
        dims = {
            "width": round(rng.uniform(0.7, 1.8), 4),
            "depth": round(rng.uniform(0.7, 1.8), 4),
            "height": round(rng.uniform(0.7, 1.8), 4),
        }
        params = {"primitive": "box", "dimensions": dims}
    elif kind == "sphere":
        radius = round(rng.uniform(0.45, 0.95), 4)
        dims = {"radius": radius, "diameter": radius * 2.0}
        params = {"primitive": "sphere", "radius": radius}
        symmetry = ("radial",)
    elif kind == "ellipsoid":
        radii = [round(rng.uniform(0.35, 1.0), 4) for _ in range(3)]
        dims = {"radius_x": radii[0], "radius_y": radii[1], "radius_z": radii[2]}
        params = {"primitive": "ellipsoid", "radii": radii}
    elif kind == "cylinder":
        radius = round(rng.uniform(0.35, 0.85), 4)
        height = round(rng.uniform(0.8, 2.2), 4)
        dims = {"radius": radius, "height": height}
        params = {"primitive": "cylinder", "radius": radius, "height": height}
        symmetry = ("z_axis_radial",)
    elif kind == "frustum":
        radius_bottom = round(rng.uniform(0.5, 0.9), 4)
        radius_top = round(rng.uniform(0.15, 0.65), 4)
        height = round(rng.uniform(0.9, 2.0), 4)
        dims = {"radius_bottom": radius_bottom, "radius_top": radius_top, "height": height}
        params = {
            "primitive": "frustum",
            "radius_bottom": radius_bottom,
            "radius_top": radius_top,
            "height": height,
        }
        symmetry = ("z_axis_radial",)
    elif kind == "capsule":
        radius = round(rng.uniform(0.25, 0.65), 4)
        segment_height = round(rng.uniform(0.6, 1.8), 4)
        dims = {"radius": radius, "segment_height": segment_height, "height": segment_height + 2.0 * radius}
        params = {"primitive": "capsule", "radius": radius, "segment_height": segment_height}
        symmetry = ("z_axis_radial",)
    elif kind == "torus":
        major_radius = round(rng.uniform(0.45, 0.8), 4)
        minor_radius = round(rng.uniform(0.08, min(0.28, major_radius * 0.45)), 4)
        dims = {"major_radius": major_radius, "minor_radius": minor_radius}
        params = {"primitive": "torus", "major_radius": major_radius, "minor_radius": minor_radius}
        symmetry = ("z_axis_radial",)
        challenges.append(ChallengeTag.HOLES)
        failures.append(FailureModeTag.VISUAL_HULL_CONCAVITY_FILL)
    elif kind == "rounded_box":
        dims = {
            "width": round(rng.uniform(0.8, 1.8), 4),
            "depth": round(rng.uniform(0.8, 1.8), 4),
            "height": round(rng.uniform(0.8, 1.8), 4),
        }
        radius = round(rng.uniform(0.05, 0.18), 4)
        params = {"primitive": "rounded_box", "dimensions": dims, "radius": radius}
        dims["rounding"] = radius
    elif kind == "superquadric":
        radii = [round(rng.uniform(0.45, 0.95), 4) for _ in range(3)]
        exponents = [round(rng.uniform(0.55, 2.2), 4), round(rng.uniform(0.55, 2.2), 4)]
        taper = round(rng.uniform(-0.25, 0.25), 4)
        dims = {"radius_x": radii[0], "radius_y": radii[1], "radius_z": radii[2], "taper": taper}
        params = {"primitive": "superquadric", "radii": radii, "exponents": exponents, "taper": taper}
        challenges.append(ChallengeTag.PROFILE_BANDS)
    else:
        raise ValueError(f"Unknown analytic primitive: {kind}")

    return SyntheticShapeSpec(
        shape_id=stable_shape_id(kind, seed, variant),
        family=ShapeFamily.ANALYTIC_PRIMITIVE.value,
        seed=seed,
        parameters=params,
        transforms={
            "position": [0.0, 0.0, 0.0],
            "rotation_deg": [round(rng.uniform(-8.0, 8.0), 4) for _ in range(3)],
            "scale": [1.0, 1.0, 1.0],
        },
        materials={"base_color": [0.2, 0.2, 0.2, 1.0], "roughness": 0.55},
        intended_challenges=challenge_values(challenges),
        symmetry=symmetry,
        known_dimensions=dims,
        expected_failure_modes=failure_values(failures),
    )


def profile_lathe_spec(kind: str, seed: int) -> SyntheticShapeSpec:
    rng = SeededGenerator(seed, "profile").rng(kind)
    profiles = {
        "vase": [[-1.0, 0.28], [-0.75, 0.48], [-0.3, 0.38], [0.35, 0.52], [0.78, 0.22], [1.0, 0.24]],
        "bottle": [[-1.0, 0.48], [-0.35, 0.52], [0.2, 0.34], [0.45, 0.17], [1.0, 0.17]],
        "bowl": [[-0.45, 0.18], [-0.2, 0.55], [0.2, 0.78], [0.45, 0.82]],
        "cup": [[-0.75, 0.36], [0.65, 0.38], [0.82, 0.42]],
        "chess_pawn": [[-1.0, 0.34], [-0.82, 0.5], [-0.62, 0.28], [-0.1, 0.2], [0.18, 0.36], [0.48, 0.24], [0.78, 0.18], [1.0, 0.28]],
        "asymmetric_vase": [[-1.0, 0.32, -0.06], [-0.55, 0.55, 0.05], [-0.05, 0.35, 0.12], [0.45, 0.5, -0.04], [1.0, 0.22, 0.0]],
        "multi_lobe_profile": [[-1.0, [0.18, 0.4]], [-0.4, [0.28, 0.52]], [0.15, [0.18, 0.33, 0.58]], [0.7, [0.25, 0.5]], [1.0, [0.2]]],
    }
    if kind not in profiles:
        raise ValueError(f"Unknown profile lathe shape: {kind}")
    height = round(rng.uniform(1.4, 2.5), 4)
    return SyntheticShapeSpec(
        shape_id=stable_shape_id(kind, seed),
        family=ShapeFamily.PROFILE_LATHE.value,
        seed=seed,
        parameters={"profile_kind": kind, "height": height, "profile": profiles[kind], "segments": 96},
        transforms={"position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        materials={"base_color": [0.18, 0.18, 0.18, 1.0]},
        intended_challenges=challenge_values([ChallengeTag.PROFILE_BANDS]),
        symmetry=("z_axis_radial",) if "asymmetric" not in kind else ("approximate_z_axis_radial",),
        known_dimensions={"height": height, "max_radius": 0.82},
        expected_failure_modes=failure_values([FailureModeTag.INSUFFICIENT_VIEWS] if "multi_lobe" in kind else []),
    )


def compound_blockout_spec(kind: str, seed: int, family: ShapeFamily) -> SyntheticShapeSpec:
    rng = SeededGenerator(seed, family.value).rng(kind)
    challenges = [ChallengeTag.MULTI_PART]
    failures = [FailureModeTag.SILHOUETTE_AMBIGUITY]
    if family == ShapeFamily.FURNITURE:
        challenges.append(ChallengeTag.THIN_SUPPORTS)
        failures.extend([FailureModeTag.THIN_LEG_LOSS, FailureModeTag.INTERNAL_OVERLAP, FailureModeTag.NON_MANIFOLD_RISK])
    if kind in ("arch_gate", "car", "truck", "wrench", "pipe_elbow", "gear"):
        challenges.extend([ChallengeTag.HOLES, ChallengeTag.CONCAVITY])
        failures.append(FailureModeTag.VISUAL_HULL_CONCAVITY_FILL)
    return SyntheticShapeSpec(
        shape_id=stable_shape_id(kind, seed),
        family=family.value,
        seed=seed,
        parameters={
            "blockout_kind": kind,
            "parts": _compound_parts(kind, rng),
            "boolean_intent": "union_minus_gaps" if ChallengeTag.HOLES in challenges else "union",
        },
        transforms={"position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        materials={"base_color": [0.24, 0.24, 0.24, 1.0]},
        intended_challenges=challenge_values(challenges),
        symmetry=("bilateral",),
        known_dimensions={"width": 2.0, "depth": 1.2, "height": 1.6},
        expected_failure_modes=failure_values(failures),
    )


def adversarial_spec(kind: str, seed: int) -> SyntheticShapeSpec:
    return SyntheticShapeSpec(
        shape_id=stable_shape_id(kind, seed),
        family=ShapeFamily.ADVERSARIAL_SILHOUETTE.value,
        seed=seed,
        parameters={"mask_kind": kind, "resolution": [256, 256], "polarity": "dark_on_light"},
        transforms={},
        materials={},
        intended_challenges=challenge_values(_adversarial_challenges(kind)),
        symmetry=(),
        known_dimensions={"canvas_width": 256.0, "canvas_height": 256.0},
        expected_failure_modes=failure_values(_adversarial_failures(kind)),
    )


def capture_noise_spec(kind: str, seed: int) -> SyntheticShapeSpec:
    return SyntheticShapeSpec(
        shape_id=stable_shape_id(kind, seed),
        family=ShapeFamily.CAPTURE_NOISE.value,
        seed=seed,
        parameters={"degradation": kind, "resolution": [256, 256], "base_mask": "center_disc"},
        transforms={},
        materials={},
        intended_challenges=challenge_values(_capture_challenges(kind)),
        symmetry=(),
        known_dimensions={"canvas_width": 256.0, "canvas_height": 256.0},
        expected_failure_modes=failure_values(_capture_failures(kind)),
    )


def _compound_parts(kind: str, rng: random.Random) -> list[dict[str, Any]]:
    leg_radius = round(rng.uniform(0.035, 0.07), 4)
    if kind == "table":
        return [
            {"type": "box", "name": "top", "center": [0, 0, 0.8], "size": [1.8, 1.1, 0.16]},
            *[
                {"type": "cylinder", "name": f"leg_{x}_{y}", "center": [x, y, 0.35], "radius": leg_radius, "height": 0.7}
                for x in (-0.72, 0.72)
                for y in (-0.42, 0.42)
            ],
        ]
    if kind == "chair":
        return [
            {"type": "box", "name": "seat", "center": [0, 0, 0.55], "size": [1.0, 1.0, 0.14]},
            {"type": "box", "name": "back", "center": [0, -0.45, 1.05], "size": [1.0, 0.12, 1.0]},
            *[
                {"type": "cylinder", "name": f"leg_{x}_{y}", "center": [x, y, 0.28], "radius": leg_radius, "height": 0.55}
                for x in (-0.38, 0.38)
                for y in (-0.38, 0.38)
            ],
        ]
    if kind == "stool":
        return [{"type": "cylinder", "name": "seat", "center": [0, 0, 0.72], "radius": 0.55, "height": 0.16}] + [
            {"type": "cylinder", "name": f"leg_{i}", "center": [x, y, 0.34], "radius": leg_radius, "height": 0.68}
            for i, (x, y) in enumerate(((0.0, 0.42), (-0.36, -0.24), (0.36, -0.24)))
        ]
    if kind == "lamp":
        return [
            {"type": "cylinder", "name": "base", "center": [0, 0, 0.1], "radius": 0.36, "height": 0.2},
            {"type": "cylinder", "name": "stem", "center": [0, 0, 0.86], "radius": 0.045, "height": 1.45},
            {"type": "frustum", "name": "shade", "center": [0, 0, 1.65], "radius_bottom": 0.52, "radius_top": 0.28, "height": 0.45},
        ]
    if kind == "shelf_bookcase":
        return [
            {"type": "box", "name": "left_side", "center": [-0.62, 0, 0.75], "size": [0.08, 0.34, 1.5]},
            {"type": "box", "name": "right_side", "center": [0.62, 0, 0.75], "size": [0.08, 0.34, 1.5]},
            *[{"type": "box", "name": f"shelf_{i}", "center": [0, 0, z], "size": [1.3, 0.34, 0.07]} for i, z in enumerate((0.15, 0.55, 0.95, 1.35))],
        ]
    if kind == "arch_gate":
        return [
            {"type": "box", "name": "left_pier", "center": [-0.48, 0, 0.55], "size": [0.18, 0.28, 1.1]},
            {"type": "box", "name": "right_pier", "center": [0.48, 0, 0.55], "size": [0.18, 0.28, 1.1]},
            {"type": "torus_section", "name": "arch", "center": [0, 0, 1.08], "major_radius": 0.48, "minor_radius": 0.09, "arc_deg": 180},
        ]
    if kind == "ladder":
        return [
            {"type": "cylinder", "name": "rail_l", "center": [-0.35, 0, 0.78], "radius": 0.035, "height": 1.55},
            {"type": "cylinder", "name": "rail_r", "center": [0.35, 0, 0.78], "radius": 0.035, "height": 1.55},
            *[{"type": "box", "name": f"rung_{i}", "center": [0, 0, z], "size": [0.74, 0.08, 0.045]} for i, z in enumerate((0.22, 0.5, 0.78, 1.06, 1.34))],
        ]
    if kind == "sawhorse":
        return [
            {"type": "box", "name": "beam", "center": [0, 0, 0.95], "size": [1.6, 0.16, 0.14]},
            *[
                {"type": "box", "name": f"angled_leg_{i}", "center": [x, y, 0.45], "size": [0.1, 0.1, 0.9], "rotation_deg": [12 * sx, 0, 0]}
                for i, (x, y, sx) in enumerate(((-0.55, -0.26, -1), (-0.55, 0.26, 1), (0.55, -0.26, -1), (0.55, 0.26, 1)))
            ],
        ]
    if kind in ("car", "truck"):
        length = 2.2 if kind == "car" else 2.8
        return [
            {"type": "box", "name": "body", "center": [0, 0, 0.45], "size": [length, 0.9, 0.45]},
            {"type": "box", "name": "cabin", "center": [-0.2, 0, 0.86], "size": [0.9, 0.78, 0.52]},
            *[
                {"type": "cylinder", "name": f"wheel_gap_{i}", "center": [x, y, 0.28], "radius": 0.24, "height": 0.14, "boolean": "subtract"}
                for i, (x, y) in enumerate(((-0.78, -0.47), (-0.78, 0.47), (0.78, -0.47), (0.78, 0.47)))
            ],
        ]
    if kind == "robot_torso":
        return [
            {"type": "box", "name": "torso", "center": [0, 0, 0.8], "size": [0.65, 0.35, 0.9]},
            {"type": "sphere", "name": "head", "center": [0, 0, 1.42], "radius": 0.22},
            *[{"type": "capsule", "name": name, "center": center, "radius": 0.08, "segment_height": 0.55} for name, center in (("arm_l", [-0.48, 0, 0.82]), ("arm_r", [0.48, 0, 0.82]), ("leg_l", [-0.18, 0, 0.2]), ("leg_r", [0.18, 0, 0.2]))],
        ]
    if kind == "wrench":
        return [
            {"type": "capsule", "name": "handle", "center": [0, 0, 0], "radius": 0.12, "segment_height": 1.5, "rotation_deg": [0, 90, 0]},
            {"type": "torus_section", "name": "open_head", "center": [0.86, 0, 0], "major_radius": 0.28, "minor_radius": 0.08, "arc_deg": 265},
        ]
    if kind == "pipe_elbow":
        return [{"type": "torus_section", "name": "elbow", "center": [0, 0, 0], "major_radius": 0.58, "minor_radius": 0.15, "arc_deg": 90}]
    if kind == "gear":
        return [{"type": "gear", "name": "gear", "center": [0, 0, 0], "teeth": 12, "root_radius": 0.45, "tip_radius": 0.62, "height": 0.16, "hole_radius": 0.16}]
    raise ValueError(f"Unknown compound blockout: {kind}")


def _adversarial_challenges(kind: str) -> list[ChallengeTag]:
    mapping = {
        "off_center_dark": [ChallengeTag.AMBIGUOUS_POLARITY],
        "border_touching": [ChallengeTag.BORDER_TOUCHING],
        "single_outlier_pixel": [ChallengeTag.OUTLIER_PIXELS],
        "dust_clusters": [ChallengeTag.DUST_CLUSTERS],
        "thin_diagonal_struts": [ChallengeTag.THIN_SUPPORTS],
        "holed_silhouette": [ChallengeTag.HOLES],
        "disconnected_components": [ChallengeTag.MULTI_PART],
        "ambiguous_polarity_pair": [ChallengeTag.AMBIGUOUS_POLARITY],
        "inconsistent_front_side": [ChallengeTag.PROFILE_BANDS],
        "checkerboard_breakup": [ChallengeTag.PROFILE_BANDS],
        "frame_with_corner_gap": [ChallengeTag.BORDER_TOUCHING, ChallengeTag.HOLES],
        "off_canvas_ellipse": [ChallengeTag.BORDER_TOUCHING],
        "single_pixel_noise": [ChallengeTag.OUTLIER_PIXELS],
    }
    return mapping.get(kind, [])


def _adversarial_failures(kind: str) -> list[FailureModeTag]:
    mapping = {
        "single_outlier_pixel": [FailureModeTag.OUTLIER_BBOX_EXPANSION],
        "dust_clusters": [FailureModeTag.OUTLIER_BBOX_EXPANSION],
        "disconnected_components": [FailureModeTag.COMPONENT_DROPOUT],
        "ambiguous_polarity_pair": [FailureModeTag.POLARITY_MISCLASSIFICATION],
        "inconsistent_front_side": [FailureModeTag.INSUFFICIENT_VIEWS],
        "holed_silhouette": [FailureModeTag.VISUAL_HULL_CONCAVITY_FILL],
        "thin_diagonal_struts": [FailureModeTag.THIN_LEG_LOSS],
        "checkerboard_breakup": [FailureModeTag.SILHOUETTE_AMBIGUITY],
        "frame_with_corner_gap": [FailureModeTag.VISUAL_HULL_CONCAVITY_FILL],
        "off_canvas_ellipse": [FailureModeTag.OUTLIER_BBOX_EXPANSION],
        "single_pixel_noise": [FailureModeTag.COMPONENT_DROPOUT],
    }
    return mapping.get(kind, [FailureModeTag.SILHOUETTE_AMBIGUITY])


def _capture_challenges(kind: str) -> list[ChallengeTag]:
    mapping = {
        "uneven_lighting": [ChallengeTag.LOW_CONTRAST],
        "low_contrast": [ChallengeTag.LOW_CONTRAST],
        "partial_occlusion": [ChallengeTag.OCCLUSION],
        "alpha_premultiplication": [ChallengeTag.LOW_CONTRAST],
        "transparent_rgb_noise": [ChallengeTag.LOW_CONTRAST, ChallengeTag.DUST_CLUSTERS],
        "missing_top_view": [ChallengeTag.OCCLUSION],
        "salt_and_pepper": [ChallengeTag.OUTLIER_PIXELS],
        "scanline_jitter": [ChallengeTag.LOW_CONTRAST],
        "radial_vignette": [ChallengeTag.LOW_CONTRAST],
        "posterize": [ChallengeTag.PROFILE_BANDS],
    }
    return mapping.get(kind, [])


def _capture_failures(kind: str) -> list[FailureModeTag]:
    mapping = {
        "missing_top_view": [FailureModeTag.COMPONENT_DROPOUT],
        "salt_and_pepper": [FailureModeTag.OUTLIER_BBOX_EXPANSION],
        "scanline_jitter": [FailureModeTag.POLARITY_MISCLASSIFICATION],
        "radial_vignette": [FailureModeTag.POLARITY_MISCLASSIFICATION],
        "posterize": [FailureModeTag.SILHOUETTE_AMBIGUITY],
    }
    return mapping.get(kind, [])
