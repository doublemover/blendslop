"""Registry of synthetic shape definitions and suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .random_params import (
    adversarial_spec,
    analytic_primitive_spec,
    capture_noise_spec,
    compound_blockout_spec,
    material_fixture_spec,
    profile_lathe_spec,
)
from .materials import MATERIAL_FIXTURE_KINDS
from .specs import ShapeFamily, SyntheticShapeSpec


SpecFactory = Callable[[int], SyntheticShapeSpec]


@dataclass(frozen=True)
class ShapeDefinition:
    name: str
    family: str
    description: str
    pure_python: bool
    blender_supported: bool
    factory: SpecFactory

    def create(self, seed: int) -> SyntheticShapeSpec:
        return self.factory(seed)


ANALYTIC_PRIMITIVES = (
    "box",
    "sphere",
    "ellipsoid",
    "cylinder",
    "frustum",
    "capsule",
    "torus",
    "rounded_box",
    "superquadric",
)

PROFILE_LATHE = (
    "vase",
    "bottle",
    "bowl",
    "cup",
    "chess_pawn",
    "asymmetric_vase",
    "multi_lobe_profile",
)

FURNITURE = (
    "table",
    "chair",
    "stool",
    "lamp",
    "shelf_bookcase",
    "arch_gate",
    "ladder",
    "sawhorse",
)

VEHICLE_MECHANICAL = (
    "car",
    "truck",
    "robot_torso",
    "wrench",
    "pipe_elbow",
    "gear",
)

ADVERSARIAL_SILHOUETTES = (
    "off_center_dark",
    "border_touching",
    "single_outlier_pixel",
    "dust_clusters",
    "thin_diagonal_struts",
    "holed_silhouette",
    "disconnected_components",
    "zero_radius_rings",
    "tiny_object_large_canvas",
    "full_canvas_near_threshold",
    "ambiguous_polarity_pair",
    "inconsistent_front_side",
    "checkerboard_breakup",
    "frame_with_corner_gap",
    "off_canvas_ellipse",
    "single_pixel_noise",
)

CAPTURE_NOISE = (
    "jpeg_artifacts",
    "blur",
    "paper_sketch_lines",
    "partial_occlusion",
    "uneven_lighting",
    "low_contrast",
    "alpha_premultiplication",
    "transparent_rgb_noise",
    "missing_top_view",
    "tilted_input",
    "salt_and_pepper",
    "scanline_jitter",
    "radial_vignette",
    "posterize",
)

MATERIAL_APPEARANCE = tuple(f"material_{kind}" for kind in MATERIAL_FIXTURE_KINDS)

DETERMINISTIC_MICRO = (
    "box",
    "sphere",
    "asymmetric_vase",
    "chair",
    "car",
    "single_outlier_pixel",
    "low_contrast",
)


def _build_registry() -> dict[str, ShapeDefinition]:
    definitions: dict[str, ShapeDefinition] = {}
    for name in ANALYTIC_PRIMITIVES:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.ANALYTIC_PRIMITIVE.value,
            description=f"Analytic {name} fixture with exact SDF/occupancy.",
            pure_python=True,
            blender_supported=True,
            factory=lambda seed, shape_name=name: analytic_primitive_spec(shape_name, seed),
        )
    for name in PROFILE_LATHE:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.PROFILE_LATHE.value,
            description=f"Profile/lathe {name} fixture with exact source profile JSON.",
            pure_python=True,
            blender_supported=True,
            factory=lambda seed, shape_name=name: profile_lathe_spec(shape_name, seed),
        )
    for name in FURNITURE:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.FURNITURE.value,
            description=f"Furniture thin-support blockout fixture: {name}.",
            pure_python=True,
            blender_supported=True,
            factory=lambda seed, shape_name=name: compound_blockout_spec(shape_name, seed, ShapeFamily.FURNITURE),
        )
    for name in VEHICLE_MECHANICAL:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.VEHICLE_MECHANICAL.value,
            description=f"Vehicle/mechanical multi-part fixture: {name}.",
            pure_python=True,
            blender_supported=True,
            factory=lambda seed, shape_name=name: compound_blockout_spec(shape_name, seed, ShapeFamily.VEHICLE_MECHANICAL),
        )
    for name in ADVERSARIAL_SILHOUETTES:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.ADVERSARIAL_SILHOUETTE.value,
            description=f"Pure 2D adversarial mask fixture: {name}.",
            pure_python=True,
            blender_supported=False,
            factory=lambda seed, shape_name=name: adversarial_spec(shape_name, seed),
        )
    for name in CAPTURE_NOISE:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.CAPTURE_NOISE.value,
            description=f"Capture-style mask degradation fixture: {name}.",
            pure_python=True,
            blender_supported=False,
            factory=lambda seed, shape_name=name: capture_noise_spec(shape_name, seed),
        )
    for name in MATERIAL_APPEARANCE:
        definitions[name] = ShapeDefinition(
            name=name,
            family=ShapeFamily.ANALYTIC_PRIMITIVE.value,
            description=f"Material, UV, and appearance metric fixture: {name}.",
            pure_python=True,
            blender_supported=True,
            factory=lambda seed, shape_name=name: material_fixture_spec(_material_kind(shape_name), seed),
        )
    return definitions


REGISTRY = _build_registry()


SUITES: dict[str, tuple[str, ...]] = {
    "capture-noise": CAPTURE_NOISE,
    "deterministic-micro": DETERMINISTIC_MICRO,
    "smoke": ("box", "sphere", "vase", "table", "single_outlier_pixel"),
    "quick-blender": ("box", "vase", "table", "asymmetric_vase", "gear"),
    "blender-smoke": ("box", "vase", "table", "asymmetric_vase", "gear"),
    "adversarial-silhouettes": ADVERSARIAL_SILHOUETTES,
    "silhouette-edge-cases": (
        "border_touching",
        "single_outlier_pixel",
        "dust_clusters",
        "checkerboard_breakup",
        "frame_with_corner_gap",
        "off_canvas_ellipse",
        "single_pixel_noise",
    ),
    "degradation-stress": (
        "low_contrast",
        "partial_occlusion",
        "salt_and_pepper",
        "scanline_jitter",
        "radial_vignette",
        "posterize",
        "missing_top_view",
    ),
    "profile-band": ("vase", "bottle", "bowl", "cup", "chess_pawn", "asymmetric_vase", "multi_lobe_profile"),
    "visual-hull": ("table", "chair", "car", "truck", "gear", "pipe_elbow"),
    "primitive-fit": ("ellipsoid", "frustum", "capsule", "torus", "rounded_box", "superquadric"),
    "material-appearance": MATERIAL_APPEARANCE,
    "nightly-heavy": ANALYTIC_PRIMITIVES + PROFILE_LATHE + FURNITURE + VEHICLE_MECHANICAL + ADVERSARIAL_SILHOUETTES + CAPTURE_NOISE + MATERIAL_APPEARANCE,
    "paper-regression": ("torus", "superquadric", "gear", "pipe_elbow", "table", "inconsistent_front_side"),
}


def get_definition(name: str) -> ShapeDefinition:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown synthetic shape definition {name!r}") from exc


def list_definitions(family: str | None = None) -> tuple[ShapeDefinition, ...]:
    definitions = tuple(REGISTRY[name] for name in sorted(REGISTRY))
    if family is None:
        return definitions
    return tuple(definition for definition in definitions if definition.family == family)


def list_suites() -> tuple[str, ...]:
    return tuple(sorted(SUITES))


def suite_names(suite: str, count: int | None = None) -> tuple[str, ...]:
    if suite not in SUITES:
        raise KeyError(f"Unknown synthetic suite {suite!r}")
    names = SUITES[suite]
    if count is None:
        return names
    if count < 0:
        raise ValueError("count must be non-negative")
    repeated = []
    while len(repeated) < count:
        repeated.extend(names)
    return tuple(repeated[:count])


def specs_for_suite(suite: str, seed: int = 0, count: int | None = None) -> tuple[SyntheticShapeSpec, ...]:
    specs = []
    for index, name in enumerate(suite_names(suite, count)):
        specs.append(get_definition(name).create(seed + index))
    return tuple(specs)


def _material_kind(name: str) -> str:
    prefix = "material_"
    if not name.startswith(prefix):
        raise ValueError(f"Material fixture name must start with {prefix!r}: {name!r}")
    return name[len(prefix) :]
