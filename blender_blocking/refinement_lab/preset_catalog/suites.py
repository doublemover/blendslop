from __future__ import annotations

from .types import BLENDER_BLOCKING_ROOT, SuitePreset

SUITES: dict[str, SuitePreset] = {
    "default-vase": SuitePreset(
        name="default-vase",
        source="builtin_sample",
        description="Built-in orthogonal vase references used for fast real-image regression.",
        reference_paths={
            "front": BLENDER_BLOCKING_ROOT / "test_images" / "vase_front.png",
            "side": BLENDER_BLOCKING_ROOT / "test_images" / "vase_side.png",
            "top": BLENDER_BLOCKING_ROOT / "test_images" / "vase_top.png",
        },
        tags=("quick", "real-image", "vase"),
    ),
    "synthetic-smoke": SuitePreset(
        name="synthetic-smoke",
        source="synthetic",
        description="Small synthetic sanity suite.",
        synthetic_suites=("smoke",),
        tags=("quick", "synthetic"),
    ),
    "synthetic-blender-smoke": SuitePreset(
        name="synthetic-blender-smoke",
        source="synthetic",
        description="Small synthetic sanity suite limited to Blender mesh-backed fixtures.",
        synthetic_suites=("blender-smoke",),
        tags=("quick", "synthetic", "blender"),
    ),
    "synthetic-visual-hull": SuitePreset(
        name="synthetic-visual-hull",
        source="synthetic",
        description="Synthetic shapes that stress visual hull volume and meshing behavior.",
        synthetic_suites=("visual-hull",),
        tags=("synthetic", "visual-hull"),
    ),
    "synthetic-profile-band": SuitePreset(
        name="synthetic-profile-band",
        source="synthetic",
        description="Synthetic profile/lathe shapes for profile-band and loft tuning.",
        synthetic_suites=("profile-band",),
        tags=("synthetic", "profile"),
    ),
    "synthetic-primitive-fit": SuitePreset(
        name="synthetic-primitive-fit",
        source="synthetic",
        description="Synthetic primitive fixtures for primitive, Gaussian, and differentiable fitting.",
        synthetic_suites=("primitive-fit",),
        tags=("synthetic", "primitive"),
    ),
    "synthetic-adversarial": SuitePreset(
        name="synthetic-adversarial",
        source="synthetic",
        description="Combined synthetic edge cases for mask extraction and robustness.",
        synthetic_suites=("silhouette-edge-cases", "degradation-stress"),
        tags=("synthetic", "adversarial"),
    ),
    "synthetic-adversarial-curriculum": SuitePreset(
        name="synthetic-adversarial-curriculum",
        source="synthetic",
        description="Progressive adversarial curriculum from baseline contracts through missing-view, topology, and ambiguity stress cases.",
        synthetic_suites=(
            "adversarial-level-1",
            "adversarial-level-2",
            "adversarial-level-3",
        ),
        tags=("synthetic", "adversarial", "curriculum", "moonshot"),
    ),
    "synthetic-material-appearance": SuitePreset(
        name="synthetic-material-appearance",
        source="synthetic",
        description="Synthetic material, UV, PBR-channel, and texture-only detail fixtures for editable Blender output scoring.",
        synthetic_suites=("material-appearance",),
        tags=("synthetic", "appearance", "materials", "uv", "editable"),
    ),
    "synthetic-nightly": SuitePreset(
        name="synthetic-nightly",
        source="synthetic",
        description="Full heavy synthetic suite for broad regression checks.",
        synthetic_suites=("nightly-heavy",),
        tags=("synthetic", "nightly"),
    ),
}
