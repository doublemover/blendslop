from __future__ import annotations

from typing import Dict

from ..contracts import BenchmarkCase


BENCHMARK_CASES: Dict[str, BenchmarkCase] = {
    "smoke": BenchmarkCase(
        name="smoke",
        description="Short pure-Python hotspot coverage for local sanity checks.",
        benches=(
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
        ),
        overrides={"iterations": 10, "resolution": 16, "profile_size": 64},
    ),
    "quality-smoke": BenchmarkCase(
        name="quality-smoke",
        description="Budget-gated synthetic smoke companion benchmarks.",
        benches=(
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
            "geometry_metrics",
            "shape_program_build",
        ),
        overrides={"iterations": 20, "resolution": 24, "profile_size": 96},
    ),
    "ci": BenchmarkCase(
        name="ci",
        description="CI-sized benchmark pass with visual hull and profile coverage.",
        benches=(
            "visual_hull",
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
            "vertical_profile",
            "vertical_width_profile",
            "geometry_metrics",
            "shape_program_build",
        ),
        overrides={"iterations": 50, "repeat": 1, "resolution": 32},
    ),
    "nightly": BenchmarkCase(
        name="nightly",
        description="Broad overnight perf coverage including residual fitting.",
        benches=(
            "visual_hull",
            "surface_voxels",
            "vertical_profile",
            "vertical_width_profile",
            "profile_interpolation",
            "combine_profiles",
            "slice_metrics",
            "resfit_residual",
            "resfit_full",
            "resfit_optimize",
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
            "geometry_metrics",
            "shape_program_build",
        ),
        overrides={
            "iterations": 200,
            "repeat": 2,
            "resolution": 48,
            "resfit_full_iterations": 1,
            "resfit_opt_iterations": 2,
        },
    ),
}
