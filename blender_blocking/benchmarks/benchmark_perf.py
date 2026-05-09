"""
Lightweight performance benchmarks for key pure-Python hotspots.

Run from repo root:
  python blender_blocking/benchmarks/benchmark_perf.py --all
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Ensure blender_blocking is on sys.path
BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.progress import progress_bar


SCHEMA_VERSION = "benchmark_perf_results_v2"


def _now() -> float:
    return time.perf_counter()


def _format_duration(ms: float) -> str:
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f} s"
    return f"{ms:.2f} ms"


def _format_rate(value: float, unit: str) -> str:
    if value <= 0:
        return f"0 {unit}/s"
    scales = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for scale, suffix in scales:
        if value >= scale:
            return f"{value / scale:.2f} {suffix}{unit}/s"
    return f"{value:.2f} {unit}/s"


@dataclass
class BenchResult:
    name: str
    iterations: int
    elapsed_s: float
    per_iter_ms: float
    case: str = "ad-hoc"
    status: str = "ok"
    meta: Dict[str, object] = field(default_factory=dict)
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    benches: Tuple[str, ...]
    description: str
    overrides: Mapping[str, object] = field(default_factory=dict)


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


def _print_result(result: BenchResult) -> None:
    if result.status == "skip":
        print(f"SKIP: {result.name}: {result.skip_reason}")
        print()
        return

    total_ms = result.elapsed_s * 1000.0
    per_iter_ms = result.per_iter_ms
    print(f"OK: {result.name}")
    print(
        f"  total: {_format_duration(total_ms)}  "
        f"({result.iterations} iters, {_format_duration(per_iter_ms)} /iter)"
    )

    meta = dict(result.meta)
    throughput = meta.pop("throughput", None)
    throughput_unit = meta.pop("throughput_unit", None)
    throughput_label = meta.pop("throughput_label", "throughput")

    if throughput is not None and throughput_unit:
        print(
            f"  rate: {_format_rate(float(throughput), str(throughput_unit))} "
            f"({throughput_label})"
        )

    if meta:
        print("  meta:")
        for key in sorted(meta.keys()):
            print(f"    - {key}: {meta[key]}")
    print()


def _make_rect_silhouette(width: int, height: int, ratio: float = 0.4) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rect_w = max(1, int(width * ratio))
    x0 = (width - rect_w) // 2
    mask[:, x0 : x0 + rect_w] = True
    return mask


def _make_circle_silhouette(width: int, height: int, ratio: float = 0.35) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cx = width // 2
    cy = height // 2
    radius = max(1, int(min(width, height) * ratio))
    y, x = np.ogrid[:height, :width]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius**2] = True
    return mask


def _sample_cylinder_points(
    radius: float,
    height: float,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    side_count = int(num_points * 0.7)
    cap_count = num_points - side_count

    theta = rng.uniform(0.0, 2.0 * np.pi, size=side_count)
    z = rng.uniform(-height / 2.0, height / 2.0, size=side_count)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    side = np.column_stack([x, y, z])

    r = np.sqrt(rng.uniform(0.0, radius**2, size=cap_count))
    theta_cap = rng.uniform(0.0, 2.0 * np.pi, size=cap_count)
    x_cap = r * np.cos(theta_cap)
    y_cap = r * np.sin(theta_cap)
    z_cap = rng.choice([-height / 2.0, height / 2.0], size=cap_count)
    caps = np.column_stack([x_cap, y_cap, z_cap])

    return np.vstack([side, caps])


def _sample_cone_points(
    radius_bottom: float,
    radius_top: float,
    height: float,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = rng.uniform(0.0, 1.0, size=num_points)
    z = -height / 2.0 + t * height
    radii = radius_bottom * (1.0 - t) + radius_top * t
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.column_stack([x, y, z])


def _sample_sphere_points(
    radius: float, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=num_points)
    v = rng.uniform(0.0, 1.0, size=num_points)
    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _sample_rotated_cube_points(
    half_extent: float,
    num_points: int,
    rng: np.random.Generator,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    faces = rng.integers(0, 6, size=num_points)
    coords = rng.uniform(-half_extent, half_extent, size=(num_points, 3))
    coords[np.where(faces == 0), 0] = -half_extent
    coords[np.where(faces == 1), 0] = half_extent
    coords[np.where(faces == 2), 1] = -half_extent
    coords[np.where(faces == 3), 1] = half_extent
    coords[np.where(faces == 4), 2] = -half_extent
    coords[np.where(faces == 5), 2] = half_extent

    if rotation is None:
        rotation = np.array(
            [
                [0.8660254, -0.3535534, 0.3535534],
                [0.5, 0.6123724, -0.6123724],
                [0.0, 0.7071068, 0.7071068],
            ]
        )

    return coords @ rotation.T


def bench_visual_hull(
    resolution: int,
    num_views: int,
    include_top: bool,
    repeat: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.multi_view.visual_hull import MultiViewVisualHull
    except Exception as exc:
        return BenchResult(
            name="visual_hull_reconstruct",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if num_views <= 0:
        raise ValueError("num_views must be >= 1")

    total = 0.0
    voxel_count = 0
    steps_per_iter = num_views + (1 if include_top else 0) + 1
    total_steps = repeat * steps_per_iter
    progress_handle = progress_bar(
        total_steps,
        desc="visual_hull",
        enabled=progress,
        mininterval=0.0,
        miniters=1,
    )
    for _ in range(repeat):
        hull = MultiViewVisualHull(
            resolution=resolution,
            bounds_min=np.array([-1.0, -1.0, -1.0]),
            bounds_max=np.array([1.0, 1.0, 1.0]),
        )

        # Lateral views
        for i in range(num_views):
            angle = float(i) * (360.0 / num_views)
            silhouette = _make_rect_silhouette(64, 64, ratio=0.4)
            hull.add_view_from_silhouette(silhouette, angle=angle, view_type="lateral")
            progress_handle.update(1)

        if include_top:
            top = _make_circle_silhouette(64, 64, ratio=0.35)
            hull.add_view_from_silhouette(top, angle=0.0, view_type="top")
            progress_handle.update(1)

        start = _now()
        voxels = hull.reconstruct(verbose=False)
        total += _now() - start
        voxel_count = int(np.sum(voxels))
        progress_handle.update(1)
    progress_handle.close()

    per_iter_s = total / max(repeat, 1)
    per_iter_ms = per_iter_s * 1000.0
    views_total = num_views + (1 if include_top else 0)
    voxels_per_iter = resolution**3 * views_total
    throughput = (voxels_per_iter / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="visual_hull_reconstruct",
        iterations=repeat,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "resolution": resolution,
            "num_views": num_views,
            "include_top": include_top,
            "occupied_voxels": voxel_count,
            "throughput": throughput,
            "throughput_unit": "vox",
            "throughput_label": "voxel-projections",
        },
    )


def bench_surface_voxels(
    iterations: int,
    resolution: int,
    fill_ratio: float,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.multi_view.visual_hull import MultiViewVisualHull
    except Exception as exc:
        return BenchResult(
            name="surface_voxel_extract",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(1337)
    grid = rng.random((resolution, resolution, resolution)) < fill_ratio

    hull = MultiViewVisualHull(resolution=resolution)
    progress_handle = progress_bar(iterations, desc="surface_voxels", enabled=progress)
    start = _now()
    surface = None
    for _ in range(iterations):
        surface = hull._extract_surface_voxels(grid)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (grid.size / per_iter_s) if per_iter_s > 0 else 0.0

    surface_count = int(surface.sum()) if surface is not None else 0
    return BenchResult(
        name="surface_voxel_extract",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "resolution": resolution,
            "fill_ratio": fill_ratio,
            "surface_voxels": surface_count,
            "throughput": throughput,
            "throughput_unit": "vox",
            "throughput_label": "voxel checks",
        },
    )


def bench_canonicalize(
    iterations: int, output_size: int, progress: bool = True
) -> BenchResult:
    try:
        from validation.silhouette_iou import canonicalize_mask
    except Exception as exc:
        return BenchResult(
            name="canonicalize_mask",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    mask = _make_rect_silhouette(96, 128, ratio=0.35)
    progress_handle = progress_bar(iterations, desc="canonicalize", enabled=progress)
    start = _now()
    for _ in range(iterations):
        canonicalize_mask(mask, output_size=output_size, padding_frac=0.1)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="canonicalize_mask",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "output_size": output_size,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_vertical_profile(
    iterations: int,
    image_size: int,
    num_samples: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.shape_matching.profile_extractor import (
            extract_vertical_profile,
        )
    except Exception as exc:
        return BenchResult(
            name="vertical_profile",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    y0 = image_size // 4
    y1 = image_size - y0
    x0 = image_size // 3
    x1 = image_size - x0
    mask[y0:y1, x0:x1] = 255

    progress_handle = progress_bar(
        iterations, desc="vertical_profile", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        extract_vertical_profile(mask, num_samples=num_samples, already_silhouette=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="vertical_profile",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_size": image_size,
            "num_samples": num_samples,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_vertical_width_profile(
    iterations: int,
    image_size: int,
    num_samples: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from geometry.dual_profile import extract_vertical_width_profile_px
    except Exception as exc:
        return BenchResult(
            name="vertical_width_profile",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    x0 = image_size // 3
    x1 = image_size - x0
    mask[:, x0:x1] = 255

    progress_handle = progress_bar(iterations, desc="vertical_width", enabled=progress)
    start = _now()
    for _ in range(iterations):
        extract_vertical_width_profile_px(
            mask,
            num_samples=num_samples,
            sample_policy="endpoints",
            fill_strategy="interp_nearest",
            smoothing_window=1,
        )
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (mask.size / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="vertical_width_profile",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_size": image_size,
            "num_samples": num_samples,
            "fill_strategy": "interp_nearest",
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "input pixels",
        },
    )


def bench_profile_interpolation(
    iterations: int, num_samples: int, progress: bool = True
) -> BenchResult:
    try:
        from placement.primitive_placement import SliceAnalyzer
    except Exception as exc:
        return BenchResult(
            name="profile_interpolation",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    analyzer = SliceAnalyzer(
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(2.0, 2.0, 2.0),
        num_slices=2,
        vertical_profile=[(0.0, 0.2), (0.5, 0.6), (1.0, 1.0)],
    )
    z_values = np.linspace(0.0, 1.0, num_samples)

    total_steps = iterations * len(z_values)
    progress_handle = progress_bar(total_steps, desc="profile_interp", enabled=progress)
    start = _now()
    for _ in range(iterations):
        for z in z_values:
            analyzer._interpolate_profile(float(z))
            progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_samples / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="profile_interpolation",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_samples": num_samples,
            "throughput": throughput,
            "throughput_unit": "samples",
            "throughput_label": "interpolations",
        },
    )


def bench_compare_silhouettes(
    iterations: int, output_size: int, progress: bool = True
) -> BenchResult:
    try:
        from integration.shape_matching.shape_matcher import compare_silhouettes
    except Exception as exc:
        return BenchResult(
            name="compare_silhouettes",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.full((96, 96), 255, dtype=np.uint8)
    image[24:72, 36:60] = 0
    progress_handle = progress_bar(iterations, desc="compare", enabled=progress)
    start = _now()
    for _ in range(iterations):
        compare_silhouettes(image, image, output_size=output_size)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = ((image.size * 2) / per_iter_s) if per_iter_s > 0 else 0.0
    return BenchResult(
        name="compare_silhouettes",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "output_size": output_size,
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "image pixels",
        },
    )


def bench_combine_profiles(
    iterations: int,
    num_profiles: int,
    num_samples: int,
    method: str,
    progress: bool = True,
) -> BenchResult:
    try:
        from integration.shape_matching.mesh_profile_extractor import combine_profiles
    except Exception as exc:
        return BenchResult(
            name="combine_profiles",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    heights = np.linspace(0.0, 1.0, num_samples)
    profiles = []
    for i in range(num_profiles):
        radii = np.linspace(0.5, 1.0 + i * 0.01, num_samples)
        profiles.append([(float(h), float(r)) for h, r in zip(heights, radii)])

    progress_handle = progress_bar(
        iterations, desc="combine_profiles", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        combine_profiles(profiles, method=method)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_profiles * num_samples) / per_iter_s if per_iter_s > 0 else 0.0

    return BenchResult(
        name="combine_profiles",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_profiles": num_profiles,
            "num_samples": num_samples,
            "method": method,
            "throughput": throughput,
            "throughput_unit": "samples",
            "throughput_label": "profile samples",
        },
    )


def bench_slice_metrics(
    iterations: int, num_profiles: int, progress: bool = True
) -> BenchResult:
    try:
        from shape_matching.slice_shape_matcher import SliceBasedShapeMatcher
    except Exception as exc:
        return BenchResult(
            name="slice_metrics",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(2024)
    features1 = rng.random((num_profiles, 4))
    features2 = rng.random((num_profiles, 4))
    matcher = SliceBasedShapeMatcher(num_slices=2)

    steps_per_iter = 4
    progress_handle = progress_bar(
        iterations * steps_per_iter, desc="slice_metrics", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        norm1 = matcher._normalize_features(features1)
        progress_handle.update(1)
        norm2 = matcher._normalize_features(features2)
        progress_handle.update(1)
        matcher._cosine_similarity(norm1, norm2)
        progress_handle.update(1)
        matcher._correlation(features1[:, 0], features2[:, 0])
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_profiles / per_iter_s) if per_iter_s > 0 else 0.0

    return BenchResult(
        name="slice_metrics",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_profiles": num_profiles,
            "throughput": throughput,
            "throughput_unit": "profiles",
            "throughput_label": "feature rows",
        },
    )


def bench_resfit_residual(
    iterations: int,
    num_points: int,
    num_primitives: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_residual",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(4242)
    target_points = rng.normal(size=(num_points, 3))
    primitives = []
    for _ in range(num_primitives):
        position = tuple(rng.normal(size=3))
        orientation = (float(rng.uniform(0.0, 3.14)), float(rng.uniform(0.0, 3.14)))
        radius_bottom = float(rng.uniform(0.5, 2.0))
        radius_top = float(rng.uniform(0.2, 1.5))
        height = float(rng.uniform(0.5, 3.0))
        primitives.append(
            SuperFrustum(
                position=position,
                orientation=orientation,
                radius_bottom=radius_bottom,
                radius_top=radius_top,
                height=height,
            )
        )

    fitter = ResidualFitter()
    progress_handle = progress_bar(iterations, desc="resfit_residual", enabled=progress)
    start = _now()
    for _ in range(iterations):
        fitter.compute_residual_error(primitives, target_points)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (num_points * num_primitives) / per_iter_s if per_iter_s > 0 else 0.0

    return BenchResult(
        name="resfit_residual",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "throughput": throughput,
            "throughput_unit": "evals",
            "throughput_label": "sdf evals",
        },
    )


def bench_resfit_full(
    iterations: int,
    num_points: int,
    num_primitives: int,
    steps: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_full",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    if num_primitives <= 0:
        raise ValueError("num_primitives must be >= 1")
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    rng = np.random.default_rng(2026)
    shapes = [
        (
            "cylinder",
            _sample_cylinder_points(1.5, 3.0, num_points, rng),
        ),
        (
            "cone",
            _sample_cone_points(2.0, 0.5, 3.0, num_points, rng),
        ),
        (
            "rotated_cube",
            _sample_rotated_cube_points(1.0, num_points, rng),
        ),
        (
            "sphere",
            _sample_sphere_points(1.25, num_points, rng),
        ),
    ]

    steps_per_shape = steps + 1
    total_steps = iterations * len(shapes) * steps_per_shape
    progress_handle = progress_bar(total_steps, desc="resfit_full", enabled=progress)

    start = _now()
    for _ in range(iterations):
        for _, target_points in shapes:
            primitives = []
            for _ in range(num_primitives):
                position = tuple(rng.normal(scale=0.2, size=3))
                orientation = (
                    float(rng.uniform(0.0, 3.14)),
                    float(rng.uniform(0.0, 3.14)),
                )
                radius_bottom = float(rng.uniform(0.7, 1.8))
                radius_top = float(rng.uniform(0.4, 1.4))
                height = float(rng.uniform(1.0, 3.5))
                primitives.append(
                    SuperFrustum(
                        position=position,
                        orientation=orientation,
                        radius_bottom=radius_bottom,
                        radius_top=radius_top,
                        height=height,
                    )
                )

            fitter = ResidualFitter(
                learning_rate=0.01,
                optimization_steps=steps,
            )
            fitter.optimize_primitives(
                primitives,
                target_points,
                steps=steps,
                log_progress=progress,
                progress_callback=progress_handle.update,
            )
            fitter.compute_residual_error(primitives, target_points)
            progress_handle.update(1)
    progress_handle.close()

    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    step_units = iterations * len(shapes) * steps
    throughput = (step_units / total) if total > 0 else 0.0

    return BenchResult(
        name="resfit_full",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "steps": steps,
            "shape_count": len(shapes),
            "throughput": throughput,
            "throughput_unit": "steps",
            "throughput_label": "opt steps",
        },
    )


def bench_resfit_optimize(
    iterations: int,
    num_points: int,
    num_primitives: int,
    steps: int,
    progress: bool = True,
) -> BenchResult:
    try:
        from placement.resfitting import ResidualFitter
        from primitives.superfrustum import SuperFrustum
    except Exception as exc:
        return BenchResult(
            name="resfit_optimize",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    if iterations <= 0:
        raise ValueError("iterations must be >= 1")
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    if num_primitives <= 0:
        raise ValueError("num_primitives must be >= 1")
    if steps <= 0:
        raise ValueError("steps must be >= 1")

    rng = np.random.default_rng(9001)
    target_points = rng.normal(size=(num_points, 3))

    total_steps = iterations * steps
    progress_handle = progress_bar(
        total_steps, desc="resfit_optimize", enabled=progress
    )
    start = _now()
    for _ in range(iterations):
        primitives = []
        for _ in range(num_primitives):
            position = tuple(rng.normal(scale=0.2, size=3))
            orientation = (
                float(rng.uniform(0.0, 3.14)),
                float(rng.uniform(0.0, 3.14)),
            )
            radius_bottom = float(rng.uniform(0.7, 1.8))
            radius_top = float(rng.uniform(0.4, 1.4))
            height = float(rng.uniform(1.0, 3.5))
            primitives.append(
                SuperFrustum(
                    position=position,
                    orientation=orientation,
                    radius_bottom=radius_bottom,
                    radius_top=radius_top,
                    height=height,
                )
            )

        fitter = ResidualFitter(learning_rate=0.01, optimization_steps=steps)
        fitter.optimize_primitives(
            primitives,
            target_points,
            steps=steps,
            log_progress=progress,
            progress_callback=progress_handle.update,
        )
    progress_handle.close()

    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (iterations * steps / total) if total > 0 else 0.0

    return BenchResult(
        name="resfit_optimize",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "num_points": num_points,
            "num_primitives": num_primitives,
            "steps": steps,
            "throughput": throughput,
            "throughput_unit": "steps",
            "throughput_label": "opt steps",
        },
    )


def bench_extract_silhouette(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from geometry.silhouette import extract_binary_silhouette
    except Exception as exc:
        return BenchResult(
            name="extract_binary_silhouette",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.zeros((96, 96, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[16:80, 32:64, 3] = 255

    progress_handle = progress_bar(iterations, desc="extract", enabled=progress)
    start = _now()
    for _ in range(iterations):
        extract_binary_silhouette(image, prefer_alpha=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    per_iter_ms = per_iter_s * 1000.0
    throughput = (
        (image.shape[0] * image.shape[1] / per_iter_s) if per_iter_s > 0 else 0.0
    )
    return BenchResult(
        name="extract_binary_silhouette",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_ms,
        meta={
            "image_shape": list(image.shape),
            "throughput": throughput,
            "throughput_unit": "px",
            "throughput_label": "rgba pixels",
        },
    )


def bench_silhouette_pipeline(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from geometry.silhouette_pipeline import (
            build_uncertain_mask,
            canonicalize_silhouette,
            extract_silhouette_candidates,
        )
    except Exception as exc:
        return BenchResult(
            name="silhouette_pipeline",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    image = np.full((128, 128, 4), 255, dtype=np.uint8)
    image[:, :, 3] = 255
    image[28:110, 38:91, :3] = 0
    progress_handle = progress_bar(iterations, desc="silhouette_pipeline", enabled=progress)
    start = _now()
    candidate_count = 0
    for _ in range(iterations):
        candidates = extract_silhouette_candidates(image)
        uncertain = build_uncertain_mask(image)
        canonicalize_silhouette(uncertain.hard_mask, output_size=256)
        candidate_count = len(candidates)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="silhouette_pipeline",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "image_shape": list(image.shape),
            "candidate_count": candidate_count,
            "throughput": image.shape[0] * image.shape[1] / per_iter_s
            if per_iter_s > 0
            else 0.0,
            "throughput_unit": "px",
            "throughput_label": "pipeline pixels",
        },
    )


def bench_volume_surface_new(
    iterations: int,
    resolution: int,
    fill_ratio: float,
    progress: bool = True,
) -> BenchResult:
    try:
        from volume import extract_surface_voxels
    except Exception as exc:
        return BenchResult(
            name="volume_surface_vectorized",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    rng = np.random.default_rng(424242)
    grid = rng.random((resolution, resolution, resolution)) < fill_ratio
    progress_handle = progress_bar(iterations, desc="volume_surface", enabled=progress)
    start = _now()
    surface = None
    for _ in range(iterations):
        surface = extract_surface_voxels(grid, prefer_scipy=True)
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="volume_surface_vectorized",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "resolution": resolution,
            "fill_ratio": fill_ratio,
            "surface_voxels": int(surface.sum()) if surface is not None else 0,
            "throughput": grid.size / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "vox",
            "throughput_label": "surface checks",
        },
    )


def bench_target_builder(iterations: int, progress: bool = True) -> BenchResult:
    try:
        from config import BlockingConfig
        from reconstruction.target_builder import build_target_from_images
    except Exception as exc:
        return BenchResult(
            name="target_builder",
            iterations=0,
            elapsed_s=0.0,
            per_iter_ms=0.0,
            status="skip",
            skip_reason=str(exc),
        )

    cfg = BlockingConfig()
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    image[20:82, 30:68, :] = 0
    views = {"front": image, "side": image}
    progress_handle = progress_bar(iterations, desc="target_builder", enabled=progress)
    start = _now()
    for _ in range(iterations):
        build_target_from_images(
            views,
            config=cfg,
            bounds_minmax=((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
            profile_samples=32,
        )
        progress_handle.update(1)
    progress_handle.close()
    total = _now() - start
    per_iter_s = total / max(iterations, 1)
    return BenchResult(
        name="target_builder",
        iterations=iterations,
        elapsed_s=total,
        per_iter_ms=per_iter_s * 1000.0,
        meta={
            "views": 2,
            "profile_samples": 32,
            "throughput": image.size * 2 / per_iter_s if per_iter_s > 0 else 0.0,
            "throughput_unit": "px",
            "throughput_label": "target pixels",
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment_payload() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _result_payload(result: BenchResult) -> dict[str, object]:
    payload = asdict(result)
    metrics = {
        "iterations": result.iterations,
        "elapsed_s": result.elapsed_s,
        "per_iter_ms": result.per_iter_ms,
        "throughput": result.meta.get("throughput"),
        "status_ok": 1.0 if result.status == "ok" else 0.0,
    }
    payload["metrics"] = metrics
    return payload


def _results_payload(
    results: Sequence[BenchResult],
    *,
    budget_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    cases = sorted({result.case for result in results})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "environment": _environment_payload(),
        "cases": cases,
        "results": [_result_payload(result) for result in results],
    }
    if budget_report is not None:
        payload["budget_report"] = dict(budget_report)
    return payload


def _write_json(
    path: Path,
    results: Sequence[BenchResult],
    *,
    budget_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    payload = _results_payload(results, budget_report=budget_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_budget_report(
    *,
    current_payload: Mapping[str, Any],
    current_path: Optional[Path],
    budget_path: Optional[str],
    baseline_path: Optional[str],
    report_path: Optional[str],
) -> Optional[dict[str, object]]:
    if not budget_path:
        return None
    try:
        from scripts.quality_budget import evaluate_budget_payloads, write_report
    except Exception as exc:
        return {
            "passed": False,
            "error": f"failed to import scripts.quality_budget: {exc}",
        }

    baseline_payload = None
    if baseline_path:
        baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    report = evaluate_budget_payloads(
        current_payload,
        json.loads(Path(budget_path).read_text(encoding="utf-8")),
        baseline_payload=baseline_payload,
        artifact_path=str(current_path) if current_path else None,
        baseline_path=baseline_path,
    )
    if report_path:
        write_report(Path(report_path), report)
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performance micro-benchmarks")
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(BENCHMARK_CASES),
        help="Named benchmark case. Can be repeated. Overrides --bench unless --bench is also provided with --case-benches.",
    )
    parser.add_argument(
        "--case-benches",
        action="store_true",
        help="Run --bench selections inside each named --case instead of the case registry benches.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print benchmark case registry as JSON and exit.",
    )
    parser.add_argument(
        "--bench",
        default="all",
        help=(
            "Comma-separated list: visual_hull,surface_voxels,vertical_profile,"
            "vertical_width_profile,profile_interpolation,combine_profiles,slice_metrics,"
            "resfit_residual,resfit_full,resfit_optimize,canonicalize,compare,extract,"
            "silhouette_pipeline,volume_surface,target_builder"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Run all benches")
    parser.add_argument("--repeat", type=int, default=1, help="Repeats for visual hull")
    parser.add_argument(
        "--iterations", type=int, default=200, help="Iterations for micro-benches"
    )
    parser.add_argument(
        "--resolution", type=int, default=32, help="Voxel resolution for visual hull"
    )
    parser.add_argument(
        "--num-views", type=int, default=8, help="Lateral view count for visual hull"
    )
    parser.add_argument(
        "--include-top", action="store_true", help="Include top view in visual hull"
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=0.25,
        help="Fill ratio for surface voxel benchmark",
    )
    parser.add_argument(
        "--output-size", type=int, default=256, help="Canonical size for masks"
    )
    parser.add_argument(
        "--profile-size",
        type=int,
        default=128,
        help="Image size for vertical profile benchmark",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=100,
        help="Samples for vertical profile and profile-combine benches",
    )
    parser.add_argument(
        "--combine-profiles",
        type=int,
        default=12,
        help="Number of profiles for combine_profiles benchmark",
    )
    parser.add_argument(
        "--combine-method",
        type=str,
        default="median",
        help="Method for combine_profiles benchmark",
    )
    parser.add_argument(
        "--slice-profiles",
        type=int,
        default=64,
        help="Profile count for slice metrics benchmark",
    )
    parser.add_argument(
        "--resfit-points",
        type=int,
        default=1000,
        help="Point count for resfit residual benchmark",
    )
    parser.add_argument(
        "--resfit-primitives",
        type=int,
        default=5,
        help="Primitive count for resfit residual benchmark",
    )
    parser.add_argument(
        "--resfit-full-steps",
        type=int,
        default=5,
        help="Optimization steps for resfit_full benchmark",
    )
    parser.add_argument(
        "--resfit-full-iterations",
        type=int,
        default=1,
        help="Iteration count for resfit_full benchmark",
    )
    parser.add_argument(
        "--resfit-opt-steps",
        type=int,
        default=5,
        help="Optimization steps for resfit_optimize benchmark",
    )
    parser.add_argument(
        "--resfit-opt-iterations",
        type=int,
        default=1,
        help="Iteration count for resfit_optimize benchmark",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress bars",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--json", type=str, default=None, help="Write results to JSON")
    parser.add_argument(
        "--budget-json",
        type=str,
        default=None,
        help="Quality/perf budget JSON to evaluate against the result payload.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Previous benchmark JSON used for regression comparison.",
    )
    parser.add_argument(
        "--budget-report-json",
        type=str,
        default=None,
        help="Write standalone budget comparison report JSON.",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="Return exit code 1 when absolute budget thresholds fail.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 1 when baseline comparisons fail.",
    )
    return parser.parse_args(argv)


def _default_benches(args: argparse.Namespace) -> List[str]:
    if args.all:
        return [
            "visual_hull",
            "surface_voxels",
            "vertical_profile",
            "vertical_width_profile",
            "profile_interpolation",
            "combine_profiles",
            "slice_metrics",
            "resfit_residual",
            "resfit_optimize",
            "canonicalize",
            "compare",
            "extract",
            "silhouette_pipeline",
            "volume_surface",
            "target_builder",
        ]
    return [b.strip() for b in args.bench.split(",") if b.strip()]


def _args_for_case(args: argparse.Namespace, case: BenchmarkCase) -> argparse.Namespace:
    case_args = argparse.Namespace(**vars(args))
    for key, value in case.overrides.items():
        setattr(case_args, key, value)
    return case_args


def _run_one_benchmark(bench: str, args: argparse.Namespace) -> BenchResult:
    if bench == "visual_hull":
        return bench_visual_hull(
            resolution=args.resolution,
            num_views=args.num_views,
            include_top=args.include_top,
            repeat=args.repeat,
            progress=args.progress,
        )
    if bench == "surface_voxels":
        return bench_surface_voxels(
            iterations=args.iterations,
            resolution=args.resolution,
            fill_ratio=args.fill_ratio,
            progress=args.progress,
        )
    if bench == "vertical_profile":
        return bench_vertical_profile(
            iterations=args.iterations,
            image_size=args.profile_size,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "vertical_width_profile":
        return bench_vertical_width_profile(
            iterations=args.iterations,
            image_size=args.profile_size,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "profile_interpolation":
        return bench_profile_interpolation(
            iterations=args.iterations,
            num_samples=args.profile_samples,
            progress=args.progress,
        )
    if bench == "combine_profiles":
        return bench_combine_profiles(
            iterations=args.iterations,
            num_profiles=args.combine_profiles,
            num_samples=args.profile_samples,
            method=args.combine_method,
            progress=args.progress,
        )
    if bench == "slice_metrics":
        return bench_slice_metrics(
            iterations=args.iterations,
            num_profiles=args.slice_profiles,
            progress=args.progress,
        )
    if bench == "resfit_residual":
        return bench_resfit_residual(
            iterations=args.iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            progress=args.progress,
        )
    if bench == "resfit_full":
        return bench_resfit_full(
            iterations=args.resfit_full_iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            steps=args.resfit_full_steps,
            progress=args.progress,
        )
    if bench == "resfit_optimize":
        return bench_resfit_optimize(
            iterations=args.resfit_opt_iterations,
            num_points=args.resfit_points,
            num_primitives=args.resfit_primitives,
            steps=args.resfit_opt_steps,
            progress=args.progress,
        )
    if bench == "canonicalize":
        return bench_canonicalize(
            iterations=args.iterations,
            output_size=args.output_size,
            progress=args.progress,
        )
    if bench == "compare":
        return bench_compare_silhouettes(
            iterations=args.iterations,
            output_size=args.output_size,
            progress=args.progress,
        )
    if bench == "extract":
        return bench_extract_silhouette(
            iterations=args.iterations,
            progress=args.progress,
        )
    if bench == "silhouette_pipeline":
        return bench_silhouette_pipeline(
            iterations=args.iterations,
            progress=args.progress,
        )
    if bench == "volume_surface":
        return bench_volume_surface_new(
            iterations=args.iterations,
            resolution=args.resolution,
            fill_ratio=args.fill_ratio,
            progress=args.progress,
        )
    if bench == "target_builder":
        return bench_target_builder(
            iterations=args.iterations,
            progress=args.progress,
        )
    return BenchResult(
        name=bench,
        iterations=0,
        elapsed_s=0.0,
        per_iter_ms=0.0,
        status="skip",
        skip_reason="Unknown benchmark",
    )


def _run_case(case_name: str, args: argparse.Namespace) -> List[BenchResult]:
    case = BENCHMARK_CASES[case_name]
    case_args = _args_for_case(args, case)
    benches = _default_benches(args) if args.case_benches else list(case.benches)
    results = []
    print(f"\n=== Benchmark case: {case.name} ===")
    print(case.description)
    for bench in benches:
        result = _run_one_benchmark(bench, case_args)
        result.case = case.name
        results.append(result)
        _print_result(result)
    return results


def _print_case_registry() -> None:
    payload = {
        name: {
            "description": case.description,
            "benches": list(case.benches),
            "overrides": dict(case.overrides),
        }
        for name, case in sorted(BENCHMARK_CASES.items())
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.list_cases:
        _print_case_registry()
        return 0

    results: List[BenchResult] = []
    if args.case:
        for case_name in args.case:
            results.extend(_run_case(case_name, args))
    else:
        for bench in _default_benches(args):
            result = _run_one_benchmark(bench, args)
            results.append(result)
            _print_result(result)

    json_path = Path(args.json) if args.json else None
    base_payload = _results_payload(results)
    budget_report = _load_budget_report(
        current_payload=base_payload,
        current_path=json_path,
        budget_path=args.budget_json,
        baseline_path=args.baseline_json,
        report_path=args.budget_report_json,
    )

    if args.json:
        _write_json(Path(args.json), results, budget_report=budget_report)
        print(f"Wrote JSON results to: {args.json}")

    if budget_report:
        passed = bool(budget_report.get("passed", False))
        comparison_passed = bool(budget_report.get("comparison_passed", passed))
        threshold_passed = bool(budget_report.get("threshold_passed", passed))
        print(
            "Budget result: "
            f"{'PASS' if passed else 'FAIL'} "
            f"(thresholds={'PASS' if threshold_passed else 'FAIL'}, "
            f"comparisons={'PASS' if comparison_passed else 'FAIL'})"
        )
        if (args.fail_on_budget and not threshold_passed) or (
            args.fail_on_regression and not comparison_passed
        ):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
