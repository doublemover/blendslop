from __future__ import annotations

from .cases.registry import BENCHMARK_CASES
from .cli import _parse_args, main
from .contracts import BenchResult, BenchmarkCase, SCHEMA_VERSION
from .results import (
    _environment_payload,
    _format_duration,
    _format_rate,
    _print_result,
    _result_payload,
    _results_payload,
    _utc_now,
    _write_json,
)
from .runner import (
    _args_for_case,
    _default_benches,
    _print_case_registry,
    _run_case,
    _run_one_benchmark,
)
from .workloads.geometry import bench_geometry_metrics
from .workloads.profiles import (
    bench_combine_profiles,
    bench_profile_interpolation,
    bench_slice_metrics,
    bench_vertical_profile,
    bench_vertical_width_profile,
)
from .workloads.resfit import bench_resfit_full, bench_resfit_optimize, bench_resfit_residual
from .workloads.shape_program import bench_shape_program_build
from .workloads.silhouettes import (
    bench_canonicalize,
    bench_compare_silhouettes,
    bench_extract_silhouette,
    bench_silhouette_pipeline,
)
from .workloads.target_builder import bench_target_builder
from .workloads.volume import bench_surface_voxels, bench_visual_hull, bench_volume_surface_new

__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    raise SystemExit(main())
