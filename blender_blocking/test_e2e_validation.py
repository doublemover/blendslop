# ruff: noqa: E402,F401
"""Compatibility entry point for Blendslop E2E validation.

Business logic lives in :mod:`blender_blocking.e2e`; this file remains the
Blender ``--python`` and custom test-runner entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from blender_blocking.e2e import (  # noqa: E402
    ALL_RECONSTRUCTION_MODES,
    BACKEND_MODES,
    DEFAULT_ENSEMBLE_CANDIDATES,
    DEFAULT_SYNTHETIC_MATRIX_MODES,
    E2EValidator,
    RENDER_IOU_MODES,
    VALIDATION_MODES,
    main,
    run_synthetic_suite_matrix,
    test_with_custom_images,
    test_with_sample_images,
)
from blender_blocking.e2e.cli_args import (  # noqa: E402
    _apply_cli_args,
    _apply_overrides,
    _derive_config_label,
    _novel_threshold,
    _novel_view_names_from_args,
    _parse_args,
    _parse_csv,
    _parse_resolution,
    _parse_view_reference_entries,
    _resolve_novel_view_inputs,
)
from blender_blocking.e2e.cost import (  # noqa: E402
    _backend_cost_report_from_payload,
    _cost_gate_report,
    _cost_payload,
    _matrix_cost_summary,
)
from blender_blocking.e2e.matrix import _matrix_metrics  # noqa: E402
from blender_blocking.e2e.novel_view import (  # noqa: E402
    _aggregate_novel_reports,
    _novel_view_gate,
)
from blender_blocking.e2e.payloads import _render_filename_prefix  # noqa: E402
from blender_blocking.e2e.refinement_bridge import (  # noqa: E402
    _build_refinement_plan_from_args,
    _refinement_requested,
)

if __name__ == "__main__":
    raise SystemExit(main())
