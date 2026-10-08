from __future__ import annotations

import argparse
from typing import Optional

from blender_blocking.e2e.args_groups import add_e2e_argument_groups
from blender_blocking.e2e.constants import (
    ALL_RECONSTRUCTION_MODES,
    DEFAULT_ENSEMBLE_CANDIDATES,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate blendslop reconstruction modes from reference silhouettes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode legacy --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode loft_profile --mesh-radial-segments 32 --profile-samples 120 --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode visual_hull_voxel --validation-mode backend-status --vh-resolution 32 --vh-backend chunked --vh-chunk-size 16 --vh-mesh-method points --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode legacy --validation-mode novel-view --novel-view-angles 45,135 --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode ensemble --validation-mode backend-status --ensemble-candidates visual_hull_voxel,primitive_fit_refine,gaussian_ellipsoid_proxy --primitive-max 6 --gaussian-count 8 --no-progress
  python blender_blocking/test_e2e_validation.py --reconstruction-mode ensemble --print-config --dry-run

Modes:
  {", ".join(ALL_RECONSTRUCTION_MODES)}

Default ensemble:
  {", ".join(DEFAULT_ENSEMBLE_CANDIDATES)}
""",
    )
    add_e2e_argument_groups(parser)
    return parser.parse_args(argv)
