from __future__ import annotations

import argparse

from .args_backends import (
    add_differentiable_refine_args,
    add_ensemble_args,
    add_gaussian_proxy_args,
    add_primitive_fit_args,
    add_shape_program_args,
    add_visual_hull_args,
)
from .args_core import add_core_args
from .args_quality_refinement import add_constraints_quality_args, add_refinement_lab_args
from .args_rendering import (
    add_canonicalization_args,
    add_mesh_join_args,
    add_profile_args,
    add_render_args,
    add_silhouette_args,
)


def add_e2e_argument_groups(parser: argparse.ArgumentParser) -> None:
    add_core_args(parser)
    add_render_args(parser)
    add_profile_args(parser)
    add_silhouette_args(parser)
    add_canonicalization_args(parser)
    add_mesh_join_args(parser)
    add_visual_hull_args(parser)
    add_primitive_fit_args(parser)
    add_gaussian_proxy_args(parser)
    add_differentiable_refine_args(parser)
    add_shape_program_args(parser)
    add_ensemble_args(parser)
    add_constraints_quality_args(parser)
    add_refinement_lab_args(parser)
