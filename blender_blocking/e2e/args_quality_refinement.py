from __future__ import annotations

import argparse
from pathlib import Path

from blender_blocking.e2e.novel_args import _parse_csv

def add_constraints_quality_args(parser: argparse.ArgumentParser) -> None:
    misc = parser.add_argument_group("constraints and quality")
    misc.add_argument("--constraint-file", action="append", default=None)
    misc.add_argument(
        "--fail-on-hard-constraints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--use-constraints-for-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument("--quality-budget-json", type=str, default=None)
    misc.add_argument("--quality-compare-baseline", type=str, default=None)
    misc.add_argument(
        "--quality-fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--cost-report-json",
        type=Path,
        default=None,
        help="Write a top-level E2E cost report with validation stages and nested backend cost.",
    )
    misc.add_argument(
        "--cost-track-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track peak Python allocation memory for cost stages.",
    )
    misc.add_argument(
        "--cost-fail-max-wall-ms",
        type=float,
        default=None,
        help="Fail the run if combined validation plus backend wall time exceeds this many milliseconds.",
    )
    misc.add_argument(
        "--cost-fail-max-backend-wall-ms",
        type=float,
        default=None,
        help="Fail the run if nested backend reconstruction wall time exceeds this many milliseconds.",
    )
    misc.add_argument(
        "--environment-compatibility",
        choices=("warn", "strict", "ignore"),
        default=None,
    )
    misc.add_argument("--synthetic-suite", type=str, default=None)
    misc.add_argument("--synthetic-seed", type=int, default=None)
    misc.add_argument("--synthetic-output-root", type=str, default=None)
    misc.add_argument(
        "--synthetic-matrix",
        action="store_true",
        help="Run the synthetic suite x reconstruction mode matrix instead of a single sample/custom validation.",
    )
    misc.add_argument(
        "--synthetic-modes",
        type=_parse_csv,
        default=None,
        help="Comma-separated reconstruction modes for --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-count",
        type=int,
        default=None,
        help="Optional synthetic spec count for --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-strict-skips",
        action="store_true",
        help="Treat skipped synthetic matrix rows as failures.",
    )
    misc.add_argument(
        "--quality-report-json",
        type=Path,
        default=None,
        help="Write quality budget report JSON after --synthetic-matrix.",
    )
    misc.add_argument(
        "--synthetic-commit-small-fixtures-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--synthetic-keep-heavy-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    misc.add_argument(
        "--artifact-output-root",
        type=Path,
        default=None,
        help="Write reconstruction backend artifacts under this root.",
    )


    misc.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress bars",
    )


def add_refinement_lab_args(parser: argparse.ArgumentParser) -> None:
    refinement = parser.add_argument_group("refinement lab")
    refinement.add_argument("--refinement-suite", type=str, default=None)
    refinement.add_argument("--refinement-track", type=str, default=None)
    refinement.add_argument(
        "--refinement-search",
        choices=("grid", "random", "coordinate", "successive_halving"),
        default=None,
    )
    refinement.add_argument(
        "--refinement-objective",
        choices=(
            "quality_win",
            "min_view_iou",
            "mean_iou",
            "profile_editable",
            "visual_hull_alignment",
            "fast_preview",
            "human_adjusted",
        ),
        default=None,
    )
    refinement.add_argument("--refinement-result-root", type=Path, default=None)
    refinement.add_argument("--refinement-preset-json", type=Path, default=None)
    refinement.add_argument("--refinement-max-runs", type=int, default=None)
    refinement.add_argument("--refinement-top-k", type=int, default=None)
    refinement.add_argument("--refinement-seed", type=int, default=None)
    refinement.add_argument(
        "--refinement-variant-file",
        type=Path,
        action="append",
        default=None,
        help="Adaptive variant JSON file to append or replace generated variants.",
    )
    refinement.add_argument(
        "--refinement-variant-file-mode",
        choices=("append", "replace"),
        default=None,
    )
    refinement.add_argument(
        "--refinement-html-report", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-write-overlays",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-bounds-debug", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-autopsy", action=argparse.BooleanOptionalAction, default=None
    )
    refinement.add_argument(
        "--refinement-copy-references",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument("--refinement-stop-on-first-error", action="store_true")
    refinement.add_argument(
        "--refinement-fail-on-all-failed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-append-global-index",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    refinement.add_argument(
        "--refinement-report-failures", choices=("top", "all", "none"), default=None
    )
    refinement.add_argument("--refinement-subprocess", action="store_true")
    refinement.add_argument("--refinement-blender-exe", type=str, default=None)
