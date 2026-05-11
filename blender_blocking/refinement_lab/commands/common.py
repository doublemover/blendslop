from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..adaptive import RefinementProposal, merge_proposals, proposals_from_result_payload
from ..adaptive_loop import AdaptiveLoopOptions, run_adaptive_loop
from ..artifact_report import ReportOptions, generate_report
from ..candidate_autopsy import write_autopsy
from ..contracts import json_safe
from ..editability_study import build_editability_study_pack, load_review_rows, summarize_review_rows, write_editability_study_pack
from ..human_labels import HumanLabel, append_label
from ..matrix import build_experiment_plan, load_variants_from_files, write_plan
from ..parameter_search import promotion_decision
from ..preset_catalog import get_suite_preset, get_track_preset, list_suites, list_tracks
from ..result_index import load_index, write_leaderboard_json, write_leaderboard_md
from ..runner import RunOptions, runner_for_plan
from ..surrogate import surrogate_report

try:
    from blender_blocking.config import BlockingConfig
except ImportError:  # pragma: no cover
    from config import BlockingConfig


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="default-vase")
    parser.add_argument("--track", default="profile-loft-refinement")
    parser.add_argument("--search", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument(
        "--result-root", type=Path, default=Path("temp/refinement-runs/adhoc")
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--case-count",
        type=int,
        default=None,
        help="Limit synthetic suite cases for bounded smoke and diagnostic runs.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--variant-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--variant-file-mode",
        choices=("append", "replace"),
        default="append",
    )

def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--html-report", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--write-overlays", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--bounds-debug", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--autopsy", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fail-on-all-failed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--append-global-index", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--adaptive-proposals", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lineage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--adaptive-max-proposals", type=int, default=12)
    parser.add_argument(
        "--report-failures", choices=("top", "all", "none"), default="top"
    )
    parser.add_argument("--blender-exe", default=None)

def _runtime_can_execute(args: argparse.Namespace) -> bool:
    if args.blender_exe is not None:
        return True
    try:
        import bpy  # noqa: F401
    except Exception:
        print(
            "ERROR: --blender-exe is required when running outside Blender.",
            file=sys.stderr,
        )
        return False
    return True

def _run_options_from_args(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        html_report=args.html_report,
        write_overlays=args.write_overlays,
        write_bounds_debug=args.bounds_debug,
        write_autopsy=args.autopsy,
        fail_on_all_failed=args.fail_on_all_failed,
        append_global_index=args.append_global_index,
        report_failures=args.report_failures,
        write_adaptive_proposals=args.adaptive_proposals,
        write_lineage=args.lineage,
        adaptive_max_proposals=args.adaptive_max_proposals,
        subprocess_blender=args.blender_exe is not None,
        blender_executable=args.blender_exe,
    )
