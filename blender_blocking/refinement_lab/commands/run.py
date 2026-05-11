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
from .common import _resolve_repo_path, _run_options_from_args, _runtime_can_execute


def _cmd_plan(args: argparse.Namespace) -> int:
    track = get_track_preset(args.track)
    external_variants = load_variants_from_files(args.variant_file)
    plan = build_experiment_plan(
        suite=args.suite,
        track=args.track,
        search=args.search or track.default_search,
        objective=args.objective or track.default_objective,
        output_root=_resolve_repo_path(args.result_root),
        seed=args.seed,
        case_count=args.case_count,
        max_runs=args.max_runs,
        top_k=args.top_k,
        external_variants=external_variants,
        external_variant_mode=args.variant_file_mode,
    )
    write_plan(plan, args.out)
    print(
        json.dumps(
            {
                "plan": args.out.as_posix(),
                "cases": len(plan.cases),
                "variants": len(plan.variants),
                "external_variants": len(external_variants),
                "variant_file_mode": args.variant_file_mode,
            },
            indent=2,
        )
    )
    return 0

def _cmd_run(args: argparse.Namespace) -> int:
    track = get_track_preset(args.track)
    external_variants = load_variants_from_files(args.variant_file)
    plan = build_experiment_plan(
        suite=args.suite,
        track=args.track,
        search=args.search or track.default_search,
        objective=args.objective or track.default_objective,
        output_root=_resolve_repo_path(args.result_root),
        seed=args.seed,
        case_count=args.case_count,
        max_runs=args.max_runs,
        top_k=args.top_k,
        external_variants=external_variants,
        external_variant_mode=args.variant_file_mode,
    )
    if args.command_only:
        for variant in plan.variants:
            print(" ".join(variant.cli_args))
        return 0
    if not _runtime_can_execute(args):
        return 2
    options = _run_options_from_args(args)
    ok, _results = runner_for_plan(
        plan, options=options, base_config=BlockingConfig()
    ).run()
    return 0 if ok else 1
