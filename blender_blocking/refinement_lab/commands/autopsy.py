from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..adaptive import RefinementProposal, merge_proposals, proposals_from_result_payload
from ..adaptive_loop import AdaptiveLoopOptions, run_adaptive_loop
from ..artifact_report import ReportOptions, generate_report
from ..candidate_autopsy import write_autopsy
from ..contracts import compact_path_segment, json_safe
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


def _cmd_autopsy(args: argparse.Namespace) -> int:
    results, _malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    selected = [
        result
        for result in results
        if args.variant is None or result.variant_id == args.variant
    ]
    for result in selected:
        path = (
            args.run_root
            / "c"
            / compact_path_segment(result.case_id, max_length=32, fallback="case")
            / "v"
            / compact_path_segment(result.variant_id, max_length=40, fallback="variant")
            / "autopsy.json"
        )
        write_autopsy(
            result, path, run_root=args.run_root, bounds_debug=result.bounds_debug
        )
        print(path.as_posix())
    return 0 if selected else 1
