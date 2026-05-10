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


def _cmd_report(args: argparse.Namespace) -> int:
    results, malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    if malformed:
        print(f"WARN: skipped {malformed} malformed index rows")
    path = generate_report(
        run_root=args.run_root,
        results=results,
        objective=args.objective,
        options=ReportOptions(),
    )
    print(path.as_posix())
    return 0

def _cmd_rank(args: argparse.Namespace) -> int:
    results, malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    if malformed:
        print(f"WARN: skipped {malformed} malformed index rows")
    write_leaderboard_json(
        results, args.run_root / "leaderboard.json", objective=args.objective
    )
    write_leaderboard_md(
        results, args.run_root / "leaderboard.md", objective=args.objective
    )
    print((args.run_root / "leaderboard.md").as_posix())
    return 0
