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


def _cmd_label(args: argparse.Namespace) -> int:
    label = HumanLabel(
        run_id=args.run_id,
        case_id=args.case_id,
        variant_id=args.variant,
        result_json=args.result_json,
        label=args.label,
        score=args.score,
        tags=tuple(args.tag or ()),
        notes=args.notes,
    )
    append_label(args.run_root / "human-labels.jsonl", label)
    print(json.dumps(label.to_dict(), indent=2, sort_keys=True))
    return 0

def _cmd_study_pack(args: argparse.Namespace) -> int:
    results, malformed = load_index(
        args.run_root / "index.jsonl",
        run_root=args.run_root,
    )
    if malformed:
        print(f"WARN: skipped {malformed} malformed index rows", file=sys.stderr)
    pack = build_editability_study_pack(
        results,
        run_root=args.run_root,
        objective=args.objective,
        top_k=args.top_k,
        include_failed=args.include_failed,
    )
    output_dir = args.out or (args.run_root / "editability-study")
    paths = write_editability_study_pack(pack, output_dir)
    print(json.dumps({key: path.as_posix() for key, path in paths.items()}, indent=2))
    return 0

def _cmd_score_study(args: argparse.Namespace) -> int:
    rows = load_review_rows(args.review_jsonl)
    summary = summarize_review_rows(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(args.out.as_posix())
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0

def _cmd_promote(args: argparse.Namespace) -> int:
    results, _malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    for result in results:
        if result.variant_id == args.variant:
            promotion = promotion_decision(result)
            if not promotion.promotable and not args.allow_review_required:
                blockers = ", ".join(promotion.blockers) or promotion.tier
                print(
                    (
                        f"variant {args.variant!r} is {promotion.tier} and cannot be "
                        f"promoted without --allow-review-required: {blockers}"
                    ),
                    file=sys.stderr,
                )
                return 2
            payload = {
                "schema_version": "refinement_preset_v1",
                "source_run_id": result.run_id,
                "source_variant_id": result.variant_id,
                "case_id": result.case_id,
                "mode": result.mode,
                "metrics": result.metrics,
                "command": list(result.command),
                "score": result.score,
                "promotion": promotion.to_dict(),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(args.out.as_posix())
            return 0
    print(f"variant not found: {args.variant}", file=sys.stderr)
    return 1
