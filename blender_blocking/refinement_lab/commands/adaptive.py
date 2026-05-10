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
from .common import _run_options_from_args, _runtime_can_execute


def _cmd_loop(args: argparse.Namespace) -> int:
    if not _runtime_can_execute(args):
        return 2
    track = get_track_preset(args.track)
    external_variants = load_variants_from_files(args.variant_file)
    summary = run_adaptive_loop(
        suite=args.suite,
        track=args.track,
        search=args.search or track.default_search,
        objective=args.objective or track.default_objective,
        output_root=args.result_root,
        seed=args.seed,
        max_runs=args.max_runs,
        top_k=args.top_k,
        external_variants=external_variants,
        external_variant_mode=args.variant_file_mode,
        options=AdaptiveLoopOptions(
            generations=args.generations,
            parent_top_k=args.parent_top_k,
            children_per_parent=args.children_per_parent,
            stop_when_no_children=not args.keep_going_without_children,
            run_options=_run_options_from_args(args),
        ),
        base_config=BlockingConfig(),
    )
    print(
        json.dumps(
            {
                "summary": summary.summary_path.as_posix(),
                "generations": len(summary.generations),
                "stopped_reason": summary.stopped_reason,
                "final_child_variants": len(summary.final_child_variants),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def _cmd_adapt(args: argparse.Namespace) -> int:
    proposals = _adaptive_proposals_from_files(
        args.result_json,
        max_proposals=args.max_proposals,
    )
    payload = {
        "schema_version": "refinement_adaptive_proposals_v1",
        "source_result_json": [path.as_posix() for path in args.result_json],
        "max_proposals": args.max_proposals,
        "proposal_count": len(proposals),
        "proposals": [proposal.to_dict() for proposal in proposals],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    variants_path = None
    if args.variants_out is not None:
        variants = [
            proposal.to_variant(parent_variant_id=args.parent_variant).to_dict()
            for proposal in proposals
        ]
        variants_payload = {
            "schema_version": "refinement_adaptive_variants_v1",
            "source_result_json": [path.as_posix() for path in args.result_json],
            "parent_variant_id": args.parent_variant,
            "variant_count": len(variants),
            "variants": variants,
        }
        args.variants_out.parent.mkdir(parents=True, exist_ok=True)
        args.variants_out.write_text(
            json.dumps(json_safe(variants_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        variants_path = args.variants_out.as_posix()

    print(
        json.dumps(
            {
                "proposals": len(proposals),
                "out": args.out.as_posix(),
                "top": proposals[0].title if proposals else None,
                "variants_out": variants_path,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def _adaptive_proposals_from_files(
    paths: list[Path],
    *,
    max_proposals: int,
) -> list[RefinementProposal]:
    proposals: list[RefinementProposal] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        proposals.extend(
            _adaptive_proposals_from_payload(
                payload,
                max_proposals=max_proposals,
            )
        )
    return list(merge_proposals(proposals, max_proposals=max_proposals))

def _adaptive_proposals_from_payload(
    payload: object,
    *,
    max_proposals: int,
) -> list[RefinementProposal]:
    if isinstance(payload, list):
        proposals: list[RefinementProposal] = []
        for item in payload:
            proposals.extend(
                _adaptive_proposals_from_payload(
                    item,
                    max_proposals=max_proposals,
                )
            )
        return proposals
    if not isinstance(payload, dict):
        return []
    return list(proposals_from_result_payload(payload, max_proposals=max_proposals))
