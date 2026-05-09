"""Command line entry point for the reconstruction refinement lab."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[1]
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))

from .adaptive_planner import RefinementProposal, proposals_from_result_payload
from .artifact_report import ReportOptions, generate_report
from .candidate_autopsy import write_autopsy
from .contracts import json_safe
from .human_labels import HumanLabel, append_label
from .matrix import build_experiment_plan, write_plan
from .presets import get_suite_preset, get_track_preset, list_suites, list_tracks
from .result_index import load_index, write_leaderboard_json, write_leaderboard_md
from .runner import RunOptions, runner_for_plan

try:
    from blender_blocking.config import BlockingConfig
except ImportError:  # pragma: no cover
    from config import BlockingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blender_blocking.refinement_lab.cli"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-suites")
    subparsers.add_parser("list-tracks")

    plan_parser = subparsers.add_parser("plan")
    _add_plan_args(plan_parser)
    plan_parser.add_argument("--out", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    _add_plan_args(run_parser)
    _add_run_args(run_parser)
    run_parser.add_argument("--command-only", action="store_true")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument("--objective", default="quality_win")

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--run-root", type=Path, required=True)
    rank_parser.add_argument("--objective", default="quality_win")
    rank_parser.add_argument("--top-k", type=int, default=10)

    autopsy_parser = subparsers.add_parser("autopsy")
    autopsy_parser.add_argument("--run-root", type=Path, required=True)
    autopsy_parser.add_argument("--variant", default=None)

    adapt_parser = subparsers.add_parser("adapt")
    adapt_parser.add_argument(
        "--result-json", type=Path, action="append", required=True
    )
    adapt_parser.add_argument("--out", type=Path, required=True)
    adapt_parser.add_argument("--variants-out", type=Path, default=None)
    adapt_parser.add_argument("--parent-variant", default="")
    adapt_parser.add_argument("--max-proposals", type=int, default=8)

    label_parser = subparsers.add_parser("label")
    label_parser.add_argument("--run-root", type=Path, required=True)
    label_parser.add_argument("--run-id", default="")
    label_parser.add_argument("--case-id", default="")
    label_parser.add_argument("--variant", required=True)
    label_parser.add_argument("--result-json", default="")
    label_parser.add_argument("--label", required=True)
    label_parser.add_argument("--score", type=int, required=True)
    label_parser.add_argument("--tag", action="append", default=[])
    label_parser.add_argument("--notes", default="")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--run-root", type=Path, required=True)
    promote_parser.add_argument("--variant", required=True)
    promote_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "list-suites":
        return _cmd_list_suites()
    if args.command == "list-tracks":
        return _cmd_list_tracks()
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "rank":
        return _cmd_rank(args)
    if args.command == "autopsy":
        return _cmd_autopsy(args)
    if args.command == "adapt":
        return _cmd_adapt(args)
    if args.command == "label":
        return _cmd_label(args)
    if args.command == "promote":
        return _cmd_promote(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="default-vase")
    parser.add_argument("--track", default="profile-loft-refinement")
    parser.add_argument("--search", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument(
        "--result-root", type=Path, default=Path("temp/refinement-runs/adhoc")
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)


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
        "--report-failures", choices=("top", "all", "none"), default="top"
    )
    parser.add_argument("--blender-exe", default=None)


def _cmd_list_suites() -> int:
    payload = {name: get_suite_preset(name).to_dict() for name in list_suites()}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_list_tracks() -> int:
    payload = {name: get_track_preset(name).to_dict() for name in list_tracks()}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    track = get_track_preset(args.track)
    plan = build_experiment_plan(
        suite=args.suite,
        track=args.track,
        search=args.search or track.default_search,
        objective=args.objective or track.default_objective,
        output_root=args.result_root,
        seed=args.seed,
        max_runs=args.max_runs,
        top_k=args.top_k,
    )
    write_plan(plan, args.out)
    print(
        json.dumps(
            {
                "plan": args.out.as_posix(),
                "cases": len(plan.cases),
                "variants": len(plan.variants),
            },
            indent=2,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    track = get_track_preset(args.track)
    plan = build_experiment_plan(
        suite=args.suite,
        track=args.track,
        search=args.search or track.default_search,
        objective=args.objective or track.default_objective,
        output_root=args.result_root,
        seed=args.seed,
        max_runs=args.max_runs,
        top_k=args.top_k,
    )
    if args.command_only:
        for variant in plan.variants:
            print(" ".join(variant.cli_args))
        return 0
    if args.blender_exe is None:
        try:
            import bpy  # noqa: F401
        except Exception:
            print(
                "ERROR: --blender-exe is required when running outside Blender.",
                file=sys.stderr,
            )
            return 2
    options = RunOptions(
        html_report=args.html_report,
        write_overlays=args.write_overlays,
        write_bounds_debug=args.bounds_debug,
        write_autopsy=args.autopsy,
        fail_on_all_failed=args.fail_on_all_failed,
        append_global_index=args.append_global_index,
        report_failures=args.report_failures,
        subprocess_blender=args.blender_exe is not None,
        blender_executable=args.blender_exe,
    )
    ok, _results = runner_for_plan(
        plan, options=options, base_config=BlockingConfig()
    ).run()
    return 0 if ok else 1


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
            / "cases"
            / result.case_id
            / "variants"
            / result.variant_id
            / "autopsy.json"
        )
        write_autopsy(
            result, path, run_root=args.run_root, bounds_debug=result.bounds_debug
        )
        print(path.as_posix())
    return 0 if selected else 1


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
    return _dedupe_proposals(proposals, max_proposals=max_proposals)


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


def _dedupe_proposals(
    proposals: list[RefinementProposal],
    *,
    max_proposals: int,
) -> list[RefinementProposal]:
    by_id: dict[str, RefinementProposal] = {}
    for proposal in proposals:
        current = by_id.get(proposal.proposal_id)
        if current is None or proposal.priority < current.priority:
            by_id[proposal.proposal_id] = proposal
    return sorted(by_id.values(), key=lambda proposal: proposal.priority)[
        :max_proposals
    ]


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


def _cmd_promote(args: argparse.Namespace) -> int:
    results, _malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    for result in results:
        if result.variant_id == args.variant:
            payload = {
                "schema_version": "refinement_preset_v1",
                "source_run_id": result.run_id,
                "source_variant_id": result.variant_id,
                "case_id": result.case_id,
                "mode": result.mode,
                "metrics": result.metrics,
                "command": list(result.command),
                "score": result.score,
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


if __name__ == "__main__":
    raise SystemExit(main())
