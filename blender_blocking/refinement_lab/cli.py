"""Command line entry point for the reconstruction refinement lab."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[1]
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))

from .commands import (
    _add_plan_args,
    _add_run_args,
    _cmd_adapt,
    _cmd_autopsy,
    _cmd_calibrate_masks,
    _cmd_label,
    _cmd_list_suites,
    _cmd_list_tracks,
    _cmd_loop,
    _cmd_patch_masks,
    _cmd_plan,
    _cmd_promote,
    _cmd_rank,
    _cmd_report,
    _cmd_run,
    _cmd_score_study,
    _cmd_study_pack,
    _cmd_surrogate,
)

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

    loop_parser = subparsers.add_parser("loop")
    _add_plan_args(loop_parser)
    _add_run_args(loop_parser)
    loop_parser.add_argument("--generations", type=int, default=3)
    loop_parser.add_argument("--parent-top-k", type=int, default=3)
    loop_parser.add_argument("--children-per-parent", type=int, default=4)
    loop_parser.add_argument(
        "--keep-going-without-children",
        action="store_true",
        help="continue until generation limit even if a generation emits no child variants",
    )

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument("--objective", default="quality_win")

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--run-root", type=Path, required=True)
    rank_parser.add_argument("--objective", default="quality_win")
    rank_parser.add_argument("--top-k", type=int, default=10)

    surrogate_parser = subparsers.add_parser("surrogate")
    surrogate_parser.add_argument("--run-root", type=Path, required=True)
    surrogate_parser.add_argument("--plan", type=Path, default=None)
    surrogate_parser.add_argument("--objective", default="quality_win")
    surrogate_parser.add_argument("--top-k", type=int, default=20)
    surrogate_parser.add_argument("--ridge", type=float, default=1e-6)
    surrogate_parser.add_argument("--out", type=Path, default=None)

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

    calibrate_parser = subparsers.add_parser("calibrate-masks")
    calibrate_parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Reference mask as VIEW=PATH. Repeat once per view.",
    )
    calibrate_parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate/render mask as VIEW=PATH. Repeat once per view.",
    )
    calibrate_parser.add_argument("--out", type=Path, default=None)
    calibrate_parser.add_argument("--max-offset-px", type=int, default=12)
    calibrate_parser.add_argument("--step-px", type=int, default=4)
    calibrate_parser.add_argument("--min-area-iou-delta", type=float, default=0.01)
    calibrate_parser.add_argument("--min-boundary-iou-delta", type=float, default=0.01)
    calibrate_parser.add_argument(
        "--max-sdf-loss-increase",
        type=float,
        default=0.0,
    )
    calibrate_parser.add_argument("--min-area-iou", type=float, default=0.0)
    calibrate_parser.add_argument("--min-boundary-iou", type=float, default=None)
    calibrate_parser.add_argument("--max-sdf-loss", type=float, default=None)

    patch_parser = subparsers.add_parser("patch-masks")
    patch_parser.add_argument("--reference", type=Path, required=True)
    patch_parser.add_argument("--candidate", type=Path, required=True)
    patch_parser.add_argument("--image", type=Path, default=None)
    patch_parser.add_argument("--out", type=Path, required=True)
    patch_parser.add_argument("--refined-mask-out", type=Path, default=None)
    patch_parser.add_argument("--patch-sizes", default="32,64,96")
    patch_parser.add_argument("--max-patches", type=int, default=8)
    patch_parser.add_argument("--threshold", type=float, default=0.5)
    patch_parser.add_argument("--correction-strength", type=float, default=1.0)
    patch_parser.add_argument("--min-area-iou-delta", type=float, default=0.01)
    patch_parser.add_argument("--min-boundary-iou-delta", type=float, default=0.0)
    patch_parser.add_argument(
        "--alignment-mode",
        choices=("none", "mean", "affine"),
        default="none",
    )
    patch_parser.add_argument("--feather-fraction", type=float, default=0.12)

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

    study_parser = subparsers.add_parser("study-pack")
    study_parser.add_argument("--run-root", type=Path, required=True)
    study_parser.add_argument("--out", type=Path, default=None)
    study_parser.add_argument("--objective", default="quality_win")
    study_parser.add_argument("--top-k", type=int, default=10)
    study_parser.add_argument("--include-failed", action="store_true")

    score_study_parser = subparsers.add_parser("score-study")
    score_study_parser.add_argument("--review-jsonl", type=Path, required=True)
    score_study_parser.add_argument("--out", type=Path, default=None)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--run-root", type=Path, required=True)
    promote_parser.add_argument("--variant", required=True)
    promote_parser.add_argument("--out", type=Path, required=True)
    promote_parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help="allow degraded, research-only, metric-only, or otherwise unverified results",
    )

    args = parser.parse_args(argv)
    if args.command == "list-suites":
        return _cmd_list_suites()
    if args.command == "list-tracks":
        return _cmd_list_tracks()
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "loop":
        return _cmd_loop(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "rank":
        return _cmd_rank(args)
    if args.command == "surrogate":
        return _cmd_surrogate(args)
    if args.command == "autopsy":
        return _cmd_autopsy(args)
    if args.command == "adapt":
        return _cmd_adapt(args)
    if args.command == "calibrate-masks":
        return _cmd_calibrate_masks(args)
    if args.command == "patch-masks":
        return _cmd_patch_masks(args)
    if args.command == "label":
        return _cmd_label(args)
    if args.command == "study-pack":
        return _cmd_study_pack(args)
    if args.command == "score-study":
        return _cmd_score_study(args)
    if args.command == "promote":
        return _cmd_promote(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
