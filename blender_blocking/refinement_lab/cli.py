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

from .adaptive_planner import (
    RefinementProposal,
    merge_proposals,
    proposals_from_result_payload,
)
from .adaptive_loop import AdaptiveLoopOptions, run_adaptive_loop
from .artifact_report import ReportOptions, generate_report
from .candidate_autopsy import write_autopsy
from .contracts import json_safe
from .editability_study import (
    build_editability_study_pack,
    load_review_rows,
    summarize_review_rows,
    write_editability_study_pack,
)
from .human_labels import HumanLabel, append_label
from .matrix import build_experiment_plan, load_variants_from_files, write_plan
from .parameter_search import promotion_decision
from .presets import get_suite_preset, get_track_preset, list_suites, list_tracks
from .result_index import load_index, write_leaderboard_json, write_leaderboard_md
from .runner import RunOptions, runner_for_plan
from .surrogate import surrogate_report

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
    external_variants = load_variants_from_files(args.variant_file)
    plan = build_experiment_plan(
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
        output_root=args.result_root,
        seed=args.seed,
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


def _cmd_surrogate(args: argparse.Namespace) -> int:
    results, malformed = load_index(
        args.run_root / "index.jsonl", run_root=args.run_root
    )
    if malformed:
        print(f"WARN: skipped {malformed} malformed index rows")
    plan_path = args.plan or args.run_root / "plan.json"
    variants = ()
    if plan_path.exists():
        from .contracts import ExperimentPlan

        variants = ExperimentPlan.read(plan_path).variants
    report = surrogate_report(
        results,
        variants,
        objective=args.objective,
        ridge=args.ridge,
        top_k=args.top_k,
    )
    out = args.out or args.run_root / "surrogate-priorities.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": out.as_posix(),
                "examples": report["model"]["example_count"],  # type: ignore[index]
                "predictions": report["prediction_count"],
                "top_variant": (
                    report["predictions"][0]["variant_id"]  # type: ignore[index]
                    if report["predictions"]
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
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


def _cmd_calibrate_masks(args: argparse.Namespace) -> int:
    from evaluation.calibration import executable_calibration_sweep

    references = _load_view_masks(args.reference)
    candidates = _load_view_masks(args.candidate)
    report = executable_calibration_sweep(
        references,
        candidates,
        max_offset_px=args.max_offset_px,
        step_px=args.step_px,
        min_area_iou_delta=args.min_area_iou_delta,
        min_boundary_iou_delta=args.min_boundary_iou_delta,
        max_signed_distance_loss_increase=args.max_sdf_loss_increase,
        min_area_iou=args.min_area_iou,
        min_boundary_iou=args.min_boundary_iou,
        max_signed_distance_loss=args.max_sdf_loss,
    )
    payload = report.to_dict()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.out.as_posix())
    else:
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    return 0 if report.status in {"improved", "no_improvement"} else 1


def _cmd_patch_masks(args: argparse.Namespace) -> int:
    from .content_adaptive_patches import run_content_adaptive_patch_refinement

    reference = _load_mask_image(args.reference)
    candidate = _load_mask_image(args.candidate)
    image = _load_rgb_image(args.image) if args.image is not None else None
    result = run_content_adaptive_patch_refinement(
        reference,
        candidate_mask=candidate,
        image=image,
        patch_sizes=_parse_int_csv(args.patch_sizes),
        max_patches=args.max_patches,
        threshold=args.threshold,
        correction_strength=args.correction_strength,
        min_area_iou_delta=args.min_area_iou_delta,
        min_boundary_iou_delta=args.min_boundary_iou_delta,
        alignment_mode=args.alignment_mode,
        feather_fraction=args.feather_fraction,
    )
    refined_mask_path = args.refined_mask_out
    if refined_mask_path is not None:
        _save_mask_image(refined_mask_path, result.refined_mask)
    payload = {
        "schema_version": "content_adaptive_patch_mask_result_v1",
        "reference": args.reference.as_posix(),
        "candidate": args.candidate.as_posix(),
        "image": args.image.as_posix() if args.image is not None else None,
        "refined_mask": refined_mask_path.as_posix() if refined_mask_path else None,
        **result.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "patches": len(result.patches),
                "area_iou_delta": result.improvement.get("area_iou_delta"),
                "boundary_iou_delta": result.improvement.get("boundary_iou_delta"),
                "out": args.out.as_posix(),
                "refined_mask": refined_mask_path.as_posix()
                if refined_mask_path
                else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.accepted else 1


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


def _load_view_masks(entries: list[str]) -> dict[str, object]:
    masks = {}
    for entry in entries:
        view, path = _parse_view_path(entry)
        masks[view] = _load_mask_image(path)
    return masks


def _parse_view_path(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise SystemExit(f"expected VIEW=PATH, got {entry!r}")
    view, raw_path = entry.split("=", 1)
    view = view.strip()
    if not view:
        raise SystemExit(f"empty view name in {entry!r}")
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"mask path does not exist: {path}")
    return view, path


def _load_mask_image(path: Path) -> object:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency check covers this
        raise SystemExit(f"Pillow/numpy are required to load masks: {exc}") from exc
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image)
    alpha = array[..., 3]
    rgb = array[..., :3].mean(axis=2)
    return (alpha > 0) & (rgb > 8)


def _load_rgb_image(path: Path) -> object:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency check covers this
        raise SystemExit(f"Pillow/numpy are required to load images: {exc}") from exc
    return np.asarray(Image.open(path).convert("RGB"))


def _save_mask_image(path: Path, mask: object) -> None:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency check covers this
        raise SystemExit(f"Pillow/numpy are required to save masks: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(mask)
    if array.ndim != 2:
        raise SystemExit("refined mask must be a 2D array")
    Image.fromarray(array.astype(bool).astype("uint8") * 255, mode="L").save(path)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = []
    for raw in str(value or "").split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            item = int(text)
        except ValueError as exc:
            raise SystemExit(f"invalid integer in --patch-sizes: {text!r}") from exc
        if item <= 0:
            raise SystemExit("--patch-sizes values must be positive")
        parsed.append(item)
    if not parsed:
        raise SystemExit("--patch-sizes must contain at least one positive integer")
    return tuple(parsed)


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


if __name__ == "__main__":
    raise SystemExit(main())
