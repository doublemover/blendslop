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
    from ..content_adaptive_patches import run_content_adaptive_patch_refinement

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
