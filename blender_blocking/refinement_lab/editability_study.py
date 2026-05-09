"""Human editability study packs for Blender reconstruction candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .contracts import ExperimentResult, json_safe, stable_hash, utc_now
from .parameter_search import rank_results, score_result


@dataclass(frozen=True)
class EditabilityCriterion:
    criterion_id: str
    label: str
    prompt: str
    weight: float = 1.0
    anchors: Mapping[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion_id": self.criterion_id,
            "label": self.label,
            "prompt": self.prompt,
            "weight": float(self.weight),
            "anchors": {str(key): value for key, value in self.anchors.items()},
        }


@dataclass(frozen=True)
class EditabilityStudyItem:
    item_id: str
    rank: int
    run_id: str
    case_id: str
    variant_id: str
    mode: str
    status: str
    objective_score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    review_template: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "rank": int(self.rank),
            "run_id": self.run_id,
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "mode": self.mode,
            "status": self.status,
            "objective_score": float(self.objective_score),
            "metrics": json_safe(self.metrics),
            "evidence": json_safe(self.evidence),
            "artifacts": dict(self.artifacts),
            "review_template": json_safe(self.review_template),
        }


@dataclass(frozen=True)
class EditabilityStudyPack:
    study_id: str
    created_utc: str
    run_root: str
    objective: str
    criteria: tuple[EditabilityCriterion, ...]
    items: tuple[EditabilityStudyItem, ...]
    instructions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "editability_study_pack_v1",
            "study_id": self.study_id,
            "created_utc": self.created_utc,
            "run_root": self.run_root,
            "objective": self.objective,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "items": [item.to_dict() for item in self.items],
            "instructions": list(self.instructions),
        }


DEFAULT_EDITABILITY_CRITERIA: tuple[EditabilityCriterion, ...] = (
    EditabilityCriterion(
        "semantic_parts",
        "Semantic Parts",
        "Can a Blender artist identify and select meaningful object parts?",
        weight=1.2,
        anchors={
            1: "single fused blob or noisy shell",
            3: "major regions visible but not cleanly separable",
            5: "clear named/inspectable parts suitable for direct editing",
        },
    ),
    EditabilityCriterion(
        "topology_cleanliness",
        "Topology Cleanliness",
        "Does the mesh avoid non-manifold edges, holes, degenerate faces, and loose vertices?",
        weight=1.3,
        anchors={
            1: "broken topology blocks normal Blender editing",
            3: "usable with cleanup",
            5: "watertight or intentionally open with clean local topology",
        },
    ),
    EditabilityCriterion(
        "mesh_density",
        "Mesh Density",
        "Is the mesh density appropriate for sculpting/editing without hiding the shape?",
        weight=0.9,
        anchors={
            1: "too dense/sparse to edit productively",
            3: "acceptable after decimation or remesh",
            5: "density supports direct sculpt/blockout edits",
        },
    ),
    EditabilityCriterion(
        "primitive_controls",
        "Primitive Controls",
        "Does the output preserve parametric or primitive handles where possible?",
        weight=1.1,
        anchors={
            1: "only opaque triangles/voxels",
            3: "some proxy primitives or simple mesh regions",
            5: "clear primitive/shape-program controls for major forms",
        },
    ),
    EditabilityCriterion(
        "modifier_stack",
        "Modifier Stack",
        "Are Blender modifiers, bevels, normals, booleans, and cleanup steps readable and non-destructive where useful?",
        weight=0.8,
        anchors={
            1: "destructive or missing stack with unclear cleanup",
            3: "basic modifiers present but limited semantics",
            5: "organized, named, editable modifier workflow",
        },
    ),
    EditabilityCriterion(
        "scale_orientation",
        "Scale And Orientation",
        "Is the asset centered, consistently scaled, and oriented for immediate Blender use?",
        weight=0.8,
        anchors={
            1: "wrong scale/orientation/framing",
            3: "minor transform cleanup needed",
            5: "ready to append, inspect, render, and edit",
        },
    ),
    EditabilityCriterion(
        "silhouette_fidelity",
        "Silhouette Fidelity",
        "Does the editable output preserve the required input silhouettes and boundaries?",
        weight=1.2,
        anchors={
            1: "obvious missed silhouette intent",
            3: "roughly matches but loses boundary details",
            5: "front/side/top and boundary intent are preserved",
        },
    ),
    EditabilityCriterion(
        "export_reliability",
        "Export Reliability",
        "Does the asset survive OBJ/GLB/Blend export and reimport expectations?",
        weight=0.9,
        anchors={
            1: "missing or broken export path",
            3: "exports with warnings",
            5: "round-trip/export QA is clean",
        },
    ),
)


DEFAULT_STUDY_INSTRUCTIONS = (
    "Open each mesh or Blender artifact if present, then inspect the reference/render images.",
    "Score each criterion from 1 to 5 using the anchors; leave null only when evidence is missing.",
    "Prefer editability over visual-only fidelity: a pretty opaque mesh can score lower than a clean primitive program.",
    "Record required cleanup steps as tags or notes so future automated scoring can learn from them.",
)


def build_editability_study_pack(
    results: Sequence[ExperimentResult],
    *,
    run_root: Path,
    objective: str = "quality_win",
    top_k: Optional[int] = 10,
    include_failed: bool = False,
    criteria: Sequence[EditabilityCriterion] = DEFAULT_EDITABILITY_CRITERIA,
) -> EditabilityStudyPack:
    filtered = [
        result
        for result in results
        if include_failed or result.status == "pass"
    ]
    ranked = rank_results(filtered, objective=objective)
    if top_k is not None:
        ranked = ranked[: max(0, int(top_k))]
    items = tuple(
        _item_from_result(
            result,
            rank=index + 1,
            objective_score=float(score.get("total", 0.0)),
            criteria=criteria,
            run_root=Path(run_root),
        )
        for index, (result, score) in enumerate(ranked)
    )
    study_id = stable_hash(
        {
            "run_root": Path(run_root).as_posix(),
            "objective": objective,
            "items": [item.item_id for item in items],
            "criteria": [criterion.criterion_id for criterion in criteria],
        },
        length=16,
    )
    return EditabilityStudyPack(
        study_id=f"editability-{study_id}",
        created_utc=utc_now(),
        run_root=Path(run_root).as_posix(),
        objective=objective,
        criteria=tuple(criteria),
        items=items,
        instructions=DEFAULT_STUDY_INSTRUCTIONS,
    )


def write_editability_study_pack(
    pack: EditabilityStudyPack,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "editability-study.json"
    md_path = output_dir / "editability-study.md"
    template_path = output_dir / "review-template.jsonl"
    json_path.write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_study_markdown(pack), encoding="utf-8")
    with template_path.open("w", encoding="utf-8") as handle:
        for item in pack.items:
            handle.write(
                json.dumps(item.review_template, sort_keys=True, default=str) + "\n"
            )
    return {
        "json": json_path,
        "markdown": md_path,
        "review_template": template_path,
    }


def score_review_row(
    row: Mapping[str, Any],
    *,
    criteria: Sequence[EditabilityCriterion] = DEFAULT_EDITABILITY_CRITERIA,
) -> dict[str, Any]:
    scores = row.get("scores", {})
    if not isinstance(scores, Mapping):
        scores = {}
    weight_by_id = {criterion.criterion_id: float(criterion.weight) for criterion in criteria}
    total_weight = 0.0
    weighted_score = 0.0
    missing: list[str] = []
    for criterion in criteria:
        raw = scores.get(criterion.criterion_id)
        if raw is None:
            missing.append(criterion.criterion_id)
            continue
        value = max(1.0, min(5.0, float(raw)))
        weight = weight_by_id[criterion.criterion_id]
        weighted_score += (value / 5.0) * weight
        total_weight += weight
    normalized = weighted_score / total_weight if total_weight > 0.0 else 0.0
    return {
        "item_id": row.get("item_id", ""),
        "variant_id": row.get("variant_id", ""),
        "score": normalized,
        "weighted_score": weighted_score,
        "total_weight": total_weight,
        "missing_criteria": missing,
        "reviewer": row.get("reviewer", ""),
        "label": _score_label(normalized),
    }


def load_review_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_review_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    criteria: Sequence[EditabilityCriterion] = DEFAULT_EDITABILITY_CRITERIA,
) -> dict[str, Any]:
    scored = [score_review_row(row, criteria=criteria) for row in rows]
    if not scored:
        return {"count": 0, "mean_score": 0.0, "items": []}
    return {
        "count": len(scored),
        "mean_score": sum(float(row["score"]) for row in scored) / len(scored),
        "items": scored,
        "top_item": max(scored, key=lambda row: float(row["score"])),
    }


def _item_from_result(
    result: ExperimentResult,
    *,
    rank: int,
    objective_score: float,
    criteria: Sequence[EditabilityCriterion],
    run_root: Path,
) -> EditabilityStudyItem:
    metrics = _study_metrics(result)
    artifacts = _study_artifacts(result, run_root=run_root)
    evidence = {
        "command": list(result.command),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "backend": _backend_evidence(result.backend_result),
        "autopsy": json_safe(result.autopsy),
        "bounds_debug": json_safe(result.bounds_debug),
    }
    item_id = stable_hash(
        {
            "run_id": result.run_id,
            "case_id": result.case_id,
            "variant_id": result.variant_id,
            "mode": result.mode,
            "result_json": result.result_json.as_posix() if result.result_json else "",
        },
        length=12,
    )
    template = {
        "schema_version": "editability_review_v1",
        "item_id": item_id,
        "run_id": result.run_id,
        "case_id": result.case_id,
        "variant_id": result.variant_id,
        "mode": result.mode,
        "result_json": (
            _rel_path(result.result_json, run_root) if result.result_json else ""
        ),
        "reviewer": "",
        "scores": {criterion.criterion_id: None for criterion in criteria},
        "label": "",
        "tags": [],
        "notes": "",
    }
    return EditabilityStudyItem(
        item_id=item_id,
        rank=rank,
        run_id=result.run_id,
        case_id=result.case_id,
        variant_id=result.variant_id,
        mode=result.mode,
        status=result.status,
        objective_score=objective_score,
        metrics=metrics,
        evidence=evidence,
        artifacts=artifacts,
        review_template=template,
    )


def _study_metrics(result: ExperimentResult) -> dict[str, Any]:
    keys = (
        "average_iou",
        "min_view_iou",
        "front_iou",
        "side_iou",
        "top_iou",
        "mean_boundary_iou",
        "min_boundary_iou",
        "mean_signed_distance_loss",
        "topology_score",
        "topology_penalty",
        "editability_score",
        "complexity_penalty",
        "novel_view_psnr",
        "novel_view_ssim",
        "novel_view_lpips",
    )
    metrics = {key: result.metrics.get(key) for key in keys if key in result.metrics}
    metrics.setdefault("average_iou", result.avg_iou)
    metrics.setdefault("min_view_iou", result.min_iou)
    metrics["objective_score_raw"] = score_result(result).get("total", 0.0)
    return json_safe(metrics)


def _study_artifacts(result: ExperimentResult, *, run_root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if result.result_json:
        artifacts["result_json"] = _rel_path(result.result_json, run_root)
    for prefix, mapping in (
        ("render", result.render_paths),
        ("reference", result.reference_paths),
        ("artifact", result.artifacts),
    ):
        for key, path in mapping.items():
            artifacts[f"{prefix}_{key}"] = _rel_path(path, run_root)
    selected = _selected_backend(result.backend_result)
    for key in ("mesh_path", "primitive_path", "volume_path"):
        value = selected.get(key)
        if value:
            artifacts[key] = _rel_path(Path(str(value)), run_root)
    nested = selected.get("artifacts")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if value:
                artifacts[f"backend_{key}"] = _rel_path(Path(str(value)), run_root)
    return artifacts


def _backend_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = _selected_backend(payload)
    metric_result = selected.get("metric_result", {})
    extras = metric_result.get("extras", {}) if isinstance(metric_result, Mapping) else {}
    return {
        "candidate_id": selected.get("candidate_id", ""),
        "backend_name": selected.get("backend_name", ""),
        "status": selected.get("status", ""),
        "warnings": selected.get("warnings", []),
        "errors": selected.get("errors", []),
        "topology": extras.get("topology") if isinstance(extras, Mapping) else {},
        "editability": extras.get("editability") if isinstance(extras, Mapping) else {},
        "export_qa": (
            extras.get("export_qa")
            or extras.get("asset_export")
            or extras.get("export")
            if isinstance(extras, Mapping)
            else {}
        ),
        "mesh": extras.get("mesh") if isinstance(extras, Mapping) else {},
    }


def _selected_backend(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    selected = payload.get("selected")
    if isinstance(selected, Mapping):
        return selected
    return payload


def _rel_path(path: Path, run_root: Path) -> str:
    path = Path(path)
    try:
        return path.resolve(strict=False).relative_to(
            run_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return path.as_posix()


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "sculptable"
    if score >= 0.65:
        return "useful_proxy"
    if score >= 0.45:
        return "review_required"
    return "reject"


def _study_markdown(pack: EditabilityStudyPack) -> str:
    lines = [
        "# Blender Editability Study",
        "",
        f"Study: `{pack.study_id}`",
        f"Objective: `{pack.objective}`",
        f"Run root: `{pack.run_root}`",
        "",
        "## Instructions",
        "",
    ]
    for instruction in pack.instructions:
        lines.append(f"- {instruction}")
    lines.extend(
        [
            "",
            "## Rubric",
            "",
            "| Criterion | Weight | 1 | 3 | 5 |",
            "|---|---:|---|---|---|",
        ]
    )
    for criterion in pack.criteria:
        anchors = criterion.anchors
        lines.append(
            "| {label} | {weight:.2f} | {one} | {three} | {five} |".format(
                label=criterion.label,
                weight=criterion.weight,
                one=str(anchors.get(1, "")),
                three=str(anchors.get(3, "")),
                five=str(anchors.get(5, "")),
            )
        )
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| Rank | Variant | Mode | Status | Score | Min IoU | Editability | Topology | Mesh | Result |",
            "|---:|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in pack.items:
        artifacts = item.artifacts
        metrics = item.metrics
        lines.append(
            "| {rank} | {variant} | {mode} | {status} | {score:.3f} | {min_iou:.3f} | {editability:.3f} | {topology:.3f} | {mesh} | {result} |".format(
                rank=item.rank,
                variant=item.variant_id,
                mode=item.mode,
                status=item.status,
                score=item.objective_score,
                min_iou=_float(metrics.get("min_view_iou")),
                editability=_float(metrics.get("editability_score")),
                topology=_float(metrics.get("topology_score")),
                mesh=artifacts.get("mesh_path", artifacts.get("backend_mesh_obj", "")),
                result=artifacts.get("result_json", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Review Template",
            "",
            "Fill `review-template.jsonl` by replacing null scores with 1-5 values.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
