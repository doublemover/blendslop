"""Concrete proposal mutation builders used by adaptive refinement."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import safe_slug, stable_hash
from .contracts import RefinementProposal


def _boundary_first(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "boundary-first",
        "Boundary-first silhouette refinement",
        "Boundary IoU is lagging area IoU; tune mask morphology and signed-distance loss before changing geometry.",
        mode="ensemble",
        cli_args=(
            "--ensemble-policy",
            "fidelity",
            "--ref-morph-close",
            "5",
            "--ref-morph-open",
            "1",
            "--primitive-loss-weights-json",
            '{"silhouette":1.0,"sdf":0.35,"boundary":0.45}',
            "--diff-loss-weights-json",
            '{"silhouette":1.0,"sdf":0.35}',
        ),
        expected_win={
            "silhouette.mean_boundary_iou": "increase",
            "silhouette.mean_signed_distance_loss": "decrease",
        },
        tags=("boundary", "sdf", "silhouette"),
        priority=10,
        source={"metrics": metrics, "failures": failures},
    )


def _content_adaptive_patches(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "content-adaptive-patches",
        "Content-adaptive patch detail pass",
        "Boundary/detail metrics suggest local high-frequency error; run a global candidate plus focused residual patches, then fuse patch corrections back into the editable candidate.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,shape_program,differentiable_refine",
            "--ensemble-policy",
            "research_fidelity",
            "--shape-residual-policy",
            "suggest_patches",
            "--primitive-loss-weights-json",
            '{"silhouette":1.0,"boundary":0.35,"sdf":0.3,"surface":0.5}',
            "--diff-loss-weights-json",
            '{"silhouette":1.0,"boundary":0.35,"sdf":0.35}',
            "--vh-mesh-method",
            "lewiner",
        ),
        expected_win={
            "silhouette.mean_boundary_iou": "increase",
            "silhouette.mean_signed_distance_loss": "decrease",
            "geometry.fscore_tau": "increase",
            "geometry.surface_coverage": "increase",
        },
        tags=("content-adaptive-patches", "residual", "detail", "fusion"),
        priority=12,
        risk="high",
        source={"metrics": metrics, "failures": failures},
    )


def _visual_hull_resolution(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "sparse-hull-resolution",
        "Sparse visual hull resolution climb",
        "Low required-view IoU suggests recoverable volume detail; increase sparse/chunked resolution with explicit mesh metadata.",
        mode="visual_hull_voxel",
        cli_args=(
            "--validation-mode",
            "render-iou",
            "--vh-backend",
            "sparse_hash",
            "--vh-resolution",
            "128",
            "--vh-chunk-size",
            "32",
            "--vh-mesh-method",
            "lewiner",
            "--vh-occupancy-threshold",
            "0.5",
        ),
        expected_win={
            "silhouette.min_view_iou": "increase",
            "geometry.volumetric_iou": "increase",
        },
        tags=("visual-hull", "sparse", "resolution"),
        priority=20,
        source={"metrics": metrics, "failures": failures},
    )


def _uncertainty_sweep(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "uncertainty-profile-sweep",
        "Uncertainty-aware profile and mask sweep",
        "Ambiguous or noisy silhouettes should feed profile confidence and occupancy aggregation instead of a single hard threshold.",
        mode="ensemble",
        cli_args=(
            "--ensemble-policy",
            "balanced",
            "--vh-uncertainty-aggregation",
            "product",
            "--ref-gray-threshold",
            "128",
            "--ref-morph-close",
            "3",
            "--ref-fill-holes",
        ),
        expected_win={
            "silhouette.min_view_iou": "increase",
            "uncertainty.consistency": "increase",
        },
        tags=("uncertainty", "profile-band", "mask"),
        priority=30,
        source={"metrics": metrics, "failures": failures},
    )


def _topology_preserving_mesh(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "topology-preserving-mesh",
        "Topology-preserving mesh extraction",
        "The geometry is plausible but mesh QA is weak; try Lewiner marching cubes with no destructive postprocess before Poisson experiments.",
        mode="visual_hull_voxel",
        cli_args=(
            "--validation-mode",
            "render-iou",
            "--vh-backend",
            "chunked",
            "--vh-resolution",
            "96",
            "--vh-mesh-method",
            "lewiner",
            "--vh-postprocess",
            "none",
            "--ensemble-policy",
            "printable",
        ),
        expected_win={
            "topology.score": "increase",
            "topology.non_manifold_edges": "decrease",
        },
        tags=("topology", "mesh", "printable"),
        priority=40,
        source={"metrics": metrics, "failures": failures},
    )


def _shape_program_editability(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "shape-program-editability",
        "Editable shape-program candidate",
        "Mesh/proxy output is not sufficiently editable; emit a structured primitive program and residual patch hints for human-editable Blender output.",
        mode="shape_program",
        cli_args=(
            "--validation-mode",
            "backend-status",
            "--shape-root-strategy",
            "hybrid_profile_bounds",
            "--shape-residual-policy",
            "suggest_patches",
            "--shape-max-nodes",
            "96",
            "--shape-editability-bias",
            "1.0",
        ),
        expected_win={
            "editability.editable_reconstruction_index": "increase",
            "artifact.shape_program": "present",
        },
        tags=("shape-program", "editable", "research"),
        priority=50,
        risk="high",
        validation_mode="backend-status",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures},
    )


def _appearance_asset_audit(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "appearance-asset-audit",
        "Texture, UV, and material audit",
        "Image quality or editability is being limited by asset-delivery evidence; require explicit UV/PBR/material metrics and prevent texture-only detail from hiding missing geometry.",
        mode="shape_program",
        cli_args=(
            "--validation-mode",
            "backend-status",
            "--shape-root-strategy",
            "hybrid_profile_bounds",
            "--shape-residual-policy",
            "suggest_patches",
            "--evaluate-texture-materials",
            "--uv-strict",
            "--material-target",
            "pbr",
            "--max-texture-memory-mb",
            "128",
            "--shape-run-export-qa",
        ),
        expected_win={
            "appearance.uv_valid": "pass",
            "appearance.pbr_channel_coverage_ratio": "increase",
            "appearance.attribution_texture_only_detail_score": "decrease",
            "editability.editable_reconstruction_index": "increase",
        },
        tags=("appearance", "uv", "material", "editable", "asset-delivery"),
        priority=35,
        risk="medium",
        validation_mode="backend-status",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures},
    )


def _proxy_grounding(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "proxy-grounding",
        "Ground proxy candidates against volume and topology",
        "Gaussian or differentiable proxies need a mesh/volume cross-check before they can win selection.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,gaussian_ellipsoid_proxy,differentiable_refine",
            "--ensemble-policy",
            "research_fidelity",
            "--gaussian-export-mesh-proxy",
            "--vh-mesh-method",
            "lewiner",
        ),
        expected_win={
            "geometry.proxy_grounding": "increase",
            "editability.complexity_penalty": "decrease",
        },
        tags=("proxy", "gaussian", "differentiable", "crosscheck"),
        priority=60,
        source={"metrics": metrics, "failures": failures},
    )


def _primitive_fit_proxy_retry(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "primitive-fit-proxy-retry",
        "Primitive-fit proxy mismatch retry",
        "Primitive-fit produced artifacts but failed its internal/proxy fit floor; retry with a larger objective budget, silhouette/profile weighting, and broader primitive families before promotion.",
        mode="primitive_fit_refine",
        cli_args=(
            "--validation-mode",
            "backend-status",
            "--primitive-families",
            "superquadric,superfrustum,ellipsoid,capsule",
            "--primitive-steps",
            "50",
            "--primitive-max",
            "12",
            "--primitive-max-objective-evaluations",
            "1536",
            "--primitive-loss-weights-json",
            '{"silhouette":1.25,"profile":1.0,"surface":0.65,"topology":0.15}',
        ),
        expected_win={
            "backend.area_iou_min": "increase",
            "backend.objective_proxy_iou": "increase",
            "diagnostics.bounds_axis_probe": "clear",
        },
        tags=("primitive-fit", "proxy-mismatch", "objective-budget", "bounds-probe"),
        priority=5,
        risk="medium",
        validation_mode="backend-status",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures},
    )


def _shape_program_render_qa_retry(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "shape-program-render-qa",
        "Shape-program render-QA compile check",
        "Compiled shape-program artifacts need required-view render QA before grammar expansion or promotion.",
        mode="shape_program",
        cli_args=(
            "--validation-mode",
            "render-iou",
            "--shape-compile-blender",
            "--shape-root-strategy",
            "hybrid_profile_bounds",
            "--shape-residual-policy",
            "suggest_patches",
            "--shape-run-export-qa",
        ),
        expected_win={
            "render.per_view.front.area_iou": "present",
            "render.per_view.side.area_iou": "present",
            "render.per_view.top.area_iou": "present",
            "render.qa.missing_required_metrics": "false",
        },
        tags=("shape-program", "render-qa", "compile-validation"),
        priority=4,
        risk="low",
        validation_mode="render-iou",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures},
    )


def _compile_or_crosscheck(
    metrics: Mapping[str, float],
    failures: Sequence[str],
    status: str,
) -> RefinementProposal:
    return _proposal(
        "research-crosscheck",
        "Cross-check research candidate",
        "Research-only or degraded output should be compared against a conservative backend before being promoted.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,shape_program",
            "--ensemble-policy",
            "editable",
            "--validation-mode",
            "backend-status",
        ),
        expected_win={"candidate.status": f"escape {status}"},
        tags=("research", "crosscheck", "ensemble"),
        priority=70,
        validation_mode="backend-status",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures, "status": status},
    )


def _active_view_capture(
    metrics: Mapping[str, float],
    failures: Sequence[str],
) -> RefinementProposal:
    return _proposal(
        "active-view-capture",
        "Capture another high-information silhouette view",
        "The current silhouette set is likely underconstrained; add a diagonal or oblique view before overfitting backend parameters.",
        mode="ensemble",
        cli_args=(
            "--ensemble-candidates",
            "visual_hull_voxel,primitive_fit_refine,shape_program",
            "--ensemble-policy",
            "fidelity",
            "--validation-mode",
            "backend-status",
        ),
        expected_win={
            "geometry.ambiguity_gap": "decrease",
            "silhouette.min_view_iou": "increase",
        },
        tags=("active-view", "ambiguity", "human-input"),
        priority=15,
        risk="low",
        validation_mode="backend-status",
        diagnostic_only=True,
        source={"metrics": metrics, "failures": failures},
    )


def _proposal(
    slug: str,
    title: str,
    hypothesis: str,
    *,
    mode: str,
    cli_args: Sequence[str],
    expected_win: Mapping[str, str | float],
    tags: Sequence[str],
    priority: int,
    source: Mapping[str, Any],
    risk: str = "medium",
    validation_mode: str = "render-iou",
    diagnostic_only: bool = False,
) -> RefinementProposal:
    normalized_args = _without_validation_mode(cli_args)
    proposal_id = f"{safe_slug(slug)}_{stable_hash({'slug': slug, 'args': list(normalized_args), 'validation_mode': validation_mode}, length=8)}"
    return RefinementProposal(
        proposal_id=proposal_id,
        title=title,
        hypothesis=hypothesis,
        expected_win=expected_win,
        mode=mode,
        validation_mode=validation_mode,
        cli_args=tuple(str(item) for item in normalized_args),
        tags=tuple(str(item) for item in tags),
        priority=priority,
        risk=risk,
        source_evidence=source,
        diagnostic_only=diagnostic_only,
    )


def _without_validation_mode(cli_args: Sequence[str]) -> tuple[str, ...]:
    stripped: list[str] = []
    skip_next = False
    for item in cli_args:
        if skip_next:
            skip_next = False
            continue
        text = str(item)
        if text == "--validation-mode":
            skip_next = True
            continue
        stripped.append(text)
    return tuple(stripped)
