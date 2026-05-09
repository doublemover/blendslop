"""Adaptive refinement proposal engine.

The refinement lab can already enumerate fixed grids.  This module adds the
source-level contracts for a closed-loop workflow: read candidate evidence,
classify the weakness, and emit concrete parameter/mode proposals for the next
batch without hard-coding those decisions into one CLI run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import ExperimentVariant, safe_slug, stable_hash


@dataclass(frozen=True)
class RefinementProposal:
    proposal_id: str
    title: str
    hypothesis: str
    expected_win: Mapping[str, str | float] = field(default_factory=dict)
    mode: str = "ensemble"
    validation_mode: str = "backend-status"
    cli_args: tuple[str, ...] = ()
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    priority: int = 50
    risk: str = "medium"
    source_evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "expected_win": dict(self.expected_win),
            "mode": self.mode,
            "validation_mode": self.validation_mode,
            "cli_args": list(self.cli_args),
            "config_overrides": dict(self.config_overrides),
            "tags": list(self.tags),
            "priority": self.priority,
            "risk": self.risk,
            "source_evidence": dict(self.source_evidence),
        }

    def to_variant(self, *, parent_variant_id: str = "") -> ExperimentVariant:
        variant_id = safe_slug(f"adaptive_{self.proposal_id}")
        return ExperimentVariant(
            variant_id=variant_id,
            label=self.title,
            mode=self.mode,
            validation_mode=self.validation_mode,
            parameters={
                "proposal_id": self.proposal_id,
                "hypothesis": self.hypothesis,
                "priority": self.priority,
                "risk": self.risk,
            },
            cli_args=self.cli_args,
            config_overrides=self.config_overrides,
            expected_artifacts=("evaluation_bundle", "autopsy_pack"),
            tags=("adaptive",) + self.tags,
            parent_variant_id=parent_variant_id,
            stage="adaptive_refinement",
            diagnostic_only=False,
        )


def proposals_from_bundle(
    bundle: Any,
    *,
    max_proposals: int = 8,
) -> tuple[RefinementProposal, ...]:
    metrics = _metric_index(bundle)
    failures = _failure_codes(bundle)
    proposals: list[RefinementProposal] = []

    min_iou = _metric(metrics, "silhouette.min_view_iou")
    boundary = _metric(metrics, "silhouette.mean_boundary_iou")
    editable = _metric(metrics, "editability.editable_reconstruction_index")
    topology = _metric(metrics, "topology.score")
    penalty = _metric(metrics, "topology.penalty")

    if boundary < max(0.35, min_iou - 0.15) or any("boundary" in f for f in failures):
        proposals.append(_boundary_first(metrics, failures))
    if min_iou < 0.7 or any("silhouette" in f for f in failures):
        proposals.append(_visual_hull_resolution(metrics, failures))
        proposals.append(_uncertainty_sweep(metrics, failures))
    if topology < 0.85 or penalty > 0.1 or any("topology" in f for f in failures):
        proposals.append(_topology_preserving_mesh(metrics, failures))
    if editable < 0.55 or any("editable" in f or "asset" in f for f in failures):
        proposals.append(_shape_program_editability(metrics, failures))
    if any("metric_only" in f or "proxy" in f for f in failures):
        proposals.append(_proxy_grounding(metrics, failures))
    if _status(bundle) in {"research_only", "degraded"}:
        proposals.append(_compile_or_crosscheck(metrics, failures, _status(bundle)))

    deduped = _dedupe(proposals)
    return tuple(sorted(deduped, key=lambda item: item.priority)[:max_proposals])


def variants_from_bundle(
    bundle: Any,
    *,
    parent_variant_id: str = "",
    max_proposals: int = 8,
) -> tuple[ExperimentVariant, ...]:
    return tuple(
        proposal.to_variant(parent_variant_id=parent_variant_id)
        for proposal in proposals_from_bundle(bundle, max_proposals=max_proposals)
    )


def proposals_from_result_payload(
    payload: Mapping[str, Any],
    *,
    max_proposals: int = 8,
) -> tuple[RefinementProposal, ...]:
    bundles = payload.get("evaluation_bundles")
    if isinstance(bundles, Sequence) and not isinstance(bundles, (str, bytes)):
        proposals: list[RefinementProposal] = []
        for bundle in bundles:
            proposals.extend(proposals_from_bundle(bundle, max_proposals=max_proposals))
        return tuple(sorted(_dedupe(proposals), key=lambda item: item.priority)[:max_proposals])
    bundle = payload.get("evaluation_bundle")
    if isinstance(bundle, Mapping):
        return proposals_from_bundle(bundle, max_proposals=max_proposals)
    backend = payload.get("backend_result")
    if isinstance(backend, Mapping):
        return _fallback_proposals_from_backend(backend, max_proposals=max_proposals)
    return ()


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
            "backend-status",
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
            "backend-status",
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
        source={"metrics": metrics, "failures": failures, "status": status},
    )


def _fallback_proposals_from_backend(
    backend: Mapping[str, Any],
    *,
    max_proposals: int,
) -> tuple[RefinementProposal, ...]:
    status = str(backend.get("status", ""))
    metrics = backend.get("metric_result", {})
    fake_bundle = {
        "status": status,
        "metric_groups": (),
        "failures": (),
        "metrics": metrics if isinstance(metrics, Mapping) else {},
    }
    return proposals_from_bundle(fake_bundle, max_proposals=max_proposals)


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
) -> RefinementProposal:
    proposal_id = f"{safe_slug(slug)}_{stable_hash({'slug': slug, 'args': list(cli_args)}, length=8)}"
    return RefinementProposal(
        proposal_id=proposal_id,
        title=title,
        hypothesis=hypothesis,
        expected_win=expected_win,
        mode=mode,
        cli_args=tuple(str(item) for item in cli_args),
        tags=tuple(str(item) for item in tags),
        priority=priority,
        risk=risk,
        source_evidence=source,
    )


def _metric_index(bundle: Any) -> Mapping[str, float]:
    if hasattr(bundle, "metric_index"):
        return {
            key: _float(getattr(metric, "value", None))
            for key, metric in bundle.metric_index().items()
        }
    if isinstance(bundle, Mapping):
        metrics: dict[str, float] = {}
        direct = bundle.get("metrics")
        if isinstance(direct, Mapping):
            metrics.update({str(key): _float(value) for key, value in direct.items()})
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if isinstance(metric, Mapping):
                    metrics[str(metric.get("name", ""))] = _float(metric.get("value"))
        return metrics
    return {}


def _failure_codes(bundle: Any) -> tuple[str, ...]:
    failures = getattr(bundle, "failures", None)
    if failures is None and isinstance(bundle, Mapping):
        failures = bundle.get("failures", ())
    codes = []
    for failure in failures or ():
        if hasattr(failure, "code"):
            codes.append(str(failure.code))
        elif isinstance(failure, Mapping):
            codes.append(str(failure.get("code", "")))
        else:
            codes.append(str(failure))
    return tuple(code for code in codes if code)


def _status(bundle: Any) -> str:
    if hasattr(bundle, "status"):
        return str(bundle.status)
    if isinstance(bundle, Mapping):
        return str(bundle.get("status", ""))
    return ""


def _metric(metrics: Mapping[str, float], name: str, default: float = 0.0) -> float:
    return _float(metrics.get(name, default), default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _dedupe(proposals: Sequence[RefinementProposal]) -> tuple[RefinementProposal, ...]:
    by_title: dict[str, RefinementProposal] = {}
    for proposal in proposals:
        current = by_title.get(proposal.title)
        if current is None or proposal.priority < current.priority:
            by_title[proposal.title] = proposal
    return tuple(by_title.values())
