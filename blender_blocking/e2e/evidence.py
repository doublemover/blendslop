"""Refresh serialized bundles only with independently measured E2E evidence."""
from types import SimpleNamespace
from dataclasses import replace


def attach_evaluation_evidence(payload, *, geometry_payload=None):
    from blender_blocking.evaluation.bundle_builder.silhouette import _silhouette_group
    from blender_blocking.evaluation.bundle_builder.geometry import _geometry_group
    from blender_blocking.evaluation.schemas import EvaluationBundle, MetricGroup, worst_status
    from blender_blocking.evaluation.failure_taxonomy import classify_bundle_failures
    from blender_blocking.evaluation.autopsy import autopsy_pack_from_bundle
    result = dict(payload)
    views = result.get("views", {})
    summary = result.get("silhouette_summary", {})
    extras = dict(geometry_payload or {})
    boundary_values = [float(view['boundary_iou']) for view in views.values()
                       if view.get('boundary_iou') is not None]
    boundary_mean = summary.get('boundary_iou_mean')
    if boundary_mean is None:
        boundary_mean = sum(boundary_values)/len(boundary_values) if boundary_values else 0.
    metrics = SimpleNamespace(per_view=views, area_iou_min=result.get("min_view_iou", 0),
        area_iou_mean=result.get("average_iou", 0),
        boundary_iou_mean=boundary_mean,
        extras=extras)
    replacements = {}
    if result.get("validation_mode") == "render-iou" and views:
        group = _silhouette_group(metrics).to_dict()
        group["metadata"] = dict(group.get("metadata", {}), measurement_source="external_blender_render")
        replacements["silhouette"] = group
    if geometry_payload:
        replacements["geometry"] = _geometry_group(metrics).to_dict()
    bundle = result.get("evaluation_bundle")
    if not isinstance(bundle, dict):
        # Only the selected ensemble candidate was externally rendered.
        selected = result.get("backend_result", {}).get("selected", {})
        selected_id = selected.get("candidate_id") if isinstance(selected, dict) else None
        bundle = next((b for b in result.get("evaluation_bundles", [])
                       if b.get("candidate_id") == selected_id), None)
    if isinstance(bundle, dict):
        parsed = EvaluationBundle.from_dict(bundle)
        groups = tuple(MetricGroup.from_dict(replacements[g.name]) if g.name in replacements else g
                       for g in parsed.metric_groups)
        # Absent optional groups are neutral. Backend errors/degradation stay visible.
        statuses = [g.status for g in groups if g.status != "not_applicable"]
        if parsed.errors:
            statuses.append("fail")
        if parsed.degradation_state.get("degraded"):
            statuses.append("degraded")
        updated = replace(parsed, metric_groups=groups, status=worst_status(statuses), failures=())
        updated = replace(updated, failures=classify_bundle_failures(updated))
        result["evaluation_bundle"] = updated.to_dict()
        result["autopsy_pack"] = autopsy_pack_from_bundle(updated).to_dict()
        if isinstance(result.get("autopsy_packs"), list):
            result["autopsy_packs"] = [result["autopsy_pack"] if p.get("candidate_id") == updated.candidate_id else p
                                      for p in result["autopsy_packs"]]
        if isinstance(result.get("evaluation_bundles"), list):
            result["evaluation_bundles"] = [updated.to_dict() if b.get("candidate_id") == updated.candidate_id else b
                                             for b in result["evaluation_bundles"]]
    return result
