"""Measured candidate evidence and budgeted routing for calibrated inputs."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import time
import numpy as np


def freeze_geometry(result, request, *, project_diagnostics=True):
    """Capture geometry once; polygon diagnostics are optional on fresh-render paths."""
    if not result.succeeded or request.candidate_artifact_root() is None:
        return result
    from blender_blocking.reconstruction.native_geometry import GeometryArrays, GeometryCache, NativeOwnedGeometry, evaluated_arrays
    from blender_blocking.evaluation.cost_model import timed_call
    from blender_blocking.evaluation.comparable_geometry import read_obj
    from blender_blocking.reconstruction.mesh_io import write_obj
    from blender_blocking.e2e.payloads import _find_renderable_mesh
    recorder = getattr(request.context, "cost_recorder", None)
    root = request.candidate_artifact_root()
    data = result.geometry
    obj = _find_renderable_mesh(result.payload)
    if obj is not None:
        data = timed_call(recorder, "native_evaluation", evaluated_arrays, obj, recorder)
        path = timed_call(recorder, "serialization", write_obj, root / "m" / "evaluated.obj",
                          {"vertices": data.vertices, "faces": data.faces})
        import bpy
        artist = root / "artist.blend"
        timed_call(recorder, "serialization", bpy.ops.wm.save_as_mainfile,
                   filepath=str(artist), check_existing=False)
        result = replace(result, mesh_path=path, payload=None,
                         artifacts={**result.artifacts, "evaluated_mesh": path, "artist_source": artist})
    if result.mesh_path is None:
        return result
    if data is None or not getattr(request.context, "native_resident", True):
        vertices, faces = timed_call(recorder, "serialization", read_obj, result.mesh_path)
        data = GeometryArrays.capture(vertices, faces)
        if recorder is not None:
            recorder.count("obj_parses"); recorder.count("obj_read_bytes", result.mesh_path.stat().st_size)
    if recorder is not None:
        recorder.count("obj_boundary_write_bytes", result.mesh_path.stat().st_size)
    retessellation = None
    if request.config.get("retessellate_coplanar", False) and len(data.faces)>1000:
        from .planar_retessellation import retessellate_coplanar
        def progress(stage,count,total,elapsed):
            print(f"retessellation stage={stage} count={count}/{total} elapsed={elapsed:.2f}s",flush=True)
        simplified,retessellation = timed_call(recorder,"geometry_retessellation",retessellate_coplanar,data,
            timeout_s=float(request.config.get("retessellation_timeout_s",5.)),progress=progress)
        if simplified.content_hash != data.content_hash:
            original_path=result.mesh_path
            path=timed_call(recorder,"serialization",write_obj,root/"m"/"retessellated.obj",simplified)
            result=replace(result,mesh_path=path,artifacts={**result.artifacts,
                "original_evaluated_mesh":original_path,"retessellated_mesh":path})
            data=simplified
    cache = getattr(request.context, "geometry_cache", None)
    if cache is None:
        cache = GeometryCache()
        if request.context is not None:
            request.context.geometry_cache = cache
    hits = cache.hits
    topology = timed_call(recorder, "topology", cache.topology_report, data)
    if recorder is not None:
        recorder.count("topology_cache_hits", cache.hits-hits)
    per_view = result.metric_result.per_view
    from .evidence_identity import target_evidence_hash
    extras = {**result.metric_result.extras, "input_evidence_hash":target_evidence_hash(request.target),
              "backend_native_metrics": result.metric_result.to_dict(),
              "selection_topology": topology, "geometry_content_hash": data.content_hash,
              "geometry_connectivity_hash": data.connectivity_hash,
              "projection_diagnostics_computed": bool(project_diagnostics),
              "planar_retessellation":retessellation}
    if retessellation and retessellation.get("status") == "retessellated":
        from .output_qualification import qualify_retained_output
        extras["retained_output_qualification"] = qualify_retained_output(data,request.config)
        extras["single_solid_qualified"] = False
        extras["qualification_scope"] = "actual retessellated output; original artist/source receipts cannot certify changed connectivity"
    if project_diagnostics:
        from .projected_metrics import projected_mesh_metrics
        per_view = timed_call(recorder, "projection", projected_mesh_metrics,
                              request.target, data.vertices, data.faces)
        extras["selection_evidence"] = "evaluated_mesh_polygon_projection"
        if recorder is not None:
            recorder.count("polygon_projection_faces", len(data.faces) * len(request.target.constraints))
    metric = replace(result.metric_result, per_view=per_view, area_iou_mean=0., area_iou_min=0.,
                     boundary_iou_mean=0., extras=extras)
    geometry = data
    if getattr(request.context, "native_resident", True):
        geometry = timed_call(recorder, "native_evaluation", NativeOwnedGeometry,
                              data, "Candidate_" + request.candidate_id, recorder)
    return replace(result, metric_result=metric, geometry=geometry, payload=None)


def render_evidence(result, request):
    """Render one owned mesh with fixed cameras; scalar/proxy scores never admit it."""
    from blender_blocking.config import RenderConfig
    from .visibility import evaluate_visible_pair
    from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views_detailed
    from blender_blocking.evaluation.comparable_geometry import read_obj
    from blender_blocking.evaluation.silhouette_eval import evaluate_silhouette_pair, SilhouetteGateConfig
    from blender_blocking.evaluation.cost_model import timed_call
    from blender_blocking.reconstruction.native_geometry import NativeOwnedGeometry
    from contextlib import contextmanager
    from PIL import Image
    import bpy
    root = request.candidate_artifact_root()
    recorder = getattr(request.context, "cost_recorder", None)
    owner = result.geometry if isinstance(result.geometry, NativeOwnedGeometry) else None
    @contextmanager
    def temporary_object():
        vertices, faces = timed_call(recorder, "serialization", read_obj, result.mesh_path)
        if recorder is not None:
            recorder.count("obj_parses"); recorder.count("obj_read_bytes", result.mesh_path.stat().st_size)
        mesh = bpy.data.meshes.new("SelectionEvidence")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist()); mesh.update()
        obj = bpy.data.objects.new("SelectionEvidence", mesh); bpy.context.collection.objects.link(obj)
        try:
            yield obj
        finally:
            bpy.data.objects.remove(obj, do_unlink=True); bpy.data.meshes.remove(mesh)
    calibration = request.target.extras.get("view_calibration", {})
    if not calibration:
        raise ValueError("measured routing requires recorded camera calibration")
    size = request.target.constraints[0].mask.shape
    cfg = RenderConfig(resolution=(size[1], size[0]), view_calibration=dict(calibration), force_material=True)
    started = time.perf_counter()
    with owner.rendered() if owner is not None else temporary_object() as obj:
        rendered = timed_call(recorder, "render", render_orthogonal_views_detailed,
                             str(root / "sel"), target_objects=[obj], render_config=cfg,
                             views=[c.view for c in request.target.constraints])
        per_view = {}
        for c in request.target.constraints:
            with Image.open(rendered.paths[c.view]) as source:
                pixels = np.asarray(source.convert("RGBA"))
            mask = pixels[:, :, 3] > 127
            per_view[c.view] = evaluate_visible_pair(np.asarray(c.mask, bool), mask, c, view=c.view,
                config=SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.1,
                                           max_signed_distance_loss=.1), required=True)
            from .feature_evidence import known_empty_feature_guard
            features = known_empty_feature_guard(c, mask)
            per_view[c.view]["known_empty_features"] = features
            if not features["passed"]:
                per_view[c.view].update(passed=False, reason="required known-empty feature filled")
                per_view[c.view]["pass"] = False
            per_view[c.view]["candidate_projection_source"] = "fresh_blender_render_fixed_capture_camera"
    extras = {**result.metric_result.extras, "selection_evidence": "fresh_blender_render",
              "selection_render_s": time.perf_counter() - started,
              "selection_render_metadata": rendered.to_dict()}
    if result.metric_result.extras.get("projection_diagnostics_computed"):
        extras["polygon_selection_metrics"] = result.metric_result.per_view
    metric = replace(result.metric_result, per_view=per_view, area_iou_mean=0., area_iou_min=0.,
                     boundary_iou_mean=0., extras=extras)
    return replace(result, metric_result=metric, render_paths={k: Path(v) for k, v in rendered.paths.items()})


def quality_key(result):
    m=result.metric_result
    return (m.area_iou_min,m.area_iou_mean,m.boundary_iou_mean,-(m.elapsed_s if m.elapsed_s is not None else float("inf")))


def routing_run(requests, total_timeout_s=45., max_render_candidates=3, *, executor=None):
    """Admit bounded waves on the same isolated queue used by normal ensembles."""
    from .projection_contract import observed_holes
    from .process_executor import PersistentProcessExecutor, candidate_payload, candidate_outcome
    from .types import CandidateResult
    from contextlib import nullcontext
    started = time.perf_counter()
    deadline = None if total_timeout_s is None else time.time() + float(total_timeout_s)
    holes = observed_holes(requests[0].target) if requests else {}
    negative_space = any(v >= 4 for v in holes.values())
    from .geometry_selection import prefer_candidate
    from .feature_evidence import hole_capable_request
    from .quality_config import quality_config
    requests = [replace(r, config=quality_config(r.config)) for r in requests]
    structural_routing = any(r.config.get("routing_policy") == "structure_v1" for r in requests)
    proposal_screen = {}
    screening_errors = []
    if structural_routing:
        shape_requests = [r for r in requests if r.backend_name == "shape_program"]
        if shape_requests:
            try:
                from blender_blocking.primitives.shape_program import ShapeProgram
                from .program_proposals import whole_primitive_programs
                from .proposal_screening import screen_whole_programs
                seed = ShapeProgram("shape-program-v1", "routing-support", ())
                proposals = [] if negative_space else whole_primitive_programs(shape_requests[0].target, seed, max_elapsed_s=1.5)
                from .polygon_proposals import planar_extrusion_programs
                from .sweep_proposals import generalized_sweep_programs
                if shape_requests[0].config.get('generalized_sweep_search',False):
                    proposals += generalized_sweep_programs(shape_requests[0].target,seed)
                try:
                    proposals += planar_extrusion_programs(shape_requests[0].target,seed)
                except (ImportError,RuntimeError,ValueError) as exc:
                    screening_errors.append({"family":"polygon_extrusion","error":str(exc)})
                screened = screen_whole_programs(shape_requests[0].target, proposals)
                if screened:
                    _, program, evidence = screened[0]
                    proposal_screen = {"status":"prepared", "family_availability":screening_errors, **evidence,
                                       "proposal":program.metadata.get("proposal")}
                    requests = [replace(r, config={**r.config, "routed_program":program,
                        "routing_proposal_evidence":evidence}) if r.backend_name == "shape_program" else r for r in requests]
                else:
                    proposal_screen = {"status":"no_complementary_whole_proposal","family_availability":screening_errors}
            except Exception as exc:
                proposal_screen = {"status":"unavailable", "error":str(exc)}
    fast = {"profile_loft", "visual_hull_voxel"}
    ordered = sorted(requests, key=lambda r: (
        0 if r.backend_name == "visual_hull_voxel" and negative_space else
        1 if r.backend_name in fast else
        2 if structural_routing and r.backend_name == "shape_program" else
        4 if r.backend_name == "implicit_residual" else 3))
    results = []
    best = None
    renders = 0
    submitted = 0
    stop_reason = "candidate_pool_exhausted"
    candidate_wall = {}
    manager = nullcontext(executor) if executor is not None else PersistentProcessExecutor(2)
    with manager as pool:
        pending = []
        index = 0
        while index < len(ordered) or pending:
            while (index < len(ordered) and len(pending) < pool.max_workers and
                   (renders + len(pending) < max_render_candidates or not pending)):
                request = ordered[index]
                # This explicitly requested residual depends on all prior source
                # candidates, not whichever parallel seed happens to finish first.
                if request.backend_name == "implicit_residual" and pending:
                    break
                index += 1
                reason = None
                if negative_space and not hole_capable_request(request):
                    reason = "observed_negative_space_incompatible_with_single_loft"
                elif deadline is not None and time.time() >= deadline:
                    reason = "routing_budget_exhausted_before_candidate"
                    stop_reason = "routing_deadline"
                elif (not structural_routing and best is not None and
                      best.metric_result.area_iou_min >= .94 and best.metric_result.area_iou_mean >= .97):
                    reason = "validated_fast_candidate_sufficient; refinement_not_admitted"
                    stop_reason = "validated_quality_stop"
                elif renders + len(pending) >= max_render_candidates:
                    reason = "fixed_camera_render_budget_exhausted"
                    stop_reason = "fixed_camera_render_budget_exhausted"
                if reason:
                    results.append(CandidateResult(request.candidate_id, request.backend_name, "skipped", warnings=(reason,)))
                    continue
                if request.backend_name == "differentiable_refine" and request.config.get("backend") == "dvx":
                    seeds = {r.backend_name:r for r in results if r.succeeded and r.geometry is not None}
                    if seeds:
                        request = replace(request,config={**request.config,"seed_results":seeds})
                if request.backend_name == "implicit_residual":
                    seeds = {r.backend_name:r for r in results if r.succeeded and r.geometry is not None}
                    if seeds and not request.config.get("seed_results"):
                        request = replace(request,config={**request.config,"seed_results":seeds})
                if request.backend_name == "hybrid_loft_hull":
                    seeds = {r.backend_name: r for r in results if r.backend_name in {"profile_loft", "visual_hull_voxel"}}
                    if seeds:
                        request = replace(request, config={**request.config, "seed_results": seeds})
                job_id = pool.submit("candidate", candidate_payload(request, pool, measured=True),
                                     timeout_s=request.budget.timeout_s, deadline=deadline)
                pending.append((request, job_id, time.perf_counter()))
                submitted += 1
            if not pending:
                continue
            # Consume in admission order for reproducible worst-view guarded selection.
            # Other completed results are retained by the coordinator while this waits.
            request, job_id, candidate_started = pending.pop(0)
            result = candidate_outcome(request, pool.result(job_id))
            if result.metric_result.extras.get("selection_evidence") == "fresh_blender_render":
                renders += 1
            measured = result.metric_result.extras.get("selection_evidence") == "fresh_blender_render"
            passed = bool(result.metric_result.per_view) and all(
                v["passed"] for v in result.metric_result.per_view.values())
            if result.succeeded and not (measured and passed):
                result = replace(result, status="failed", errors=(*result.errors,
                    "routing candidate lacks passing fresh fixed-camera evidence"))
            if result.succeeded and measured and passed:
                if best is None:
                    best = result
                elif prefer_candidate(result, best, policy="structure_v1" if structural_routing else "silhouette_only"):
                    best = result
                else:
                    result = replace(result, warnings=(*result.warnings,
                        "refinement_not_admitted: no mean gain under worst-view guard"))
            candidate_wall[result.candidate_id] = time.perf_counter() - candidate_started
            results.append(result)
    by_id = {r.candidate_id: r for r in results}
    results = [by_id[r.candidate_id] for r in ordered]
    ledger = {
        "routing": "recorded_camera_measured_process_v3",
        "routing_policy": "structure_v1" if structural_routing else "silhouette_only",
        "proposal_screen": proposal_screen,
        "render_allowance_scope": "actual fresh render completions plus in-flight reservations; construction-only failures release slots", "observed_hole_pixels": holes,
        "rendered_candidates": renders, "submitted_candidates": submitted,
        "candidate_full_wall_s": candidate_wall, "max_render_candidates": max_render_candidates,
        "max_workers": pool.max_workers, "budget_s": total_timeout_s,
        "budget_boundary": "coordinated worker deadlines; completed results retained independently",
        "full_elapsed_s": time.perf_counter() - started, "termination": stop_reason,
        "selected_backend": best.backend_name if best else None,
        "admission": ("measured required views and known-empty features pass; compactness prior within .006 evidence tolerance"
                      if structural_routing else "mean gain > .002; worst-view drop <= .002; all measured required views pass"),
        "confidence_limit": "finite candidate pool; silhouette evidence cannot establish hidden concavity or artist usability",
        "candidate_reasons": {r.candidate_id: list(r.warnings) + list(r.errors) for r in results},
    }
    if best:
        best = replace(best, metric_result=replace(best.metric_result,
            extras={**best.metric_result.extras, "routing_ledger": ledger}))
        results = [best if r.candidate_id == best.candidate_id else r for r in results]
    return results, best, ledger
