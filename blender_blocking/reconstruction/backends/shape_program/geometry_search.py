"""Compile/render program alternatives through the run's coordinated executor."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np


def geometric_key(result, structural_score=0.):
    metrics = result.metric_result
    distances = [float(row["signed_distance_loss"]) for row in metrics.per_view.values() if row.get("signed_distance_loss") is not None]
    return (metrics.area_iou_min, metrics.area_iou_mean, metrics.boundary_iou_mean,
            -float(np.mean(distances)) if distances else -1., float(structural_score))


def evaluate_program_job(payload):
    import bpy
    from types import SimpleNamespace
    from .compiler_bridge import _compile_program
    from ...types import CandidateResult, CandidateMetrics
    from ...measured_selection import freeze_geometry, render_evidence
    from ...projection_contract import viewport_for_constraint
    from ...native_geometry import GeometryCache
    request, program = payload
    objects_before, meshes_before, collections_before = set(bpy.data.objects), set(bpy.data.meshes), set(bpy.data.collections)
    groups_before = set(bpy.data.node_groups)
    geometry = None
    try:
        compiled = _compile_program(replace(program,program_id=request.candidate_id),
                                    request.config, context=request.context)
        context = SimpleNamespace(blender_available=True,native_resident=True,geometry_cache=GeometryCache())
        target = request.target
        if not target.extras.get("view_calibration"):
            calibration = {}
            for constraint in target.constraints:
                axes, bounds = viewport_for_constraint(target,constraint)
                calibration[constraint.view] = {"projection":"orthographic","world_bounds":list(bounds),
                    "axes":list(axes),"world_units":"metres","orientation":"canonical_positive_axes",
                    "source":"typed_input_camera_viewport"}
            target = replace(target,extras={**target.extras,"view_calibration":calibration})
        request = replace(request,context=context,target=target)
        result = CandidateResult(request.candidate_id,"shape_program","success",payload=compiled.root_object,
                                 metric_result=CandidateMetrics(editability_score=1.))
        result = freeze_geometry(result,request,project_diagnostics=False)
        geometry = result.geometry
        result = render_evidence(result,request)
        data = geometry.data
        geometry.release(); geometry = None
        return replace(result,geometry=data,payload=None)
    finally:
        if geometry is not None:
            geometry.release()
        # Only this job's Blender resources are removed; no factory reset while a parent waits.
        for obj in set(bpy.data.objects)-objects_before:
            bpy.data.objects.remove(obj,do_unlink=True)
        for mesh in set(bpy.data.meshes)-meshes_before:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for group in set(bpy.data.node_groups)-groups_before:
            if group.users == 0:
                bpy.data.node_groups.remove(group)
        for collection in set(bpy.data.collections)-collections_before:
            if collection.users == 0 or not len(collection.objects):
                bpy.data.collections.remove(collection)


def parameter_variants(program, limit=6, fraction=.06, *, start_control=0, release_deformations=False):
    """Independent local size/translation/rotation moves with cyclic coverage."""
    from ...program_transforms import local_pose_edit, DIMENSION_KEYS
    eligible = [(i, n) for i, n in enumerate(program.root_nodes)
                if n.operation in {"add", "union", "subtract", "difference", "intersect", "intersection"}]
    # Interleave nodes before cycling to the next axis or parameter group.
    controls = [(i, n, kind, axis, direction)
                for kind in ('size','section_exponent','sweep_bend','translation','sweep_taper','rotation')
                for axis in (range(1) if kind=='section_exponent' else range(2) if kind in {'sweep_bend','sweep_taper'} else range(3))
                for i,n in eligible for direction in (-1,1)
                if (kind not in {'section_exponent','sweep_bend','sweep_taper'} or n.primitive_type=='generalized_sweep')
                and not (kind == 'size' and axis == 1 and n.primitive_type == 'capsule')]
    controls += [(i,n,kind,0,direction) for i,n in eligible
                 if n.primitive_type == 'rounded_triangle'
                 for kind in ('triangle_corner','triangle_dome_balance') for direction in (-1,1)]
    if release_deformations:
        controls += [(i,n,kind,0,direction) for i,n in eligible
                     if n.primitive_type=='deformed_superquadric'
                     for kind in ('taper_x','taper_y','bend_angle') for direction in (-1,1)]
    if not controls or limit <= 0:
        return ()
    variants = []
    for offset in range(min(limit, len(controls))):
        index, node, kind, axis, direction = controls[(start_control + offset) % len(controls)]
        params = deepcopy(dict(node.parameters))
        step = direction * fraction
        if kind == "size":
            dimension = DIMENSION_KEYS[axis]
            if node.primitive_type == 'rounded_triangle':
                from blender_blocking.primitives.rounded_triangle import RoundedTrianglePrimitive
                part = RoundedTrianglePrimitive.from_program_parameters(params, world=False)
                if axis < 2:
                    if dimension in params:
                        params[dimension] *= np.exp(step)
                    else:
                        scale = part.scale_xy.copy()
                        scale[axis] *= np.exp(step)
                        params['scale_xy'] = scale.tolist()
                else:
                    params[dimension] = part.thickness * np.exp(step)
            elif node.primitive_type == 'capsule':
                from blender_blocking.primitives.capsule import CapsulePrimitive
                part = CapsulePrimitive.from_program_parameters(params, world=False)
                if axis == 0:
                    params['radius_world'] = part.radius * np.exp(step)
                    params['segment_height_world'] = part.segment_height
                elif 'segment_height_world' in params:
                    params['segment_height_world'] = part.segment_height * np.exp(step)
                else:
                    params[dimension] = (part.segment_height + 2 * part.radius) * np.exp(step)
            else:
                params[dimension] = max(1e-6, float(params.get(dimension, 1.)) * np.exp(step))
            if "profile_curve" in params:
                rows = [dict(row) for row in params["profile_curve"]]
                keys = (("radius_x_world", "center_offset_world"),
                        ("radius_y_world", "center_y_offset_world"), ("z_world",))[axis]
                for row in rows:
                    for key in keys:
                        if key in row:
                            row[key] *= np.exp(step)
                params["profile_curve"] = tuple(rows)
        elif kind == 'triangle_corner':
            params['corner_radius_world'] = float(params.get('corner_radius_world', params.get('corner_radius', .16))) * np.exp(step)
        elif kind == 'triangle_dome_balance':
            balance = float(params.get('front_fraction', .5))
            odds = balance / (1 - balance) * np.exp(step)
            params['front_fraction'] = odds / (1 + odds)
        elif kind in {'taper_x','taper_y','bend_angle'}:
            from blender_blocking.primitives.deformed_superquadric import DeformedSuperquadricPrimitive
            part = DeformedSuperquadricPrimitive.from_program_parameters(params)
            value = float(params.get(kind,0.))+step
            bound = part.maximum_bend_angle() if kind=='bend_angle' else part.MAXIMUM_TAPER
            params[kind] = float(np.clip(value,-bound,bound))
        elif kind in {'section_exponent','sweep_bend','sweep_taper'}:
            if kind=='section_exponent':
                params['section_exponent']=float(np.clip(float(params.get('section_exponent',1.))*np.exp(step),.15,2.))
            else:
                knots=np.asarray(params['section_knots_normalized'],float).copy()
                if kind=='sweep_bend':knots[:,axis+1]+=step*.25*(1.-(2.*knots[:,0])**2)
                else:knots[:,axis+3]*=np.exp(step*2.*knots[:,0])
                params['section_knots_normalized']=knots.tolist()
        else:
            increment = np.zeros(3)
            increment[axis] = step
            if kind == "translation":
                increment[axis] *= float(params.get(DIMENSION_KEYS[axis], 1.))
                params = local_pose_edit(params, translation=increment)
            else:
                params = local_pose_edit(params, rotation_increment=increment)
        if node.primitive_type == 'deformed_superquadric':
            from blender_blocking.primitives.deformed_superquadric import DeformedSuperquadricPrimitive
            try:
                DeformedSuperquadricPrimitive.from_program_parameters(params)
            except ValueError:
                # A size/taper edit can invalidate a coupled bend-radius bound.
                # Skip it rather than aborting search or silently changing bend.
                continue
        nodes = list(program.root_nodes)
        nodes[index] = replace(node, parameters=params)
        variants.append(replace(program, root_nodes=tuple(nodes), metadata={**program.metadata,
            "refinement_control": {"node": node.node_id, "kind": kind, "axis": axis, "direction": direction}}))
    return tuple(variants)


def geometric_program_search(request, seed):
    from blender_blocking.primitives.program_search import search_shape_program_candidates, _program_signature
    from blender_blocking.reconstruction.process_executor import executor_scope
    started = time.perf_counter()
    construction_failures = []
    candidates = search_shape_program_candidates(seed,max_candidates=min(16,int(request.config.get("program_search_candidates",4))))
    entries = [("seed",seed,0.)] + [(candidate.candidate_id,candidate.program,candidate.score)
        for candidate in candidates.candidates if _program_signature(candidate.program) != _program_signature(seed)]
    if request.config.get('generalized_sweep_search',False) and request.config.get('routed_program') is None:
        from ...sweep_proposals import generalized_sweep_programs
        try:
            entries += [('sweep_'+str(i),p,0.) for i,p in enumerate(generalized_sweep_programs(request.target,seed))]
        except Exception as exc:
            construction_failures.append({'family':'generalized_sweep','error':str(exc)})
    if request.config.get("planar_extrusion_search", False) and request.config.get("routed_program") is None:
        from ...polygon_proposals import planar_extrusion_programs
        try:
            entries += [("planar_"+str(i),p,0.) for i,p in enumerate(planar_extrusion_programs(request.target,seed))]
        except Exception as exc:
            construction_failures.append({"family":"polygon_extrusion","error":str(exc)})
    if request.config.get("whole_support_search", False) and request.config.get("routed_program") is None:
        from ...program_proposals import whole_primitive_programs
        try:
            proposals = whole_primitive_programs(request.target, seed,
                max_evaluations=int(request.config.get("support_fit_evaluations", 96)),
                max_elapsed_s=min(3., float(request.budget.timeout_s or 3.)))
            entries += [("whole_"+str(i), p, 0.) for i, p in enumerate(proposals)]
        except Exception as exc:
            construction_failures.append({"family": "whole_support", "error": str(exc)})
    if request.config.get("cuboid_search", False) and request.config.get("routed_program") is None:
        from ...program_proposals import cuboid_programs
        entries += [("cuboid_level_"+str(i), p, 0.) for i, p in enumerate(cuboid_programs(request.target, seed))]
    if request.config.get("convex_proxy_search", False) and request.config.get("routed_program") is None:
        from ...convex_proxy import convex_proxy_programs
        try:
            entries += [("convex_level_"+str(i), p, 0.) for i, p in enumerate(convex_proxy_programs(request.target, seed))]
        except Exception as exc:
            construction_failures.append({"family": "convex", "error": str(exc)})
    if request.config.get("routed_program") is not None:
        # The parent already prepared numeric geometry. Do not regenerate or
        # execute a full grammar campaign to validate this one challenger.
        entries = [("screened_whole", seed, 0.)]
    # Interleave families so a grammar quota cannot silently remove all convex levels.
    family_entries = {}
    for entry in entries:
        family = ('sweep' if entry[0].startswith('sweep_') else 'planar' if entry[0].startswith('planar_') else 'whole' if entry[0].startswith('whole_') else 'cuboid' if entry[0].startswith('cuboid_')
                  else 'convex' if entry[0].startswith('convex_') else 'grammar')
        family_entries.setdefault(family, []).append(entry)
    interleaved = []
    while any(family_entries.values()):
        for family in ('planar','whole','sweep','grammar','cuboid','convex'):
            if family_entries.get(family):
                interleaved.append(family_entries[family].pop(0))
    entries = interleaved[:max(1,min(24,int(request.config.get("program_search_candidates",4))))]
    manager = executor_scope(int(request.config.get("program_workers", 2)), context=request.context)
    summaries, best, best_program, best_structural = [], None, seed, 0.
    seen = set()
    timeout = request.budget.timeout_s or float(request.config.get("program_timeout_s", 45.))
    def remaining():
        return None if timeout is None else max(0.,float(timeout)-(time.perf_counter()-started))
    with manager as executor:
        root = request.artifact_root or executor.root / "artifacts"
        def score(rows, stage):
            nonlocal best,best_program,best_structural
            unique = []
            for row in rows:
                signature = _program_signature(row[1])
                if signature not in seen:
                    unique.append(row)
                    seen.add(signature)
            rows = unique
            jobs = []
            for index,(label,program,structural) in enumerate(rows):
                trial = replace(request,candidate_id=f"{request.candidate_id}-{stage}-{index}",artifact_root=root,context=None)
                jobs.append(("program_geometry",(trial,program),remaining()))
            outcomes = executor.map(jobs,timeout_s=remaining())
            for (label,program,structural),outcome in zip(rows,outcomes):
                if outcome.status != "success":
                    summaries.append({"label":label,"stage":stage,"status":outcome.status,"error":outcome.error})
                    continue
                result = outcome.value
                key = geometric_key(result,structural)
                summaries.append({"label":label,"stage":stage,"status":"scored","geometric_key":key,
                                  "metrics":result.metric_result.to_dict(),"artifacts":result.to_dict()})
                if best is None or key > geometric_key(best,best_structural):
                    best,best_program,best_structural = result,program,structural
        score(entries,"grammar")
        if (best is not None and request.config.get("structural_search", False) and
            request.config.get("routed_program") is None):
            from ...program_proposals import structural_programs, subtractive_programs
            proposals = structural_programs(request.target, best_program, best.geometry, limit=4)
            if request.config.get("subtractive_search", False):
                proposals += subtractive_programs(request.target, best_program, best.geometry)
            score([(f"structure_{i}", p, 0.) for i, p in enumerate(proposals)], "structure")
        refinement_steps = 0 if request.config.get("routed_program") is not None else int(request.config.get("program_refinement_steps",1))
        for step in range(max(0,min(4,refinement_steps))):
            if best is None or (remaining() is not None and remaining() <= .1):
                break
            trial_count = min(24, int(request.config.get("program_refinement_trials", 6)))
            trials = parameter_variants(best_program, limit=trial_count,
                                       fraction=.06*.5**step, start_control=step*trial_count,
                                       release_deformations=bool(request.config.get("deformation_refinement",False)))
            if not trials:
                break
            score([(f"coordinate_{i}",program,best_structural) for i,program in enumerate(trials)],f"refine{step}")
    return best_program,best,{"selection":"actual_compiled_fixed_camera_geometry","attempts":summaries,
                              "elapsed_s":time.perf_counter()-started,"geometry_scored":best is not None, "construction_failures": construction_failures,
                              "scheduled_candidate_labels": [row[0] for row in entries]}
