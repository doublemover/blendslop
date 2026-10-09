"""Lazy CPU DVX adapter. Dependency installation and execution remain opt-in.

Authoritative API: https://github.com/mworchel/differentiable-voxelization
DVX returns winding grids in z,y,x order over [-1,1]^3.
"""
from __future__ import annotations
from importlib import metadata
import sys
import time
import numpy as np

REQUIRED = {'dvx-python': '0.1.1', 'torch': '2.14.1+cpu'}
CPU_WHEEL_INDEX = 'https://download.pytorch.org/whl/cpu'


def dependency_state():
    found = {}
    for package in REQUIRED:
        try: found[package] = metadata.version(package)
        except metadata.PackageNotFoundError: found[package] = None
    compatible = sys.version_info[:2] == (3, 13) and all(found[k] == v for k, v in REQUIRED.items())
    try: geometry_version=metadata.version('shapely')
    except metadata.PackageNotFoundError: geometry_version=None
    try:
        geometry_tuple=tuple(int(v) for v in geometry_version.split('.')[:2]) if geometry_version is not None else ()
        projected_available=(2,1)<=geometry_tuple<(3,0)
    except ValueError:
        projected_available=False
    return {'available': compatible, 'required_python': 'CPython 3.13 CPU', 'platform': sys.platform,
        'python': sys.version.split()[0], 'required': REQUIRED, 'installed': found,
        'projected_mesh_rays_available':compatible and projected_available,'shapely_version':geometry_version,
        'cpu_wheel_index': CPU_WHEEL_INDEX, 'installation_performed': False,
        'api_source': 'https://github.com/mworchel/differentiable-voxelization'}


def helper_call(python_executable, payload=None, *, timeout_s=10., warm=False,
                ownership_root=None):
    """Approved interpreter; cold transport retains owned diagnostics."""
    if warm:
        from blender_blocking.reconstruction.differentiable.helper_session import dvx_session
        return dvx_session(python_executable).call(payload, timeout_s=timeout_s)
    import json
    import os
    import pickle
    import subprocess
    from pathlib import Path
    from reconstruction.process_executor import read_packet
    from utils.run_ownership import OwnedRun
    from utils.primary_process_cleanup import finish_primary_process
    repository = Path(__file__).resolve().parents[3]
    script = repository / "scripts" / "dvx_worker.py"
    executable = Path(os.path.abspath(python_executable))
    if not executable.is_file():
        raise FileNotFoundError(executable)
    owner = OwnedRun(ownership_root or repository / "temp/cold-dvx",
                     producer="cold_dvx_helper", max_generated_bytes=268435456,
                     shared_inputs={"helper_interpreter": str(executable),
                                    "helper_script": str(script)})
    root = owner.root
    process = None
    stdout = stderr = ""
    cleanup_receipt = None
    pipes_drained = False
    error = None
    started = time.monotonic()
    try:
        args = [str(executable), "-B", str(script)]
        if payload is None:
            args.append("--probe")
        else:
            payload = dict(payload)
            paths = list(payload.get("progress_paths", ()))
            paths.append(str(root / "progress.pkl"))
            payload["progress_paths"] = paths
            packet = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            owner.reserve_bytes(len(packet) + 32768)
            (root / "input.pkl").write_bytes(packet)
            owner.register_file("input.pkl", "disposable")
            args += ["--input", str(root / "input.pkl"),
                     "--output", str(root / "output.pkl")]
        command = json.dumps({"args": args, "timeout_s": timeout_s}, indent=2)
        owner.reserve_bytes(len(command.encode()) + 32768)
        (root / "command.json").write_text(command, encoding="utf-8")
        owner.register_file("command.json", "diagnostic")
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            pipes_drained = True
        except subprocess.TimeoutExpired:
            stdout, stderr, cleanup_receipt = finish_primary_process(process)
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout or ""
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
            owner.mark_failed("helper_timeout")
            if (root / "progress.pkl").exists():
                retained = read_packet(root / "progress.pkl")["value"]
                return {**retained, "partial": True, "stop_reason": "helper_timeout",
                        "final_update_evaluated": False}
            raise TimeoutError("DVX helper exceeded its shared time allowance; no scored checkpoint")
        if process.returncode:
            owner.mark_failed("helper_failed")
            raise RuntimeError("DVX helper failed: " + stderr[-3000:])
        return json.loads(stdout) if payload is None else read_packet(root / "output.pkl")
    except BaseException as exc:
        error = exc
        raise
    finally:
        auxiliary = []
        child_joined = process is None
        try:
            if process is not None:
                if cleanup_receipt is None:
                    drained_out, drained_err, cleanup_receipt = finish_primary_process(
                        process, drain_pipes=not pipes_drained)
                    if not pipes_drained:
                        stdout = drained_out.decode(errors="replace") if isinstance(drained_out, bytes) else drained_out or ""
                        stderr = drained_err.decode(errors="replace") if isinstance(drained_err, bytes) else drained_err or ""
                child_joined = cleanup_receipt["primary_joined"]
                if not cleanup_receipt["transport_closed"] or cleanup_receipt["errors"]:
                    raise RuntimeError("Cold helper cleanup remains unconfirmed: " +
                                       "; ".join(cleanup_receipt["errors"]))
        except BaseException as join_error:
            cleanup_receipt = getattr(join_error, "primary_process_receipt", cleanup_receipt)
            child_joined = bool(cleanup_receipt and cleanup_receipt["primary_joined"])
            if cleanup_receipt and cleanup_receipt["errors"]:
                auxiliary.append(RuntimeError("Cold helper cleanup remains unconfirmed: " +
                                               "; ".join(cleanup_receipt["errors"])))
            elif join_error is not error:
                auxiliary.append(join_error)
        # An unconfirmed child keeps an active lease, even if cancellation or
        # diagnostic publication failed. Recovery requires a separate decision.
        failure = error or (auxiliary[0] if auxiliary else None)
        if failure is not None:
            owner.mark_failed(repr(failure))
        try:
            for name, log in (("stdout-tail.txt", stdout), ("stderr-tail.txt", stderr)):
                (root / name).write_bytes(log.encode(errors="replace")[-8192:])
                owner.register_file(name, "diagnostic")
            summary = {"returncode": None if process is None else process.returncode,
                       "elapsed_s": time.monotonic() - started,
                       "child_joined": child_joined, "primary_cleanup": cleanup_receipt,
                       "state": "succeeded" if owner.state == "active" else owner.state,
                       "error": owner.error}
            (root / "helper-result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            owner.register_file("helper-result.json", "diagnostic")
            for name in ("output.pkl", "progress.pkl"):
                if (root / name).exists():
                    owner.register_file(name, "diagnostic")
        except Exception as publication_error:
            auxiliary.append(publication_error)
        owner.auxiliary_errors.extend(repr(exc) for exc in auxiliary)
        if child_joined and (cleanup_receipt is None or cleanup_receipt["transport_closed"]):
            try:
                owner.close(error=error or (auxiliary[0] if auxiliary else None))
            except Exception as close_error:
                auxiliary.append(close_error)
        if error is not None:
            for secondary in auxiliary:
                if hasattr(error, "add_note"):
                    error.add_note("Cold helper lifecycle also failed: " + repr(secondary))
        elif auxiliary:
            raise auxiliary[0]


def tiny_gradient_check(*, approved=False):
    if not approved:
        raise PermissionError('DVX forward/backward qualification requires explicit dependency installation approval')
    state = dependency_state()
    if not state['available']: raise RuntimeError(str(state))
    import os
    import torch
    import dvx.torch as dvx
    torch.set_num_threads(max(1, min(4, int(os.environ.get('OMP_NUM_THREADS', 4)))))
    v = torch.tensor([[-.4, -.4, -.4], [.5, -.4, -.4], [0., .5, -.4], [0., 0., .5]], dtype=torch.float32, requires_grad=True)
    f = torch.tensor([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=torch.int64)
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError('DVX qualification requires the exact CPU Torch wheel')
    grid = dvx.voxelize(16, v, f)
    raw_loss = grid.square().mean()
    raw_loss.backward(retain_graph=True)
    raw_gradient = v.grad.detach().clone()
    v.grad.zero_()
    weights = torch.linspace(.3, 1.7, 16**3).reshape(16, 16, 16)
    loss = (grid*weights).mean(); loss.backward()
    passed = bool(grid.shape == (16, 16, 16) and torch.isfinite(grid).all() and torch.isfinite(v.grad).all() and v.grad.abs().max() > 0 and torch.isfinite(raw_gradient).all() and raw_gradient.abs().max() > 0)
    return {'passed': passed, 'resolution': 16, 'loss': float(loss.detach()), 'gradient_max': float(v.grad.abs().max()),
            'raw_loss': float(raw_loss.detach()), 'raw_gradient_max': float(raw_gradient.abs().max()),
            'weighted_nonconstant_loss': True, 'dependencies': state}


def fit_job(payload):
    """One full helper optimization; target and connectivity cross IPC once."""
    if not payload.get('execution_approved', False):
        raise PermissionError('DVX dependency execution has not been approved')
    if payload.get('helper_python'):
        forwarded = dict(payload)
        executable = forwarded.pop('helper_python')
        return helper_call(executable, forwarded, timeout_s=payload.get('timeout_s') or 30.,
                           warm=bool(payload.get('helper_warm',False)))
    state = dependency_state()
    if not state['available']: raise RuntimeError(str(state))
    if payload.get('objective')=='observed_projected_rays' and not state.get('projected_mesh_rays_available',False):
        raise RuntimeError('projected mesh rays require approved optional Shapely>=2.1 geometry runtime')
    import os
    import torch
    import dvx.torch as dvx
    torch.set_num_threads(max(1, min(4, int(os.environ.get('OMP_NUM_THREADS', 4)))))
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError('DVX qualification requires the exact CPU Torch wheel')
    if payload.get('objective') in {'observed_rays', 'observed_projected_rays', 'filtered_occupancy'}:
        # The standalone CPU helper deliberately loads only numeric modules;
        # importing the reconstruction package would require unrelated cv2/bpy.
        from pathlib import Path
        import importlib,types
        package='blendslop_numeric_dvx'
        if package not in sys.modules:
            namespace=types.ModuleType(package)
            namespace.__path__=[str(Path(__file__).resolve().parent)]
            sys.modules[package]=namespace
        solver=importlib.import_module(package+'.conditioned_solver')
        return solver.fit_conditioned_job(payload,state)
    v_world, faces = np.asarray(payload['vertices'], float), np.asarray(payload['faces'], np.int64)
    center = np.asarray(payload['center'], float); scale = float(payload['scale'])
    normalized = (v_world-center)/scale
    if np.max(np.abs(normalized)) >= 1.: raise ValueError('fixed DVX transform must contain every vertex inside [-1,1]^3')
    v = torch.tensor(normalized, dtype=torch.float32, requires_grad=True)
    seed = v.detach().clone(); f = torch.tensor(faces, dtype=torch.int64)
    target = torch.tensor(np.asarray(payload['target_zyx'], np.float32))
    valid = torch.tensor(np.asarray(payload.get('target_valid_zyx', np.ones(target.shape)), np.float32))
    if valid.shape != target.shape or not torch.isfinite(valid).all() or valid.sum() <= 0:
        raise ValueError('DVX target requires finite observed-cell weights')
    n = target.shape[0]
    if target.shape != (n, n, n) or n not in (16, 32, 64): raise ValueError('DVX resolution must be 16/32/64')
    optimizer = torch.optim.Adam([v], lr=float(payload.get('learning_rate', .005)))
    best, best_loss, history = seed.clone(), float('inf'), []
    started = time.perf_counter()
    requested = min(64, max(1, int(payload.get('steps', 12))))
    updates, evaluations, best_index = 0, 0, None
    stop_reason = "requested_steps"
    final_evaluated = False
    import os, pickle, uuid
    from pathlib import Path

    def timed_out():
        return payload.get('timeout_s') is not None and time.perf_counter()-started >= payload['timeout_s']

    def snapshot():
        return {'vertices': best.numpy()*scale+center, 'faces': faces, 'history': list(history),
            'dependencies': state, 'fixed_transform': {'center': center.tolist(), 'scale': scale},
            'layout': 'DVX z,y,x', 'target_transferred_once': True, 'device': 'cpu',
            'requested_steps': requested, 'optimizer_updates': updates,
            'objective_evaluations': evaluations, 'best_evaluation': best_index,
            'stop_reason': stop_reason, 'optimization_wall_s': time.perf_counter()-started,
            'final_update_evaluated': final_evaluated, 'partial': stop_reason != "requested_steps"}

    def evaluate():
        nonlocal best, best_loss, best_index, evaluations
        evaluations += 1
        grid = dvx.voxelize(n, v, f)
        loss = (((grid-target)**2)*valid).sum()/valid.sum()+.02*((v-seed)**2).mean()
        if torch.isfinite(loss):
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best, best_loss, best_index = v.detach().clone(), value, evaluations
            value = snapshot()
            for name in payload.get('progress_paths', ()):
                path = Path(name)
                temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
                with temporary.open('wb') as stream:
                    pickle.dump({'status': 'partial', 'value': value,
                                 'recorded_evaluations': evaluations}, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, path)
        return loss

    for step in range(requested):
        if timed_out():
            stop_reason = "elapsed_time_budget"
            break
        optimizer.zero_grad(set_to_none=True)
        loss = evaluate()
        if not torch.isfinite(loss):
            stop_reason = "nonfinite_loss"
            break
        loss.backward()
        if v.grad is None or not torch.isfinite(v.grad).all():
            stop_reason = "nonfinite_or_missing_gradient"
            break
        optimizer.step()
        updates += 1
        with torch.no_grad():
            v.clamp_(-.98, .98)
    # The final update belongs to this same allowance, not a new search budget.
    if updates == requested and not timed_out():
        with torch.no_grad():
            loss = evaluate()
        final_evaluated = bool(torch.isfinite(loss))
        if not final_evaluated:
            stop_reason = "nonfinite_final_loss"
    elif updates == requested:
        stop_reason = "elapsed_time_budget_before_final_evaluation"
    if not history:
        raise TimeoutError('DVX allowance exhausted without a finite scored state')
    return snapshot()


def run_candidate(request):
    from dataclasses import replace
    from reconstruction.types import CandidateResult, CandidateMetrics
    started = time.perf_counter()
    timeout = request.budget.timeout_s
    def remaining():
        return None if timeout is None else max(0., timeout-(time.perf_counter()-started))
    helper = request.config.get('dvx_helper_python')
    state = dependency_state()
    if helper and request.config.get('dvx_execution_approved', False):
        try:
            state = helper_call(helper, timeout_s=min(10., timeout or 10.),
                                warm=bool(request.config.get('dvx_warm_helper',False)))
        except Exception as exc:
            state = {**state, 'available': False, 'helper_error': str(exc)}
    # No implicit installation, execution, or fallback pretending to be DVX.
    if not state['available'] or not request.config.get('dvx_execution_approved', False):
        return CandidateResult(request.candidate_id, request.backend_name, 'skipped',
            warnings=('DVX requires approved CPython3.13 CPU dependency qualification; nothing installed or executed',),
            metric_result=CandidateMetrics(extras={'dvx': state, 'qualification': 'not_run'}))
    from blender_blocking.reconstruction.process_executor import executor_scope
    from reconstruction.mesh_io import combine_primitive_meshes
    from placement.resfit_initialization import initialize_ellipsoids_from_points, PrimitiveInitializationConfig
    from reconstruction.point_cloud import target_surface_points
    from reconstruction.native_geometry import GeometryArrays
    from reconstruction.projected_metrics import projected_mesh_metrics
    from blender_blocking.reconstruction.differentiable.seed_selection import select_existing_seed
    seed, seed_report = select_existing_seed(request.target,request.config.get('seed_results'),
        maximum_vertices=int(request.config.get('dvx_seed_vertex_limit',4096)))
    primitives = ()
    if seed is None:
        points, _ = target_surface_points(request.target, resolution=32, max_points=2048)
        primitives = tuple(initialize_ellipsoids_from_points(points, PrimitiveInitializationConfig(primitive_count=4)))
        mesh = combine_primitive_meshes(primitives, resolution=16)
        from reconstruction.grouped_solids import oriented_generated_mesh
        seed = oriented_generated_mesh(mesh)
        seed_report.update(seed_content_hash=seed.content_hash,initialization='four_ellipsoid_last_fallback')
    # Numeric retained meshes and artist sources remain distinct. Deformation
    # never claims to be replayed by unchanged primitive/program parameters.
    from reconstruction.differentiable.mesh_projection import _triangulated_mesh_faces
    parts, vertex_start, face_start = [], 0, 0
    for index, primitive in enumerate(primitives):
        part_mesh = primitive.to_mesh_data(16)
        vertex_count = len(part_mesh.vertices)
        face_count = len(_triangulated_mesh_faces(part_mesh.faces))
        parts.append({'part':index,'type':primitive.to_dict()['type'],
            'vertex_start':vertex_start,'vertex_count':vertex_count,'face_start':face_start,'face_count':face_count})
        vertex_start += vertex_count; face_start += face_count
    if not parts:
        parts=[{'part':0,'type':'retained_evaluated_seed_surface','vertex_start':0,
                'vertex_count':len(seed.vertices),'face_start':0,'face_count':len(seed.faces)}]
    n = int(request.config.get('dvx_resolution', 32))
    if n not in (16, 32, 64): raise ValueError('DVX resolution must be 16/32/64')
    lo, hi = np.asarray(request.target.bounds.to_min_max(), float)
    center = (hi+lo)/2.
    scale = max(float(np.max(hi-lo)/1.8), float(np.max(np.abs(seed.vertices-center))/.9))
    # The physical cube stays fixed. Original camera bounds must never be
    # replaced with that cube when querying masks, including legacy viewports.
    from blender_blocking.reconstruction.differentiable.ray_evidence import prepare_coverage_target,prepare_ray_targets
    objective=request.config.get('dvx_objective') or 'legacy_occupancy'
    if objective=='observed_projected_rays' and not state.get('projected_mesh_rays_available',False):
        return CandidateResult(request.candidate_id,request.backend_name,'skipped',
            warnings=('pinned CPU projected-ray extension requires optional Shapely>=2.1 in the selected helper runtime',))
    if objective=='observed_projected_rays' and len(seed.faces)>2048:
        from reconstruction.types import CandidateMetrics
        return CandidateResult(request.candidate_id,request.backend_name,'skipped',
            metric_result=CandidateMetrics(extras={'seed_content_hash':seed.content_hash,'projected_triangle_allowance':2048}),
            warnings=('retained seed exceeds bounded projected-union triangle allowance; simplify the actual seed separately or choose an explicit alternative',))
    from reconstruction.visibility import point_support
    anchors_observed=point_support(request.target,seed.vertices)[1]
    anchor_weights=1.+4.*(1.-anchors_observed/max(1,len(request.target.constraints)))
    levels=request.config.get('dvx_grid_levels') or [n]
    levels=[int(level) for level in levels]
    if sorted(set(levels)) != levels or any(level not in (16,32,64) for level in levels) or levels[-1] != n:
        raise ValueError('DVX grid levels must increase to the declared final resolution')
    if objective=='observed_rays':
        from blender_blocking.reconstruction.differentiable.ray_evidence import global_thin_ray_screen
        screens={level:global_thin_ray_screen(seed.vertices,scale,level) for level in levels}
        seed_report['ray_filter_scale_screen']=screens
        if not request.config.get('dvx_allow_subvoxel_ray_surrogate',False):
            levels=[level for level in levels if not screens[level]['global_envelope_below_two_voxels']]
            if not levels:
                from reconstruction.types import CandidateMetrics
                return CandidateResult(request.candidate_id,request.backend_name,'skipped',
                    metric_result=CandidateMetrics(extras={'thin_ray_screen':screens,'seed_content_hash':seed.content_hash}),
                    warnings=('retained globally thin seed: max-filtered-volume ray surrogate is not valid projected coverage at requested grids; choose direct planar/mesh evidence or explicitly opt into the surrogate',))
    stages=[]
    preparation_start=time.perf_counter()
    for level in levels:
        if remaining() is not None and remaining()<=.01:
            return CandidateResult(request.candidate_id,request.backend_name,'failed',
                errors=('DVX allowance exhausted during fixed-cube evidence preparation',))
        quadrature=1 if objective=='legacy_occupancy' else int(request.config.get('dvx_target_quadrature',2))
        def report(done,total,elapsed):
            print(f'DVX target grid={level} stage={done}/{total} elapsed={time.perf_counter()-preparation_start:.2f}s',flush=True)
        if objective in {'observed_rays','observed_projected_rays'}:
            target_data={'target_zyx':np.zeros((level,)*3,np.float32),
                'target_valid_zyx':np.zeros((level,)*3,np.float32),
                'voxel_target_unused_by_ray_objective':True}
            target_data['ray_targets']=prepare_ray_targets(request.target,center,scale,level,quadrature=quadrature,
                filter_mode=request.config.get('dvx_ray_filter_mode') or 'pixel_cell_box_exact',
                use_pixel_maps=request.config.get('pixel_evidence_mode')!='view_mean_legacy')
            report(1,1,time.perf_counter()-preparation_start)
        else:
            target_data=prepare_coverage_target(request.target,center,scale,level,quadrature=quadrature,progress=report)
        stages.append(target_data)
    payload={'vertices':seed.vertices,'faces':seed.faces,**stages[-1],
        'center':center,'scale':scale,'steps':request.config.get('dvx_steps',12),
        'timeout_s':remaining(),'execution_approved':True,'helper_python':helper}
    if objective!='legacy_occupancy':
        payload.update(objective=objective,stages=stages,
            parameterization=request.config.get('dvx_parameterization') or 'cage',
            differential_strength=float(request.config.get('dvx_differential_strength',4.)),
            anchor_weights=anchor_weights,part_ranges=parts,
            voxel_semantics='per_part_union_coverage' if primitives else 'retained_seed_winding_surface')
    target_preparation_s=time.perf_counter()-preparation_start
    from reconstruction.process_executor import progress_path
    checkpoint = progress_path()
    if checkpoint is not None:
        payload['progress_paths'] = [str(checkpoint)]
    if remaining() is not None and remaining() <= .01:
        return CandidateResult(request.candidate_id, request.backend_name, 'failed', errors=('DVX allowance exhausted during setup',))
    if helper and request.config.get('dvx_warm_helper',False):
        # The same owner that probed the CPU process submits the fit, avoiding
        # a second coordinator worker importing the operator in another helper.
        from reconstruction.process_executor import JobOutcome
        helper_started=time.perf_counter()
        forwarded=dict(payload);forwarded.pop('helper_python',None)
        fitted=helper_call(helper,forwarded,timeout_s=remaining() or 30.,warm=True)
        elapsed=time.perf_counter()-helper_started
        outcome=JobOutcome('success',fitted,elapsed_s=elapsed,
            partial=bool(fitted.get('partial')),total_wall_s=elapsed,stop_reason=fitted.get('stop_reason'))
    else:
        manager = executor_scope(1, context=request.context)
        with manager as executor:
            outcome = executor.map([('dvx_fit', payload, remaining())], timeout_s=remaining())[0]
    if outcome.status != 'success' and not outcome.partial:
        return CandidateResult(request.candidate_id, request.backend_name, 'failed', errors=(outcome.error,))
    fitted = dict(outcome.value)
    if outcome.partial:
        fitted.update(partial=True, stop_reason=outcome.stop_reason, final_update_evaluated=False)
    data = GeometryArrays.capture(fitted['vertices'], fitted['faces'])
    def metric(g): return CandidateMetrics(per_view=projected_mesh_metrics(request.target, g.vertices, g.faces))
    initial, final = metric(seed), metric(data)
    from reconstruction.grouped_solids import solid_guard
    proposed_guard = solid_guard(data)
    from blender_blocking.evaluation.triangle_contacts import within_part_boundary_guard
    part_guard = within_part_boundary_guard(data.vertices, data.faces, timeout_s=remaining())
    admitted = proposed_guard['valid_solid'] and part_guard['passed'] and final.area_iou_min >= initial.area_iou_min-.002 and final.area_iou_mean > initial.area_iou_mean+.002
    retained, measured = (data, final) if admitted else (seed, initial)
    from reconstruction.mesh_io import write_obj, write_primitive_set
    from reconstruction.differentiable.mesh_projection import _triangulated_mesh_faces
    artifacts = {}
    mesh_path = primitive_path = None
    root = request.candidate_artifact_root()
    metadata = {'contract': 'deformed_multipart_with_editable_initialization',
        'initialization_resolution': 16 if primitives else None, 'part_ranges': parts, 'seed_content_hash': seed.content_hash,
        'seed_selection':seed_report,'coordinate_model':fitted.get('coordinate_model'),
        'evidence_objective':fitted.get('evidence_objective','legacy_occupancy'),
        'voxel_semantics':fitted.get('voxel_semantics','legacy_summed_winding'),
        'stage_records':fitted.get('stages',[]),'target_preparation_s':target_preparation_s,
        'helper_session':fitted.get('helper_session'),
        'retained_content_hash': retained.content_hash, 'proposed_content_hash': data.content_hash,
        'fixed_transform': fitted['fixed_transform'], 'seed_parameters_reproduce_final_deformation': False,
        'single_solid_qualified': False, 'boundary_qualification': 'assembly union not checked',
        'proposed_within_part_boundary_guard': part_guard,
        'deformation_rejected_reason': part_guard.get('reason') if not part_guard['passed'] else None,
        'optimizer': {key: fitted.get(key) for key in ('requested_steps', 'optimizer_updates',
            'objective_evaluations', 'best_evaluation', 'stop_reason', 'final_update_evaluated', 'partial')},
        'history': fitted['history'], 'dependencies': state}
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        mesh_path = write_obj(root/'m'/'dvx.obj', retained)
        seed_path = write_obj(root/'seed'/'seed.obj', seed)
        state_path = root/'h'/'dvx-state.npz'
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(state_path, seed_vertices=seed.vertices, retained_vertices=retained.vertices,
                 proposed_vertices=data.vertices, faces=retained.faces)
        metadata.update(seed_mesh_path=str(seed_path.resolve()), retained_mesh_path=str(mesh_path.resolve()),
                        deformation_state_path=str(state_path.resolve()))
        source_artist=seed_report.get('source_primitive_path')
        if source_artist:
            from pathlib import Path
            import hashlib,shutil
            source_artist=Path(source_artist)
            if source_artist.is_file():
                artist_path=root/'seed'/'artist-source.json'
                shutil.copyfile(source_artist,artist_path)
                metadata.update(artist_source_path=str(artist_path.resolve()),
                    artist_source_sha256=hashlib.sha256(artist_path.read_bytes()).hexdigest())
                artifacts['artist_source_json']=artist_path
            else:
                metadata['artist_source_retention']='source artifact unavailable; retained seed mesh is preserved'
        primitive_path = write_primitive_set(root/'p'/'dvx-initialization.json', primitives, metadata=metadata)
        artifacts = {**artifacts,'primitive_json': primitive_path, 'mesh_obj': mesh_path,
                     'seed_mesh_obj': seed_path, 'deformation_state': state_path}
    return CandidateResult(request.candidate_id, request.backend_name, 'success', geometry=retained,
        mesh_path=mesh_path, primitive_path=primitive_path, artifacts=artifacts,
        warnings=('scored partial optimization retained: '+fitted['stop_reason'],) if fitted.get('partial') else (),
        metric_result=replace(measured, editability_score=.2, extras={'dvx': state,
            'full_geometry_admitted': admitted, 'proposed_solid_guard': proposed_guard,
            'proposed_within_part_boundary_guard': part_guard,
            'output_contract': metadata, 'single_solid_qualified': False,
            'metrics_refer_to_output_hash': retained.content_hash, 'history': fitted['history'],
            'fixed_transform': fitted['fixed_transform'], 'optimization_work': metadata['optimizer'],
            'queue_accounting': {'queue_s': outcome.queue_s, 'worker_execute_wall_s': outcome.elapsed_s,
                                 'total_wall_s': outcome.total_wall_s, 'stop_reason': outcome.stop_reason}}))
