"""Reuse frozen loft/hull seeds, then admit actual hull-supported geometry."""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
from ..backend import BackendCapabilities, BaseBackend
from ..types import CandidateRequest, CandidateResult, CandidateMetrics


class HybridLoftHullBackend(BaseBackend):
    def __init__(self):
        super().__init__(name='hybrid_loft_hull', capabilities=BackendCapabilities(
            requires_blender=True, supports_pure_python=False, supports_multi_view=True,
            supports_top_view=True, supports_uncertainty=True, outputs_mesh=True,
            editability_score=.45))

    def reconstruct(self, request):
        from ..native_geometry import geometry_arrays, evaluated_arrays, NativeOwnedGeometry, GeometryArrays
        from ..native_csg import boolean_mesh
        from ..grouped_solids import balanced_union, solid_guard
        from ..projected_metrics import projected_mesh_metrics
        from ..spatial_regions import connected_point_regions
        started = time.perf_counter()
        seeds = dict(request.config.get('seed_results') or {})
        reused = bool(seeds)
        if not seeds:
            # Standalone calls still produce each seed once. Ensemble runs attach
            # the already completed hash-identified results on their shared queue.
            from .visual_hull import VisualHullBackend
            from .profile_loft import ProfileLoftBackend
            for name, backend in [('visual_hull_voxel', VisualHullBackend()), ('profile_loft', ProfileLoftBackend())]:
                config = dict(request.config)
                if name == 'profile_loft':
                    config = {'unit_scale': .01, 'num_slices': 10, 'num_samples': 100,
                        'radial_segments': 24, **config.get('profile_sampling', {}),
                        **config.get('mesh_from_profile', {}), 'quality_preset': request.config.get('quality_preset', 'default')}
                    from ..quality_config import quality_config
                    config = quality_config(config)
                seed_request = replace(request, candidate_id=request.candidate_id+'-'+name, backend_name=name, config=config)
                result = backend.reconstruct(seed_request)
                if result.succeeded and result.geometry is None and result.payload is not None:
                    result = replace(result, geometry=evaluated_arrays(result.payload))
                seeds[name] = replace(result, payload=None)
        successful = {name: r for name, r in seeds.items() if r.succeeded and r.geometry is not None}
        if not successful:
            return CandidateResult(request.candidate_id, self.name, 'failed', errors=('no valid loft/hull seed geometry',))
        def score(data):
            rows = projected_mesh_metrics(request.target, data.vertices, data.faces)
            metric = CandidateMetrics(per_view=rows)
            return (metric.area_iou_min, metric.area_iou_mean, metric.boundary_iou_mean), metric
        structure_first = request.config.get('hybrid_mode', 'hull_first') == 'structure_first'
        if structure_first and 'profile_loft' in successful:
            retained_name, retained = 'profile_loft', successful['profile_loft']
        else:
            retained_name, retained = max(successful.items(), key=lambda row: score(geometry_arrays(row[1].geometry))[0])
        best = geometry_arrays(retained.geometry)
        best_key, best_metrics = score(best)
        attempts = []
        editable_parts = []
        hull_result, profile_result = successful.get('visual_hull_voxel'), successful.get('profile_loft')
        if hull_result and profile_result:
            hull = geometry_arrays(hull_result.geometry)
            profile = geometry_arrays(profile_result.geometry)
            editable_parts.append(profile)
            try:
                clipped, clip_report = boolean_mesh(profile, hull, operation='INTERSECT')
                assembly = clipped if solid_guard(clipped)['valid_solid'] else profile
                key, metric = score(assembly)
                attempts.append({'operation': 'profile_intersect_hull', 'key': key, **clip_report})
                from ..feature_evidence import mesh_empty_features
                if (mesh_empty_features(request.target, assembly)['passed'] and
                    key[0] >= best_key[0]-.002 and key[1] > best_key[1]+.002):
                    best, best_key, best_metrics = assembly, key, metric
                elif structure_first:
                    assembly = best
                # Residual surface regions must be spatially supported by the hull.
                # This bounded addition never unions all unrepresented hull cells.
                from contextlib import ExitStack
                from ..native_geometry import GeometryCache
                from ..native_queries import ResidentQueryBatch
                from ..grouped_solids import production_union
                from blender_blocking.reconstruction.process_executor import current_worker_client, executor_scope
                from mathutils import Vector
                points = hull.vertices[np.linspace(0, len(hull.vertices)-1, min(4096, len(hull.vertices))).astype(int)]
                scale = float(np.ptp(hull.vertices, axis=0).max())
                from placement.resfit_initialization import initialize_ellipsoids_from_points, PrimitiveInitializationConfig
                from reconstruction.mesh_io import combine_primitive_meshes
                from scipy.spatial import cKDTree
                rejected = np.zeros(len(points), bool)
                query_stats = None
                with ExitStack() as resources:
                    native = request.config.get('native_batch_queries', False)
                    query = resources.enter_context(ResidentQueryBatch(assembly)) if native else None
                    queue = current_worker_client() or getattr(request.context, 'process_executor', None)
                    if request.config.get('native_union_execution', False):
                        queue = resources.enter_context(executor_scope(2, context=request.context))
                    cache = GeometryCache()
                    limit = min(4, int(request.config.get('hybrid_residual_parts', 3)))
                    for index in range(limit):
                        if request.budget.timeout_s and time.perf_counter()-started > request.budget.timeout_s*.8:
                            break
                        if query is not None:
                            query.update_target(assembly)
                            distances = query.proximity(points)['native_distance']
                        else:
                            tree = cache.scalar_bvh(assembly)
                            distances = np.array([tree.find_nearest(Vector(p))[3] for p in points])
                        remaining_points = (distances > scale*.035) & ~rejected
                        if structure_first:
                            from ..projected_metrics import projected_mesh_masks
                            from ..geometry_selection import missing_observed_points
                            predictions = projected_mesh_masks(request.target, assembly.vertices, assembly.faces)
                            remaining_points &= missing_observed_points(request.target, predictions, points)
                        regions = connected_point_regions(points[remaining_points], minimum=8, max_regions=4)
                        if not regions:
                            break
                        region = regions[0]
                        parts = initialize_ellipsoids_from_points(region, PrimitiveInitializationConfig(primitive_count=1))
                        mesh = combine_primitive_meshes(parts, resolution=16)
                        from ..grouped_solids import oriented_generated_mesh
                        addition = oriented_generated_mesh(mesh)
                        addition, _ = boolean_mesh(addition, hull, operation='INTERSECT')
                        if not solid_guard(addition)['valid_solid']:
                            rejected |= cKDTree(region).query(points)[0] < 1e-9
                            continue
                        remaining_s = None if not request.budget.timeout_s else max(.001, request.budget.timeout_s-(time.perf_counter()-started))
                        candidate, report = production_union([assembly, addition], request.config,
                                                             executor=queue, timeout_s=remaining_s)
                        key, metric = score(candidate)
                        admitted = (mesh_empty_features(request.target, candidate)['passed'] and
                                    key[0] >= best_key[0]-.002 and key[1] > best_key[1]+.002)
                        attempts.append({'operation': 'supported_residual_add', 'round': index,
                            'query_target_hash': assembly.content_hash, 'key': key, 'admitted': admitted, **report})
                        if admitted:
                            assembly = candidate
                            best, best_key, best_metrics = candidate, key, metric
                            editable_parts.append(addition)
                        else:
                            rejected |= cKDTree(region).query(points)[0] < 1e-9
                    if query is not None:
                        query_stats = dict(query.stats)
                        attempts.append({'operation': 'resident_residual_queries', 'statistics': query_stats,
                                         'scope': 'one hybrid case/process; changed assembly updates only'})
            except Exception as exc:
                attempts.append({'status': 'retained_valid_incumbent', 'error': str(exc)})
        # Store the retained mesh and editable assembly together, but export and
        # render only the actual combined output.
        owner = NativeOwnedGeometry(best, 'HybridCombined')
        obj = owner.attach()
        obj.use_fake_user = False
        for index, data in enumerate(editable_parts):
            part_owner = NativeOwnedGeometry(data, 'HybridEditablePart'+str(index))
            child = part_owner.attach(); child.parent = obj; child.use_fake_user = False
            child.hide_render = True; child.hide_set(True); child['blendslop_export_exclude'] = True
        extras = {'hybrid_strategy': 'structure_first_observed_residuals' if structure_first else 'hull_first_supported_residuals',
            'reused_seed_results': reused, 'seed_hashes': {name: geometry_arrays(r.geometry).content_hash for name, r in successful.items()},
            'retained_seed': retained_name, 'attempts': attempts, 'solid_guard': solid_guard(best),
            'topology': solid_guard(best), 'metrics_refer_to_output_hash': best.content_hash}
        return CandidateResult(request.candidate_id, self.name, 'success', geometry=best, payload=obj,
            metric_result=replace(best_metrics, editability_score=.45, elapsed_s=time.perf_counter()-started, extras=extras))
