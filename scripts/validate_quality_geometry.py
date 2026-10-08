"""Bounded actual-Blender integration checks for the quality geometry paths."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking')]


def main():
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    import bpy, numpy as np
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays, GeometryArrays
    from blender_blocking.reconstruction.grouped_solids import balanced_union, solid_guard
    from blender_blocking.reconstruction.native_queries import ResidentQueryBatch
    from blender_blocking.reconstruction.convex_proxy import native_convex_mesh
    from blender_blocking.primitives.shape_program import ShapeProgram, ShapeNode
    from blender_blocking.primitives.shape_program_compiler import compile_shape_program
    from blender_blocking.reconstruction.adaptive_geometry import hierarchical_hull
    from blender_blocking.test_quality_geometry import target_for_masks
    from blender_blocking.reconstruction.backends.hybrid_loft_hull import HybridLoftHullBackend
    from blender_blocking.reconstruction.types import CandidateRequest, CandidateResult, CandidateBudget
    from types import SimpleNamespace
    out = Path(sys.argv[sys.argv.index('--')+1]); out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects_before, meshes_before, groups_before = len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups)
    bpy.ops.mesh.primitive_cube_add(size=1.); a_obj = bpy.context.object
    a = evaluated_arrays(a_obj)
    bpy.ops.mesh.primitive_cube_add(size=1., location=(.5, 0., 0.)); b_obj = bpy.context.object
    b = evaluated_arrays(b_obj)
    union, report = balanced_union([a, b])
    guard = solid_guard(union)
    assert guard['valid_solid'], guard
    assert abs(guard['signed_volume']-1.5) < 1e-5, guard
    query_counts = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups))
    with ResidentQueryBatch(a) as query:
        p = np.array([[2., 0., 0.], [3., 0., 0.]])
        first = query.proximity(p)
        identity = (query._state['obj'].name, query._state['group'].name, query._state['mesh'].name)
        second = query.proximity(p+np.array([1., 0., 0.]))
        assert identity == (query._state['obj'].name, query._state['group'].name, query._state['mesh'].name)
        np.testing.assert_allclose(first['native_distance'], [1.5, 2.5], atol=1e-5)
        np.testing.assert_allclose(second['native_distance'], [2.5, 3.5], atol=1e-5)
    assert query_counts == (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups)), "resident query leaked resources"
    convex = native_convex_mesh(a.vertices)
    assert solid_guard(convex)['valid_solid'], solid_guard(convex)
    p = ShapeProgram('shape-program-v1', 'subtraction', (
        ShapeNode('positive', 'add', 'box', {'width_world': 1., 'depth_world': 1., 'height_world': 1.}),
        ShapeNode('negative', 'subtract', 'box', {'width_world': .4, 'depth_world': .4, 'height_world': 2.})))
    compiled = compile_shape_program(p, bevel_modifier=False, weighted_normals=False)
    cut = evaluated_arrays(compiled.root_object)
    cut_guard = solid_guard(cut)
    assert cut_guard['valid_solid'] and abs(cut_guard['signed_volume']-.84) < 1e-4, cut_guard
    assert any(obj.get('blendslop_export_exclude') for obj in compiled.objects)
    negative = next(obj for obj in compiled.objects if obj.get('blendslop_shape_node_operation') == 'subtract')
    negative.scale.x *= 1.2
    bpy.context.view_layer.update()
    response = solid_guard(evaluated_arrays(compiled.root_object))['signed_volume']
    assert response < cut_guard['signed_volume']-.01, response
    tree = ShapeProgram('shape-program-v1', 'balanced_live', tuple(
        ShapeNode('p'+str(i), 'add', 'box', {'width_world': 1., 'depth_world': 1., 'height_world': 1., 'x': i*.5})
        for i in range(4))+(ShapeNode('cut', 'subtract', 'box',
            {'width_world': .2, 'depth_world': .2, 'height_world': 2.}),))
    live = compile_shape_program(tree, bevel_modifier=False, weighted_normals=False)
    branches = [obj for obj in live.objects if obj.name.startswith('EditableUnionBranch')]
    assert len(branches) == 3, len(branches)
    live_guard = solid_guard(evaluated_arrays(live.root_object))
    assert live_guard['valid_solid'] and abs(live_guard['signed_volume']-2.46) < 1e-4, live_guard
    source = next(obj for obj in live.objects if obj.get('blendslop_shape_node_id') == 'p3')
    source.location.x += .2
    bpy.context.view_layer.update()
    live_response = solid_guard(evaluated_arrays(live.root_object))['signed_volume']
    assert live_response > live_guard['signed_volume']+.1, live_response
    mask = np.zeros((32, 32), bool); mask[8:24, 8:24] = True
    target = target_for_masks({'front': mask, 'side': mask, 'top': mask})
    from blender_blocking.reconstruction.backends.shape_program.geometry_search import evaluate_program_job
    scoped_before = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups), len(bpy.data.collections))
    scoped_request = CandidateRequest('scoped_csg', 'shape_program', target,
        config={'bevel_modifier': False, 'weighted_normals': False},
        artifact_root=out, budget=CandidateBudget(timeout_s=10.))
    scoped = evaluate_program_job((scoped_request, tree))
    assert scoped.geometry is not None
    assert scoped_before == (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.node_groups), len(bpy.data.collections)), 'scoped CSG job leaked native resources'
    seeds = {'profile_loft': CandidateResult('loft', 'profile_loft', 'success', geometry=a),
        'visual_hull_voxel': CandidateResult('hull', 'visual_hull_voxel', 'success', geometry=union)}
    request = CandidateRequest('hybrid_test', 'hybrid_loft_hull', target,
        config={'seed_results': seeds, 'hybrid_residual_parts': 1}, budget=CandidateBudget(timeout_s=5.),
        artifact_root=out, context=SimpleNamespace(blender_available=True))
    hybrid = HybridLoftHullBackend().reconstruct(request)
    assert hybrid.succeeded and hybrid.metric_result.extras['reused_seed_results']
    assert hybrid.geometry.content_hash == hybrid.metric_result.extras['metrics_refer_to_output_hash']
    from blender_blocking.reconstruction.differentiable.dvx_adapter import dependency_state
    receipt = {'blender': bpy.app.version_string, 'union': report, 'solid': guard,
        'resident_query_identity_reused': identity, 'scoped_program_resources_released': True, 'convex': solid_guard(convex),
        'subtractive_program': cut_guard, 'balanced_live_program': live_guard,
        'live_positive_edit_volume': live_response, 'live_negative_edit_volume': response, 'hybrid': hybrid.to_dict(), 'dvx': dependency_state()}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, default=str), encoding='utf-8')
    print('QUALITY_GEOMETRY_NATIVE_PASSED', flush=True)


if __name__ == '__main__': main()
