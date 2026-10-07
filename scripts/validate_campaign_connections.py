"""Bounded native integration contracts; no reconstruction timing campaign."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'blender_blocking')]


def main():
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    import bpy, numpy as np
    from unittest.mock import patch
    from dataclasses import replace
    from types import SimpleNamespace
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays, GeometryArrays
    from blender_blocking.reconstruction.native_queries import ResidentQueryBatch
    from blender_blocking.reconstruction.grouped_solids import production_union, execute_union_pair, solid_guard
    from blender_blocking.reconstruction.native_qualification import qualify_geometry, toolchain_identity
    from blender_blocking.reconstruction.native_csg import boolean_mesh
    from blender_blocking.reconstruction.backends.hybrid_loft_hull import HybridLoftHullBackend
    from blender_blocking.reconstruction.types import CandidateRequest, CandidateResult, CandidateBudget
    from blender_blocking.test_quality_geometry import target_for_masks
    output = Path(sys.argv[sys.argv.index('--')+1]);output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=1.)
    a = evaluated_arrays(bpy.context.object)
    b = GeometryArrays.capture(a.vertices+(.5,0,0),a.faces)
    counts = lambda:(len(bpy.data.objects),len(bpy.data.meshes),len(bpy.data.node_groups))
    before = counts()
    with ResidentQueryBatch(a) as query:
        p = np.array([[2.,0,0],[3.,0,0]])
        query.proximity(p)
        identity = (query._state['obj'].as_pointer(),query._state['group'].as_pointer(),query._state['mesh'].as_pointer())
        moved = GeometryArrays.capture(a.vertices+(.25,0,0),a.faces)
        query.update_target(moved)
        np.testing.assert_allclose(query.proximity(p)['native_distance'],[1.25,2.25],atol=1e-5)
        changed_topology = GeometryArrays.capture(moved.vertices,moved.faces[::-1])
        query.update_target(changed_topology)
        np.testing.assert_allclose(query.proximity(p)['native_distance'],[1.25,2.25],atol=1e-5)
        assert identity == (query._state['obj'].as_pointer(),query._state['group'].as_pointer(),query._state['mesh'].as_pointer())
        stats = dict(query.stats)
        assert stats['target_coordinate_updates']==1 and stats['target_topology_rebuilds']==1 and stats['graph_builds']==1,stats
    assert counts()==before,'query state leaked'
    query = ResidentQueryBatch(a);query.proximity(p)
    try:
        query.proximity(np.array([[float('nan'),0,0]]))
        raise AssertionError('invalid query admitted')
    except ValueError:
        pass
    assert query._state is None and counts()==before,'failed query leaked'
    config = {'native_union_execution':True,'native_union_solver':'MANIFOLD','native_sdf_fallback':True,
        'native_qualification_python':str(ROOT/'.venv312/Scripts/python.exe'),'native_qualification_timeout_s':20.}
    union, report = production_union([a,b],config,timeout_s=60.,feature_thickness=1.)
    assert report['native_queue_used'] and report['balanced_unions'][0]['solver']=='MANIFOLD',report
    assert abs(solid_guard(union)['signed_volume']-1.5)<1e-4
    qa = qualify_geometry(a,python=config['native_qualification_python'],timeout_s=20.)
    qb = qualify_geometry(b,python=config['native_qualification_python'],timeout_s=20.)
    assert qa['manifold_validated'] and qb['manifold_validated']
    q = {'_toolchain_identity':toolchain_identity(config['native_qualification_python']),
         '_qualification_python':config['native_qualification_python'],a.content_hash:qa,b.content_hash:qb}
    changed = GeometryArrays.capture(b.vertices+(.01,0,0),b.faces)
    try:
        boolean_mesh(a,changed,operation='UNION',solver='MANIFOLD',qualification=q)
        raise AssertionError('stale qualification admitted')
    except ValueError:
        pass
    with patch('blender_blocking.reconstruction.native_csg.boolean_mesh',side_effect=RuntimeError('injected Boolean failure')):
        sdf, sdf_report = execute_union_pair((a,b,{'solver':'EXACT','sdf_fallback':True,'feature_thickness':1.}))
        assert sdf_report['algorithm']=='native_sdf_grid_csg' and solid_guard(sdf)['valid_solid'],sdf_report
        for thickness in (None,.001):
            try:
                execute_union_pair((a,b,{'solver':'EXACT','sdf_fallback':True,'feature_thickness':thickness}))
                raise AssertionError('unqualified thin SDF fallback admitted')
            except ValueError:
                pass
    masks = {view:np.pad(np.ones((16,16),bool),8) for view in ('front','side','top')}
    target = target_for_masks(masks)
    vertices = union.vertices.tolist()
    faces = []
    for face in union.faces:
        center = len(vertices);vertices.append(union.vertices[face].mean(axis=0).tolist())
        faces.extend([[int(face[i]),int(face[(i+1)%3]),center] for i in range(3)])
    dense_hull = GeometryArrays.capture(vertices,faces)
    seeds = {'profile_loft':CandidateResult('loft','profile_loft','success',geometry=a),
             'visual_hull_voxel':CandidateResult('hull','visual_hull_voxel','success',geometry=dense_hull)}
    request = CandidateRequest('resident_hybrid','hybrid_loft_hull',target,
        {'seed_results':seeds,'native_batch_queries':True,'hybrid_residual_parts':3},
        budget=CandidateBudget(timeout_s=15.),context=SimpleNamespace(blender_available=True))
    hybrid = HybridLoftHullBackend().reconstruct(request)
    query_rows = [r for r in hybrid.metric_result.extras['attempts'] if r.get('operation')=='resident_residual_queries']
    assert query_rows and query_rows[0]['statistics']['calls']>=2,hybrid.to_dict()
    from blender_blocking.primitives.shape_program import ShapeProgram,ShapeNode
    from blender_blocking.primitives.shape_program_compiler import compile_shape_program
    program = ShapeProgram('shape-program-v1','qualified-output',(
        ShapeNode('a','add','box',{'width_world':1.,'depth_world':1.,'height_world':1.}),
        ShapeNode('b','add','box',{'width_world':1.,'depth_world':1.,'height_world':1.,'x':.5})))
    compiled = compile_shape_program(program,bevel_modifier=False,weighted_normals=False,csg_options=config,timeout_s=60.)
    baked = evaluated_arrays(compiled.root_object)
    assert abs(solid_guard(baked)['signed_volume']-1.5)<1e-4,solid_guard(baked)
    assert 'baked_qualified_output' in compiled.output_contract
    receipt = {'blender':bpy.app.version_string,'resident_updates':stats,'resource_cleanup':True,
        'native_union':report,'stale_receipt_rejected':True,'sdf_fallback':sdf_report,
        'thin_unknown_sdf_rejected':True,'hybrid_repeated_queries':query_rows,
        'compiled_output_contract':compiled.to_dict(), 'scope':'component contracts only; no matched quality/timing rows'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('CAMPAIGN_CONNECTIONS_NATIVE_PASSED',flush=True)


if __name__ == '__main__':main()
