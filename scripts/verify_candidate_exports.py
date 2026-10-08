"""Inspect every available matrix output and persist failures under its real contract."""
from pathlib import Path
import argparse, copy, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'blender_blocking')]


def selected_backend(row):
    return row.get('selected_backend') or row['requested_mode']


def primitive_parameter_sources(row):
    source = Path(row['result_path']).parent
    if row['requested_mode'] != 'ensemble':
        return list((source/'artifacts/cand').glob('*/p/*.json'))
    payload = json.loads((source/'result.json').read_text())
    backend = payload.get('backend_result', {})
    selected = backend.get('selected') or backend.get('selected_result') or {}
    if isinstance(selected, str):
        selected = next((item for item in backend.get('candidates', [])
                         if item.get('candidate_id') == selected), {})
    path = selected.get('primitive_path') or selected.get('artifacts', {}).get('primitive_json')
    return [Path(path)] if path else []


def mesh_response(left, right):
    from scipy.spatial import cKDTree
    return max(float(cKDTree(left).query(right)[0].max()),float(cKDTree(right).query(left)[0].max()))


def export_row(row, output):
    import bpy, numpy as np
    from blender_blocking.evaluation.comparable_geometry import read_obj
    from blender_blocking.integration.blender_ops.export_qa import run_export_roundtrip_qa
    from blender_blocking.reconstruction.native_geometry import GeometryArrays
    from blender_blocking.reconstruction.native_qualification import qualify_geometry
    source = Path(row['result_path']).parent
    vertices, faces = read_obj(source/'artifacts/validated-mesh.obj')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mode = row['requested_mode']
    mesh = bpy.data.meshes.new(mode);mesh.from_pydata(vertices.tolist(),[],faces.tolist())
    obj = bpy.data.objects.new(mode,mesh);bpy.context.collection.objects.link(obj)
    material = bpy.data.materials.new(mode+'Material');material.use_nodes=True;mesh.materials.append(material)
    try:
        qualification = qualify_geometry(GeometryArrays.capture(vertices,faces),python=ROOT/'.venv312/Scripts/python.exe',timeout_s=15.)
    except Exception as exc:
        qualification = {'status':'unavailable','reason':str(exc)}
    checks = []
    for target in ('obj','glb'):
        report = run_export_roundtrip_qa([obj],output,targets=(target,),cleanup_imports=False)[0]
        if not report.status_ok or not report.reimport_ok:
            raise ValueError(report.to_dict())
        imported = [o for o in bpy.data.objects if o.type=='MESH' and o!=obj]
        actual = np.array([tuple(o.matrix_world@v.co) for o in imported for v in o.data.vertices])
        error = mesh_response(vertices,actual)
        if error>=1e-5 or sum(len(o.data.polygons) for o in imported)!=len(faces):
            raise ValueError(f'{target}: world geometry roundtrip mismatch {error}')
        checks.append({'target':target,'max_world_vertex_error':error,'report':report.to_dict()})
        for imported_obj in imported:
            bpy.data.objects.remove(imported_obj,do_unlink=True)
    representation = 'editable_multipart_assembly' if selected_backend(row) in {'primitive_fit_refine','gaussian_ellipsoid_proxy','differentiable_refine'} else 'evaluated_mesh_or_csg'
    return {'case':row['case'],'variant':row.get('campaign_variant','final'),'mode':mode,
        'mesh':str(source/'artifacts/validated-mesh.obj'),'exports':checks,
        'representation':row.get('backend_metrics', {}).get('extras', {}).get('output_contract', {}).get('contract', representation),
        'single_solid_qualified':qualification.get('single_solid_qualified',False),
        'selected_backend':selected_backend(row),'boundary_qualification':qualification,
        'qualification_limit':'multipart editability and one connected solid are separate contracts'}


def reload_parts(row, path, output):
    import bpy, numpy as np
    from blender_blocking.primitives.analytic_primitives import EllipsoidPrimitive,SuperquadricPrimitive,AnisotropicGaussianPrimitive
    from blender_blocking.primitives.superfrustum import SuperFrustum
    from blender_blocking.primitives.generalized_sweep import GeneralizedSweepPrimitive
    from blender_blocking.reconstruction.mesh_io import combine_primitive_meshes
    from blender_blocking.evaluation.comparable_geometry import read_obj
    factories = {'generalized_sweep':GeneralizedSweepPrimitive,'ellipsoid':EllipsoidPrimitive,'superquadric':SuperquadricPrimitive,
        'anisotropic_gaussian':AnisotropicGaussianPrimitive,'gaussian':AnisotropicGaussianPrimitive,
        'SuperFrustum':SuperFrustum,'superfrustum':SuperFrustum}
    payload = json.loads(path.read_text())
    if not payload.get('primitives'):
        metadata=payload.get('metadata',{})
        if metadata.get('contract') != 'deformed_multipart_with_editable_initialization':
            return None
        from blender_blocking.reconstruction.differentiable.dvx_artifacts import replay_dvx_artifacts
        replay=replay_dvx_artifacts(payload,())
        return {'group':row['split'],'case':row['case'],'mode':row['requested_mode'],
            'variant':row.get('campaign_variant','final'),'parameter_source':str(path),
            'reload_max_vertex_error':0.,'parts':0,'parameter_vertex_response':None,
            'artist_source':metadata.get('artist_source_path'),
            'output_contract':'conditioned retained seed surface with separate artist source',
            'deformation_artifacts':replay['receipt'],
            'limit':'retained deformation replayed by stored coordinates; artist parameter response was not measured'}
    primitives = [factories[p['type']].from_dict(p) for p in payload['primitives']]
    resolution = 24 if selected_backend(row)=='primitive_fit_refine' else 20
    metadata = payload.get('metadata', {})
    dvx_artifacts = None
    if metadata.get('contract') == 'deformed_multipart_with_editable_initialization':
        from blender_blocking.reconstruction.differentiable.dvx_artifacts import replay_dvx_artifacts
        dvx_artifacts = replay_dvx_artifacts(payload, primitives)
        resolution = metadata['initialization_resolution']
        saved_vertices = dvx_artifacts['seed'].vertices
    else:
        saved_vertices,_ = read_obj(next((path.parent.parent/'m').glob('*.obj')))
    baseline = np.asarray(combine_primitive_meshes(primitives,resolution=resolution).vertices)
    reload_error = mesh_response(baseline,saved_vertices)
    source = Path(row['result_path']).parent
    config = json.loads((source/'config.json').read_text())
    deformed = selected_backend(row)=='differentiable_refine' and config['differentiable_render']['backend']=='dvx'
    if not deformed and reload_error>=1e-6:
        raise ValueError('primitive parameters do not reproduce saved geometry: '+str(reload_error))
    modified = copy.deepcopy(primitives);primitive=modified[0]
    if hasattr(primitive,'radii'):primitive.radii[0]*=1.1
    elif hasattr(primitive,'covariance'):primitive.covariance*=1.21
    else:primitive.height*=1.1
    changed = np.asarray(combine_primitive_meshes(modified,resolution=resolution).vertices)
    response = float(np.max(np.abs(changed-baseline)))
    if response<=1e-6:raise ValueError('primitive edit has no geometry response')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for index,part in enumerate(primitives):
        data=part.to_mesh_data(20);mesh=bpy.data.meshes.new(f'Part{index}')
        mesh.from_pydata(data.vertices.tolist(),[],[tuple(f) for f in data.faces])
        obj=bpy.data.objects.new(f'Part{index}',mesh);obj['primitive_parameters']=json.dumps(part.to_dict())
        bpy.context.collection.objects.link(obj)
    blend=output/(row['split']+'-'+row.get('campaign_variant','final')+'-'+row['case']+'-'+row['requested_mode']+'-parts.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),check_existing=False)
    return {'group':row['split'],'case':row['case'],'mode':row['requested_mode'],'variant':row.get('campaign_variant','final'),
        'parameter_source':str(path),'reload_max_vertex_error':reload_error,'parts':len(primitives),
        'parameter_vertex_response':response,'artist_source':str(blend),
        'output_contract':'deformed multipart mesh with editable initialization sources' if deformed else 'parameter-reproducible multipart assembly',
        'deformation_artifacts': None if dvx_artifacts is None else dvx_artifacts['receipt'],
        'limit':'DVX deformation is not reproduced by seed parameters; semantic usability is unqualified'}


def program_artist_sources(row, output):
    import bpy
    from blender_blocking.primitives.shape_program import ShapeProgram,ShapeNode,ShapeConstraint,ResidualPatch
    from blender_blocking.primitives.shape_program_compiler import compile_shape_program
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays
    payload=row.get('backend_metrics',{}).get('extras',{}).get('shape_program')
    if not payload:return {'status':'unavailable','reason':'saved selected program not present'}
    node=lambda data:ShapeNode(**{**data,'children':tuple(data.get('children',()))})
    program=ShapeProgram(payload['schema_version'],payload['program_id'],tuple(node(x) for x in payload['root_nodes']),
        tuple(ShapeConstraint(**{**x,'target_nodes':tuple(x['target_nodes'])}) for x in payload.get('constraints',())),
        tuple(ResidualPatch(**{**x,'notes':tuple(x.get('notes',())),
            'suggested_node':node(x['suggested_node']) if x.get('suggested_node') else None}) for x in payload.get('residual_patches',())),
        payload.get('metadata',{}))
    config=json.loads((Path(row['result_path']).parent/'config.json').read_text())['shape_program']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    compiled=compile_shape_program(program,lathe_segments=config['lathe_segments'],bevel_modifier=config['bevel_modifier'],
        weighted_normals=config['weighted_normals'],timeout_s=30.,
        csg_options={'native_union_execution':True,'native_union_solver':'EXACT','native_sdf_fallback':False})
    baseline=evaluated_arrays(compiled.root_object).vertices
    response=0.
    for obj in compiled.objects:
        if obj.type!='MESH' or not obj.get('blendslop_shape_node_id'):
            continue
        original=obj.scale.x;obj.scale.x*=1.1;bpy.context.view_layer.update()
        try:response=max(response,mesh_response(baseline,evaluated_arrays(compiled.root_object).vertices))
        finally:obj.scale.x=original;bpy.context.view_layer.update()
        if response>1e-6:break
    blend=output/(row.get('campaign_variant','final')+'-'+row['case']+'-program-artist.blend')
    bpy.ops.wm.save_as_mainfile(filepath=str(blend),check_existing=False)
    return {'case':row['case'],'variant':row.get('campaign_variant','final'),'artist_source':str(blend),
        'status':'measured' if response>1e-6 else 'unavailable','parameter_vertex_response':response,
        'contract':'saved selected program replayed as Exact live artist sources; qualified baked candidate requires recompile after edits',
        'limit':'parameter response is not semantic artist certification'}


def main():
    import bpy
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);args.output.mkdir(parents=True,exist_ok=True)
    coordinated=(args.phase/'campaign-rows.json').exists()
    matrix=json.loads((args.phase/'campaign-rows.json').read_text())['rows'] if coordinated else [
        {**row,'split':group} for group in ('paired','heldout') for row in json.loads((args.phase/group/'final.json').read_text())['rows']]
    receipt={'blender':bpy.app.version_string,'mesh_exports':[],'primitive_editability':[],'program_editability':[],
             'unavailable_candidates':[],'validation_failures':[]}
    def persist():
        (args.output/'result.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for row in matrix:
        if not row.get('result_path') or row.get('candidate_status') == 'failed' or row.get('mesh_sha256', 'unrecorded') is None:
            receipt['unavailable_candidates'].append({key:row.get(key) for key in ('case','requested_mode','campaign_variant','failure')});persist();continue
        label=row.get('campaign_variant','final')+'-'+row['case']+'-'+row['requested_mode']
        try:
            if coordinated or (row['split']=='paired' and row['case']=='box'):
                receipt['mesh_exports'].append(export_row(row,args.output/label))
        except Exception as exc:
            receipt['validation_failures'].append({'cell':label,'check':'export','error':str(exc)})
        if selected_backend(row) in {'primitive_fit_refine','differentiable_refine','gaussian_ellipsoid_proxy'}:
            paths = primitive_parameter_sources(row)
            if not paths:
                receipt['validation_failures'].append({'cell':label,'check':'primitive_reload','error':'selected multipart parameters missing'})
            for path in paths:
                try:
                    result=reload_parts(row,path,args.output)
                    if result:receipt['primitive_editability'].append(result)
                except Exception as exc:
                    receipt['validation_failures'].append({'cell':label,'check':'primitive_reload','source':str(path),'error':str(exc)})
        if row['requested_mode']=='shape_program' or row.get('selected_backend')=='shape_program':
            try:receipt['program_editability'].append(program_artist_sources(row,args.output))
            except Exception as exc:receipt['validation_failures'].append({'cell':label,'check':'program_artist','error':str(exc)})
        persist()
    if not coordinated and (len(receipt['mesh_exports'])!=8 or len(receipt['primitive_editability'])!=18):
        receipt['validation_failures'].append({'check':'legacy_expected_counts','error':'missing expected legacy receipts'});persist()
    print(f"{len(receipt['mesh_exports'])} exports; {len(receipt['primitive_editability'])} primitive and {len(receipt['program_editability'])} program artist receipts")
    if receipt['validation_failures']:raise RuntimeError('export/artist failures preserved; all available cells attempted')


if __name__ == '__main__':main()
