"""Small actual render/export contracts; no matched quality or timing campaign."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking'),str(ROOT/'scripts')]


def main():
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    import bpy,numpy as np
    from PIL import Image
    from blender_blocking.config import BlockingConfig
    from blender_blocking.e2e.validator import test_with_custom_images
    from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views
    from blender_blocking.primitives.shape_program import ShapeProgram,ShapeNode
    from blender_blocking.primitives.analytic_primitives import EllipsoidPrimitive
    from blender_blocking.reconstruction.mesh_io import write_obj,combine_primitive_meshes
    from verify_candidate_exports import export_row,reload_parts,program_artist_sources
    output=Path(sys.argv[sys.argv.index('--')+1]).resolve();output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=1.)
    cfg=BlockingConfig();cfg.reconstruction.reconstruction_mode='visual_hull_voxel'
    cfg.visual_hull.resolution=16
    cfg.render_silhouette.resolution=(64,64)
    cfg.render_silhouette.view_calibration={view:{'projection':'orthographic','world_bounds':[-1,1,-1,1],
        'axes':axes,'world_units':'metres','orientation':'canonical_positive_axes','source':'component_fixture'}
        for view,axes in [('front',[0,2]),('side',[1,2]),('top',[0,1])]}
    references=render_orthogonal_views(str(output/'references'),target_objects=[bpy.context.object],render_config=cfg.render_silhouette)
    cfg.reconstruction.view_calibration=dict(cfg.render_silhouette.view_calibration)
    cfg.reconstruction.view_crops={'front':[8,0,56,64]}
    for view in ('front','side','top'):
        valid=np.ones((64,64),np.uint8)*255
        if view=='front':valid[:,48:]=0
        if view=='top':valid[:]=0
        path=output/(view+'-valid.png');Image.fromarray(valid).save(path)
        cfg.reconstruction.valid_evidence_files[view]=str(path)
    passed=test_with_custom_images(references['front'],references['side'],references['top'],workflow_config=cfg,
        render_config=cfg.render_silhouette,validation_mode='render-iou',iou_threshold=.1,
        render_output_dir=output/'renders',artifact_root=output/'artifacts',result_json=output/'partial-result.json',progress=False)
    partial=json.loads((output/'partial-result.json').read_text())
    assert passed,partial.get('views')
    assert partial['views']['top']['area_iou'] is None and not partial['views']['top']['required']
    assert partial['backend_result']['metric_result']['per_view']['top']['area_iou'] is None,partial['backend_result']
    with Image.open(partial['rendered_paths']['front']) as image:assert image.size==(48,64),image.size
    fixture=output/'export-fixture';fixture.mkdir()
    (fixture/'config.json').write_text(json.dumps(cfg.to_dict()))
    primitive=EllipsoidPrimitive(radii=(.4,.5,.6))
    geometry=combine_primitive_meshes([primitive],resolution=24)
    write_obj(fixture/'artifacts/validated-mesh.obj',geometry)
    params=fixture/'artifacts/cand/fixture/p/parts.json';params.parent.mkdir(parents=True)
    params.write_text(json.dumps({'primitives':[primitive.to_dict()]}))
    write_obj(params.parent.parent/'m/parts.obj',geometry)
    (fixture/'result.json').write_text('{}')
    row={'case':'component','split':'component','campaign_variant':'quality','requested_mode':'primitive_fit_refine',
        'result_path':str(fixture/'result.json')}
    exports=export_row(row,output/'exports')
    parts=reload_parts(row,params,output)
    program=ShapeProgram('shape-program-v1','component-live',(
        ShapeNode('a','add','box',{'width_world':1.,'depth_world':1.,'height_world':1.}),
        ShapeNode('b','add','box',{'width_world':1.,'depth_world':1.,'height_world':1.,'x':.5})))
    program_row={**row,'requested_mode':'shape_program','backend_metrics':{'extras':{'shape_program':program.to_dict()}}}
    artist=program_artist_sources(program_row,output)
    assert artist['status']=='measured',artist
    receipt={'blender':bpy.app.version_string,'partial_observation':{'front_render_resolution':[48,64],
        'top_unknown_excluded':True,'views':partial['views']},'exports':exports,'primitive_source':parts,'program_source':artist,
        'scope':'small component fixtures only; no matched matrix or speed evidence'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('CAMPAIGN_OUTPUT_CONTRACTS_PASSED',flush=True)


if __name__=='__main__':main()
