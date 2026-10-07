"""Actual Blender solids/editability fixtures; all outputs stay under --output."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]

def main():
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    import bpy
    import numpy as np
    from blender_blocking.evaluation.comparable_geometry import export_evaluated_object,read_obj
    from blender_blocking.evaluation.solid_validity import solid_validity_report
    from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views_detailed
    from blender_blocking.integration.blender_ops.export_qa import run_export_roundtrip_qa
    from placement.primitive_placement import MeshJoiner
    from blender_blocking.config import RenderConfig
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name, centers, sizes, mode in [
        ('overlap_concat',[(0,0,0),(.5,0,0)],[1.,1.],'simple'),
        ('overlap_boolean',[(0,0,0),(.5,0,0)],[1.,1.],'boolean'),
        ('disconnected',[(0,0,0),(2.,0,0)],[1.,1.],'simple'),
        ('nested_internal',[(0,0,0),(0,0,0)],[2.,.5],'simple'),
        ('duplicate_surfaces',[(0,0,0),(0,0,0)],[1.,1.],'simple')]:
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        parts=[]
        for i,(center,size) in enumerate(zip(centers,sizes)):
            bpy.ops.mesh.primitive_cube_add(size=size,location=center)
            obj=bpy.context.object;obj.name=f'{name}_Part{i}';obj['shape_family']='box';obj['width_world']=size;parts.append(obj)
        joined=MeshJoiner.join_result(parts,target_name=name,mode=mode,solver='EXACT',allow_degraded_simple_join=False)
        if joined.object is None:raise RuntimeError(f'{name} join failed: {joined.to_dict()}')
        root=out/name;root.mkdir(parents=True,exist_ok=True)
        path=export_evaluated_object(joined.object,root/'geometry.obj')
        bpy.ops.wm.save_as_mainfile(filepath=str(root/'source.blend'),check_existing=False)
        cfg=RenderConfig(resolution=(256,256),force_material=True)
        views=render_orthogonal_views_detailed(str(root/'views'),target_objects=[joined.object],render_config=cfg)
        rows.append({'case':name,'join':joined.to_dict(),'structural':solid_validity_report(*read_obj(path)),
            'mesh':str(path),'source_blend':str(root/'source.blend'),'views':views.to_dict()})
    # Separate non-destructive artist source: meaningful native parts with an editable Boolean stack.
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=1.,location=(.2,-.3,.4));body=bpy.context.object;body.name='Body'
    body['shape_family']='box';body['width_world']=1.
    bpy.ops.mesh.primitive_cube_add(size=1.,location=(.7,-.3,.4));part=bpy.context.object;part.name='EditableExtension'
    modifier=body.modifiers.new('EditableUnion','BOOLEAN');modifier.operation='UNION';modifier.object=part;modifier.solver='EXACT'
    part.hide_render=True
    material=bpy.data.materials.new('UsabilityMaterial');material.diffuse_color=(.1,.5,.8,1.);material.use_nodes=True;body.data.materials.append(material)
    bpy.context.view_layer.update()
    root=out/'editable_native';root.mkdir(parents=True,exist_ok=True)
    original=export_evaluated_object(body,root/'original.obj')
    original_matrix=[list(row) for row in body.matrix_world]
    original_verts,_=read_obj(original)
    part.location.x+=.2;bpy.context.view_layer.update()
    changed=export_evaluated_object(body,root/'parameter_changed.obj')
    changed_verts,_=read_obj(changed)
    parameter_response=abs(changed_verts[:,0].max()-original_verts[:,0].max())
    part.location.x-=.2;bpy.context.view_layer.update()
    reports=run_export_roundtrip_qa([body],root/'exports',targets=('obj','glb'),cleanup_imports=True)
    restored=original_matrix==[list(row) for row in body.matrix_world] and len(body.modifiers)==1 and modifier.object==part
    bpy.ops.wm.save_as_mainfile(filepath=str(root/'artist-source.blend'),check_existing=False)
    render_orthogonal_views_detailed(str(root/'views'),target_objects=[body],render_config=RenderConfig(resolution=(256,256),force_material=True))
    rows.append({'case':'editable_native','parts':[body.name,part.name],'modifiers':[{'name':m.name,'type':m.type} for m in body.modifiers],
        'parameter_response_world':float(parameter_response),'source_stack_and_transform_preserved':restored,
        'export_reports':[r.to_dict() for r in reports],'artist_source':str(root/'artist-source.blend'),
        'human_judgment':'pleasant to edit, semantic part quality and topology taste require an artist; not inferred from numeric editability scores'})
    (out/'result.json').write_text(json.dumps({'blender':bpy.app.version_string,'python':sys.version,'rows':rows},indent=2))
    assert parameter_response>.19 and restored
    assert all(r.status_ok and r.reimport_ok for r in reports)
    expected_extent = original_verts.max(axis=0) - original_verts.min(axis=0)
    for report in reports:
        np.testing.assert_allclose(report.bounds, expected_extent, rtol=1e-5, atol=1e-5)
        assert report.material_count and report.face_count
    print('Structural fixtures and editable stack/export checks complete')
if __name__=='__main__':main()
