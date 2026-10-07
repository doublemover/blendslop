"""Reuse bounded same-evidence surfaces before falling back to fresh parts."""
from __future__ import annotations


def select_existing_seed(target,results,*,maximum_vertices=4096, maximum_faces=None):
    from ..evidence_identity import target_evidence_hash
    from ..native_geometry import geometry_arrays
    from ..grouped_solids import solid_guard
    identity=target_evidence_hash(target)
    accepted,reasons=[],[]
    for name,result in dict(results or {}).items():
        if not result.succeeded or result.geometry is None:
            reasons.append({'source':name,'reason':'no_successful_retained_geometry'});continue
        extra=result.metric_result.extras
        if extra.get('input_evidence_hash') != identity:
            reasons.append({'source':name,'reason':'same_fixed_camera_evidence_not_established'});continue
        data=geometry_arrays(result.geometry)
        if len(data.vertices)>maximum_vertices:
            reasons.append({'source':name,'reason':'seed_vertex_allowance; simplify actual seed separately'});continue
        if maximum_faces is not None and len(data.faces)>maximum_faces:
            reasons.append({'source':name,'reason':'seed_triangle_allowance; simplify actual seed separately'});continue
        if not solid_guard(data)['valid_solid']:
            reasons.append({'source':name,'reason':'closed_orientation_volume_seed_guard_failed'});continue
        rows=result.metric_result.per_view
        from ..visibility import valid_evidence
        observed={c.view for c in target.constraints if valid_evidence(c).any()}
        if (not rows or not observed.issubset(rows)
                or not all(rows[view].get('passed',False) for view in observed)
                or not all(r.get('passed',False) for r in rows.values() if r.get('required',True))):
            reasons.append({'source':name,'reason':'required_observed_views_not_validated'});continue
        accepted.append((result.metric_result.area_iou_min,result.metric_result.area_iou_mean,
                         -len(data.faces),name,result,data))
    if not accepted:
        return None,{'status':'fallback_required','reasons':reasons,'input_evidence_hash':identity}
    _,_,_,name,result,data=max(accepted,key=lambda row:row[:3])
    return data,{'status':'retained_existing_seed','source':name,'source_candidate_id':result.candidate_id,
        'source_primitive_path':str(result.primitive_path) if result.primitive_path else None,
        'source_mesh_path':str(result.mesh_path) if result.mesh_path else None,
        'seed_content_hash':data.content_hash,'input_evidence_hash':identity,'reasons':reasons,
        'qualification_scope':'same-evidence rendered seed plus topology/orientation/volume screen; boundary certification remains separate'}
