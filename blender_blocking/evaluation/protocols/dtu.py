"""DTU adapters fail closed without native millimetre assets and official sampling."""
from __future__ import annotations
import numpy as np
from .core import apply_transform, points
from blender_blocking.evaluation.geometry import nearest_distances

DTU_MODES={"dp_gs":{"silhouette_culling":False},"sparsesurf_point":{"silhouette_culling":True},
           "partgs_point":{"silhouette_culling":True},"partgs_block":{"silhouette_culling":False}}
DTU_CONTRACT={"version":"dtu_native_mm_v1","units":"millimetres","thinning_mm":.2,
              "exclude_distance_greater_or_equal_mm":20.,"output_multiplier":1.,
              "aggregation":"half_sum_of_asymmetric_directional_means",
              "required_assets":["reference","obs_mask","observation_bb","observation_res","ground_plane","scale_mat"],
              "sampling":"official_triangle_sampling_then_0.2mm_thinning"}


def dtu_directed_metrics(prediction_queries,reference_targets,reference_queries,prediction_targets):
    p=nearest_distances(points(prediction_queries),points(reference_targets))
    r=nearest_distances(points(reference_queries),points(prediction_targets))
    valid_p=p[p<20.];valid_r=r[r<20.]
    if not len(valid_p) or not len(valid_r):
        return {"status":"unavailable","reason":"empty direction after strict 20mm filter",
                "prediction_to_reference":p,"reference_to_prediction":r}
    pm=float(valid_p.mean());rm=float(valid_r.mean())
    return {"status":"available","accuracy_mm":pm,"completeness_mm":rm,"overall_mm":(pm+rm)/2,
            "prediction_to_reference":p,"reference_to_prediction":r,
            "kept_prediction_count":len(valid_p),"kept_reference_count":len(valid_r),"contract":DTU_CONTRACT}


def dtu_native_adapter(*,mode,assets=None,prediction=None,directional_sets=None,
                       sampling_receipt=None,transform_direction=None):
    if mode not in DTU_MODES:raise ValueError("unknown DTU wrapper mode")
    assets=assets or {}
    missing=[k for k in DTU_CONTRACT['required_assets'] if k not in assets]
    if missing or prediction is None or directional_sets is None or not sampling_receipt or transform_direction!='prediction_to_native_mm':
        return {"status":"unavailable","mode":mode,"contract":DTU_CONTRACT,
                "reason":"native assets, explicit transform direction and official sampling/masking receipt required",
                "missing":missing}
    from .dtu_preparation import (DTU_PREPARATION_VERSION, DTU_REFERENCE_REVISION,
                                  DTU_REFERENCE_SHA256, implementation_sha256)
    reference_port = sampling_receipt.get("preparation_version") == DTU_PREPARATION_VERSION
    if reference_port and (
            sampling_receipt.get("implementation_sha256") != implementation_sha256() or
            sampling_receipt.get("reference_revision") != DTU_REFERENCE_REVISION or
            sampling_receipt.get("source_sha256") != DTU_REFERENCE_SHA256):
        return {"status": "unavailable", "reason": "reference preparation implementation changed", "mode": mode}
    if not reference_port and (sampling_receipt.get('verified_official_evaluator') is not True or not sampling_receipt.get('source_sha256')):
        return {"status":"unavailable","reason":"official sampling/masking implementation not verified","mode":mode}
    if sampling_receipt.get('silhouette_culling') is not DTU_MODES[mode]['silhouette_culling']:
        return {"status":"unavailable","reason":"wrapper culling receipt does not match mode","mode":mode}
    import hashlib
    array_hash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    asset_hashes = sampling_receipt.get('asset_sha256', {})
    for key in DTU_CONTRACT['required_assets']:
        if key not in asset_hashes or array_hash(assets[key]) != asset_hashes[key]:
            return {"status":"unavailable","reason":"native asset hash missing or mismatched: "+key,"mode":mode}
    required_sets = {"prediction_queries", "reference_targets", "reference_queries", "prediction_targets"}
    if set(directional_sets) != required_sets:
        return {"status": "unavailable", "reason": "incomplete asymmetric directional sets", "mode": mode}
    directional_hashes = sampling_receipt.get('directional_sha256', {})
    for key, value in directional_sets.items():
        if directional_hashes.get(key) != array_hash(value):
            return {"status":"unavailable","reason":"prepared directional set hash mismatch","mode":mode}
    # Transform verification is separate from the already prepared asymmetric sets.
    native_prediction=apply_transform(prediction,assets['scale_mat'])
    if reference_port and (sampling_receipt.get("prediction_sha256") != array_hash(points(prediction)) or
                           sampling_receipt.get("native_prediction_sha256") != array_hash(native_prediction)):
        return {"status": "unavailable", "reason": "prepared prediction hash mismatch", "mode": mode}
    result=dtu_directed_metrics(**directional_sets)
    result.update(mode=mode,calibration={"matrix":np.asarray(assets['scale_mat']).tolist(),
                  "direction":transform_direction,"prediction_extent_mm":np.ptp(native_prediction,axis=0).tolist()},
                  sampling_receipt=sampling_receipt,
                  compatibility="pinned_dtu_python_reference_port" if reference_port else "verified_prepared_native_sets")
    return result
