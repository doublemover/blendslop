"""Explicit formulas, shared coordinate frames, and failure-preserving macro means."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from blender_blocking.evaluation.geometry import nearest_distances

SUPERFLEX_REVISION = "15b63b23375c6a80c3ce0dec4be0c9b11055c722"
SUPERFIT_REVISION = "42b263343683221f7c1081baa069b029e9aafeee"
EVALUATOR_VERSION = "blendslop_metric_contracts_v1"


@dataclass(frozen=True)
class Profile:
    name: str
    revision: str
    frame: str
    extent: float | None
    reference_count: int
    prediction_count: int
    thresholds: tuple[float, ...]
    f_epsilon: float = 0.0
    chamfer_factor: float = 0.5
    display_multiplier: float = 100.0
    native: bool = False
    representation: str = "explicit_surface"

    def to_dict(self):
        return asdict(self)


PROFILES = {
    "legacy_bbox_v1": Profile("legacy_bbox_v1", "34de2d3", "independent_bbox", 1.0,
                              8192, 8192, (.02,), chamfer_factor=1., display_multiplier=1.),
    "common_surface_v1": Profile("common_surface_v1", EVALUATOR_VERSION, "target_bbox", 1.,
                                 4096, 4096, (.01, .015, .02), display_multiplier=1.),
    "superflex_formula_common_v1": Profile("superflex_formula_common_v1", SUPERFLEX_REVISION,
                "target_bbox", 1., 4096, 4096, (.01, .015, .02), f_epsilon=1e-6,
                representation="concatenated_component_surfaces"),
    "superflex_native_v1": Profile("superflex_native_v1", SUPERFLEX_REVISION, "provided_shapenet",
                None, 4096, 4096, (.01, .015, .02), f_epsilon=1e-6, native=True,
                representation="concatenated_component_surfaces"),
    "superfit_formula_common_v1": Profile("superfit_formula_common_v1", SUPERFIT_REVISION,
                "target_bbox", 1.8, 2048, 2048, (), representation="explicit_surface"),
    "superfit_native_v1": Profile("superfit_native_v1", SUPERFIT_REVISION,
                "target_bbox", 1.8, 2048, 2048, (), native=True),
}


def points(value):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3 or not len(out) or not np.isfinite(out).all():
        raise ValueError("points must be nonempty finite Nx3 arrays")
    return out


def shared_transform(reference, extent=1.0):
    ref = points(reference)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    longest = float((hi-lo).max())
    if longest <= 1e-12 or not np.isfinite(extent) or extent <= 0:
        raise ValueError("target extent must be positive and reference nondegenerate")
    matrix = np.eye(4)
    matrix[:3, :3] *= float(extent) / longest
    matrix[:3, 3] = -(lo+hi)/2 * float(extent) / longest
    return matrix


def apply_transform(value, matrix):
    value = points(value); m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], (0,0,0,1)):
        raise ValueError("transform must be a finite affine 4x4 matrix")
    if abs(np.linalg.det(m[:3,:3])) <= 1e-15:
        raise ValueError("transform is singular")
    return value @ m[:3,:3].T + m[:3,3]


def metrics_from_distances(prediction_to_reference, reference_to_prediction, profile):
    profile = PROFILES[profile] if isinstance(profile, str) else profile
    p = np.asarray(prediction_to_reference, dtype=float)
    r = np.asarray(reference_to_prediction, dtype=float)
    for distances in (p, r):
        if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("directed distances must be finite nonnegative nonempty vectors")
    l1 = float(profile.chamfer_factor*(p.mean()+r.mean()))
    l2 = float(profile.chamfer_factor*((p*p).mean()+(r*r).mean()))
    result = {"cd_l1": l1, "cd_l2": l2,
              "cd_l1_display": l1*profile.display_multiplier,
              "cd_l2_display": l2*profile.display_multiplier, "f_scores": {}}
    for threshold in profile.thresholds:
        precision = float((p <= threshold).mean()); recall = float((r <= threshold).mean())
        denominator = precision+recall+profile.f_epsilon
        f = float(2*precision*recall/denominator) if denominator > 0 else 0.
        result["f_scores"][str(threshold)] = {"precision":precision,"recall":recall,"f":f}
    if profile.name.startswith("superfit"):
        result["reported_cd"] = result["cd_l2_display"]
    return result


def evaluate_points(reference, prediction, *, profile="common_surface_v1", matrix=None,
                    enforce_native=False, reference_metadata=None):
    contract = PROFILES[profile]
    ref, pred = points(reference), points(prediction)
    metadata = reference_metadata or {}
    if contract.native or enforce_native:
        if len(ref) != contract.reference_count or len(pred) != contract.prediction_count:
            return {"status":"unavailable", "reason":"required native sample counts missing", "profile":profile}
        if profile.startswith("superflex") and not (metadata.get("reference_sampling") == "released_fps" and
                metadata.get("reference_sha256") and metadata.get("frame") == "released_shapenet" and
                metadata.get("prediction_sampling") == "area_weighted_triangles" and matrix is not None):
            return {"status":"unavailable", "reason":"released FPS points, frame and deterministic area samples required", "profile":profile}
        if profile == "superflex_native_v1" and not np.array_equal(np.asarray(matrix), np.eye(4)):
            return {"status":"unavailable", "reason":"native ShapeNet normalize=False requires the identity transform", "profile":profile}
        preparation = metadata.get("upstream_preparation_receipt", {})
        if profile.startswith("superfit") and not (isinstance(preparation, dict) and
                preparation.get("revision") == SUPERFIT_REVISION and preparation.get("target_sha256") and
                preparation.get("cleanup_reproduced") is True and preparation.get("flexicubes_extraction_reproduced") is True):
            return {"status":"unavailable", "reason":"target cleanup/FlexiCubes preparation receipt missing", "profile":profile}
    if contract.frame == "independent_bbox":
        from blender_blocking.evaluation.comparable_geometry import normalize_mesh
        ref, ref_info = normalize_mesh(ref); pred, pred_info = normalize_mesh(pred)
        transform = {"independent_reference":ref_info,"independent_prediction":pred_info}
    else:
        if matrix is None:
            if contract.frame.startswith("provided"):
                return {"status":"unavailable", "reason":"supplied native coordinate transform missing", "profile":profile}
            matrix = shared_transform(ref, contract.extent)
        ref = apply_transform(ref, matrix); pred = apply_transform(pred, matrix)
        transform = {"shared_matrix":np.asarray(matrix).tolist(), "target_extent":contract.extent}
    p = nearest_distances(pred, ref); r = nearest_distances(ref, pred)
    return {"status":"available", "profile":contract.to_dict(), "transform":transform,
            "metrics":metrics_from_distances(p,r,contract), "prediction_to_reference":p,
            "reference_to_prediction":r, "reference_points":ref,"prediction_points":pred,
            "compatibility":"native_inputs" if contract.native else "formula_on_common_cases"}


def macro_cases(cases):
    """Each case contributes once; failures make the complete macro unavailable."""
    ids = [c["case_id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("nonempty cases need unique case IDs")
    good = [c for c in cases if c.get("status") == "available"]
    keys = set.intersection(*(set(c["metrics"]) for c in good)) if good else set()
    numeric = {k:float(np.mean([c["metrics"][k] for c in good])) for k in keys
               if all(isinstance(c["metrics"][k], (int,float)) for c in good)}
    thresholds = set.intersection(*(set(c["metrics"].get("f_scores",{})) for c in good)) if good else set()
    numeric["f_scores"] = {t:{k:float(np.mean([c["metrics"]["f_scores"][t][k] for c in good]))
                                  for k in ("precision","recall","f")} for t in sorted(thresholds)}
    return {"case_count":len(cases),"available_count":len(good),"failures":[c for c in cases if c.get("status") != "available"],
            "macro":numeric if len(good)==len(cases) else None,
            "available_case_macro":numeric,"aggregation":"per_object_equal_weight"}


def union_occupancy(fields):
    arrays = [np.asarray(f, bool) for f in fields]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("active fields require matching query arrays")
    return np.logical_or.reduce(arrays)


def common_grid_sdf_diagnostic(reference_fields, prediction_fields, *, resolution=128):
    """Common cell-center query grid diagnostic, not the upstream SuperFit pipeline."""
    if resolution not in (128,256):
        raise ValueError("bounded diagnostic resolutions are 128 or 256")
    a=np.logical_or.reduce([np.asarray(f)<=0 for f in reference_fields])
    b=np.logical_or.reduce([np.asarray(f)<=0 for f in prediction_fields])
    if a.shape != (resolution,)*3 or b.shape != a.shape:
        raise ValueError("SDF arrays must share the explicit grid")
    from blender_blocking.evaluation.geometry import volumetric_iou
    return {"volumetric_iou":volumetric_iou(a,b), "resolution":resolution,
            "representation":"union_of_active_sdf_sign_fields", "compatibility":"common_grid_diagnostic"}


def released_query_occupancy(reference_labels, active_prediction_fields, *, query_hash, reference_query_hash):
    if not query_hash or query_hash != reference_query_hash:
        return {"status":"unavailable","reason":"released occupancy query hashes missing or mismatched"}
    reference=np.asarray(reference_labels,bool)
    prediction=union_occupancy(active_prediction_fields)
    if reference.shape != prediction.shape:
        raise ValueError("released occupancy labels and active fields must share queries")
    from blender_blocking.evaluation.geometry import volumetric_iou
    return {"status":"available","volumetric_iou":volumetric_iou(reference,prediction),
            "representation":"union_of_active_implicit_fields","query_sha256":query_hash}
