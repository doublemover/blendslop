"""Explicit bounded native DTU evaluation; no datasets are downloaded by this tool.

Use --assets with an NPZ containing reference, obs_mask, observation_bb,
observation_res, ground_plane, scale_mat, or --dataset-dir/--scan with the
standard Points/stl and ObsMask layout and an explicit prediction transform.
Prediction NPZ files use vertices+faces or points; OBJ/PLY are also supported.
Outputs include every prepared point set/directed distance and hash receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]
from blender_blocking.evaluation.protocols import dtu_native_adapter, prepare_dtu_prediction
from blender_blocking.evaluation.protocols.dtu import DTU_CONTRACT, DTU_MODES
from blender_blocking.evaluation.protocols.dtu_preparation import array_sha256


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_prediction(path):
    if path.suffix.lower() == ".npz":
        data = load_npz(path)
        return data["vertices" if "vertices" in data else "points"], data.get("faces")
    if path.suffix.lower() == ".obj":
        from blender_blocking.evaluation.comparable_geometry import read_obj
        return read_obj(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False), None
    if path.suffix.lower() == ".ply":
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles):
            return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
        return np.asarray(o3d.io.read_point_cloud(str(path)).points).copy(), None
    raise ValueError("prediction must be NPZ, NPY, OBJ or PLY")


def load_assets(args):
    if args.assets:
        assets = load_npz(args.assets)
        return assets, [args.assets]
    if args.scan is None or args.scan < 1:
        raise ValueError("--dataset-dir requires a positive --scan")
    from scipy.io import loadmat
    import open3d as o3d
    obs_path = args.dataset_dir / "ObsMask" / f"ObsMask{args.scan}_10.mat"
    plane_path = args.dataset_dir / "ObsMask" / f"Plane{args.scan}.mat"
    reference_path = args.dataset_dir / "Points" / "stl" / f"stl{args.scan:03}_total.ply"
    obs = loadmat(obs_path)
    if args.prediction_native_mm:
        transform = np.eye(4)
    elif args.scale_matrix:
        transform = np.load(args.scale_matrix, allow_pickle=False)
    else:
        raise ValueError("declare --prediction-native-mm or supply --scale-matrix; no frame inference")
    assets = {"reference": np.asarray(o3d.io.read_point_cloud(str(reference_path)).points).copy(),
              "obs_mask": obs["ObsMask"], "observation_bb": obs["BB"], "observation_res": obs["Res"],
              "ground_plane": loadmat(plane_path)["P"], "scale_mat": transform}
    return assets, [obs_path, plane_path, reference_path] + ([args.scale_matrix] if args.scale_matrix else [])


def evaluate_subset(args):
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing evaluation directory")
    assets, source_files = load_assets(args)
    prediction, faces = load_prediction(args.prediction)
    culling = None
    if args.culling or args.culling_receipt:
        if not args.culling or not args.culling_receipt:
            raise ValueError("both --culling and --culling-receipt are required")
        culling = {"keep": load_npz(args.culling)["keep"],
                   "receipt": json.loads(args.culling_receipt.read_text(encoding="utf-8-sig"))}
        source_files += [args.culling, args.culling_receipt]
    prepared = prepare_dtu_prediction(mode=args.mode, prediction=prediction, faces=faces, assets=assets,
                                      seed=args.seed, max_points=args.max_points, culling=culling)
    result = {key: value for key, value in prepared.items() if key not in ("prediction", "directional_sets")}
    if prepared["status"] == "available":
        result = dtu_native_adapter(mode=args.mode, assets=assets, prediction=prediction,
                                   directional_sets=prepared["directional_sets"],
                                   sampling_receipt=prepared["sampling_receipt"],
                                   transform_direction="prediction_to_native_mm")
    # Reserve a new directory, even for unavailable cases; never overwrite an
    # earlier accepted receipt or drop a failed cell from a larger manifest.
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    if prepared["status"] == "available":
        arrays.update(prepared["directional_sets"])
    for key in ("prediction_to_reference", "reference_to_prediction"):
        if key in result:
            arrays[key] = result.pop(key)
    if arrays:
        np.savez_compressed(args.output / "samples.npz", **arrays)
        result["arrays"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "sha256": array_sha256(value)} for key, value in arrays.items()}
        result["samples_file_sha256"] = hashlib.sha256((args.output / "samples.npz").read_bytes()).hexdigest()
    result.update(schema="blendslop_dtu_subset_receipt_v1", case_id=args.case_id, mode=args.mode,
                  seed=args.seed, contract=DTU_CONTRACT,
                  source_files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [args.prediction] + source_files},
                  evaluator_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                             text=True).strip())
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assets", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--scan", type=int)
    frame = parser.add_mutually_exclusive_group()
    frame.add_argument("--scale-matrix", type=Path)
    frame.add_argument("--prediction-native-mm", action="store_true")
    parser.add_argument("--mode", choices=tuple(DTU_MODES), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--culling", type=Path)
    parser.add_argument("--culling-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.assets and (args.scale_matrix or args.prediction_native_mm):
        parser.error("NPZ assets already declare scale_mat; do not override their frame")
    result = evaluate_subset(args)
    print(f"{args.case_id}: {result['status']}")
    return 0 if result["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
