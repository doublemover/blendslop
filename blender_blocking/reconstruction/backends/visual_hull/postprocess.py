from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from blender_blocking.utils.optional_deps import dependency_report, probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import dependency_report, probe_dependency
from .dependency_policy import _optional_dependency_status
from .poisson import _run_open3d_poisson


def _evaluate_postprocess(
    mesh_result: Any,
    postprocess: str,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _mesh, status = _postprocess_mesh(mesh_result, postprocess, config=config)
    return status

def _postprocess_mesh(
    mesh_result: Any,
    postprocess: str,
    *,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    method = str(postprocess).strip().lower()
    required = _postprocess_required(config)
    if method in {"", "none"}:
        return mesh_result, {
            "method": "none",
            "status": "skipped",
            "required": required,
            "message": "mesh postprocess disabled",
        }
    if method == "topology_repair":
        return _run_safe_topology_repair(mesh_result, config=config)
    if method == "smooth_guarded":
        return _run_guarded_smooth(mesh_result, config=config)
    if method not in {"poisson", "screened_poisson"}:
        return mesh_result, {
            "method": method,
            "status": "failed",
            "required": required,
            "message": f"unsupported mesh postprocess mode: {method!r}",
        }
    if mesh_result is None:
        return mesh_result, {
            "method": method,
            "status": "skipped",
            "required": required,
            "message": "mesh extraction has not completed",
        }
    if not mesh_result.available:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "mesh_status": mesh_result.status,
            "message": (
                f"postprocess {method!r} requires a mesh, but mesh extraction "
                f"status was {mesh_result.status!r}"
            ),
        }

    if config.get("external_open3d_python"):
        from .poisson_bridge import run_external_poisson
        try:
            processed = run_external_poisson(mesh_result, method, config)
            return processed, {"method":method,"status":"ok","required":required,
                "implementation":"explicit_open3d_cpu_helper","metrics":processed.metrics,
                "message":"configured external CPU helper completed; Blender ABI kept isolated"}
        except Exception as exc:
            status = {"method":method,"status":"failed" if required else "skipped","required":required,
                "message":str(exc),"error_type":type(exc).__name__}
            for key in ("poisson_ownership_receipt", "poisson_process_receipt"):
                receipt = getattr(exc, key, None)
                if receipt is not None:
                    status[key] = str(receipt)
            return mesh_result, status
    dependency = _optional_dependency_status("open3d")
    if not dependency["available"]:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "dependency": dependency,
            "message": (
                f"postprocess {method!r} requires optional dependency "
                f"{dependency['module_name']!r}"
            ),
        }
    try:
        processed = _run_open3d_poisson(mesh_result, method, config)
    except Exception as exc:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "dependency": dependency,
            "message": f"postprocess {method!r} failed: {exc}",
            "error_type": type(exc).__name__,
        }
    return processed, {
        "method": method,
        "status": "ok",
        "required": required,
        "dependency": dependency,
        "message": f"postprocess {method!r} completed with Open3D Poisson",
        "input_vertices": int(len(mesh_result.vertices)),
        "input_faces": int(len(mesh_result.faces)),
        "output_vertices": int(len(processed.vertices)),
        "output_faces": int(len(processed.faces)),
        "implementation": "open3d.geometry.TriangleMesh.create_from_point_cloud_poisson",
    }

def _run_guarded_smooth(
    mesh_result: Any,
    *,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    required = _postprocess_required(config)
    method = "smooth_guarded"
    if mesh_result is None:
        return mesh_result, {
            "method": method,
            "status": "skipped",
            "required": required,
            "message": "mesh extraction has not completed",
        }
    if not getattr(mesh_result, "available", False):
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "mesh_status": getattr(mesh_result, "status", "unknown"),
            "message": (
                "smooth_guarded requires a mesh, but mesh extraction status was "
                f"{getattr(mesh_result, 'status', 'unknown')!r}"
            ),
        }

    from metrics.topology import mesh_topology_report
    from metrics.topology_guard import evaluate_mesh_change, guard_policy_from_config
    from volume import MeshExtractionResult

    vertices = np.asarray(mesh_result.vertices, dtype=float)
    faces = np.asarray(mesh_result.faces, dtype=np.int64)
    before = mesh_topology_report(vertices, faces)
    smoothed, smooth_meta = _guarded_laplacian_smooth(
        vertices,
        faces,
        iterations=int(config.get("smooth_iterations", 2)),
        alpha=float(config.get("smooth_alpha", 0.25)),
        max_displacement_ratio=float(config.get("smooth_max_displacement_ratio", 0.02)),
        freeze_boundary=bool(config.get("smooth_freeze_boundary", True)),
    )
    after = mesh_topology_report(smoothed, faces)
    guard = evaluate_mesh_change(
        before=before,
        after=after,
        before_vertex_count=int(len(vertices)),
        after_vertex_count=int(len(smoothed)),
        before_face_count=int(len(faces)),
        after_face_count=int(len(faces)),
        policy=guard_policy_from_config(config),
    )
    if not guard.accepted:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "message": f"guarded smoothing rejected: {guard.reason}",
            "before": before.to_dict(),
            "after": after.to_dict(),
            "smooth": smooth_meta,
            "guard": guard.to_dict(),
        }
    metrics = {
        **dict(getattr(mesh_result, "metrics", {})),
        "postprocess": method,
        "postprocess_backend": "pure_python",
        **smooth_meta,
    }
    repaired = MeshExtractionResult(
        status="ok",
        method=f"{getattr(mesh_result, 'method', 'mesh')}_smooth_guarded",
        requested_method=getattr(mesh_result, "requested_method", mesh_result.method),
        method_aliases=tuple(getattr(mesh_result, "method_aliases", ()) or ()),
        vertices=smoothed,
        faces=faces,
        normals=None,
        values=getattr(mesh_result, "values", None),
        message="guarded Laplacian smoothing completed",
        metrics=metrics,
        topology=after.to_dict(),
    )
    return repaired, {
        "method": method,
        "status": "ok",
        "required": required,
        "message": "guarded Laplacian smoothing completed",
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "output_vertices": int(len(repaired.vertices)),
        "output_faces": int(len(repaired.faces)),
        "implementation": "visual_hull._guarded_laplacian_smooth",
        "before": before.to_dict(),
        "after": after.to_dict(),
        "smooth": smooth_meta,
        "guard": guard.to_dict(),
    }

def _guarded_laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    alpha: float,
    max_displacement_ratio: float,
    freeze_boundary: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return np.empty((0, 3), dtype=float), {
            "smooth_iterations": 0,
            "smooth_alpha": float(alpha),
            "smooth_moved_vertices": 0,
            "smooth_max_displacement": 0.0,
        }
    adjacency, boundary_vertices = _mesh_adjacency_and_boundary(faces, len(vertices))
    output = vertices.copy()
    iterations = max(0, int(iterations))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    max_displacement = max(0.0, bbox_diag * max(0.0, float(max_displacement_ratio)))
    movable = {
        index
        for index, neighbors in adjacency.items()
        if neighbors and (not freeze_boundary or index not in boundary_vertices)
    }
    for _ in range(iterations):
        updated = output.copy()
        for index in movable:
            neighbors = tuple(adjacency[index])
            if not neighbors:
                continue
            target = output[list(neighbors)].mean(axis=0)
            delta = (target - output[index]) * alpha
            length = float(np.linalg.norm(delta))
            if max_displacement > 0.0 and length > max_displacement:
                delta *= max_displacement / max(length, 1e-12)
            updated[index] = output[index] + delta
        output = updated
    displacement = np.linalg.norm(output - vertices, axis=1)
    return output, {
        "smooth_iterations": iterations,
        "smooth_alpha": alpha,
        "smooth_freeze_boundary": freeze_boundary,
        "smooth_boundary_vertex_count": len(boundary_vertices),
        "smooth_movable_vertex_count": len(movable),
        "smooth_moved_vertices": int(np.count_nonzero(displacement > 1e-12)),
        "smooth_mean_displacement": float(np.mean(displacement)) if displacement.size else 0.0,
        "smooth_max_displacement": float(np.max(displacement)) if displacement.size else 0.0,
        "smooth_max_displacement_ratio": float(max_displacement_ratio),
    }

def _mesh_adjacency_and_boundary(
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[dict[int, set[int]], set[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(vertex_count)}
    edge_counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        if len(face) < 3:
            continue
        clean = [int(vertex) for vertex in face if 0 <= int(vertex) < vertex_count]
        if len(set(clean)) < 3:
            continue
        for index, start in enumerate(clean):
            end = clean[(index + 1) % len(clean)]
            if start == end:
                continue
            adjacency[start].add(end)
            adjacency[end].add(start)
            edge = (start, end) if start < end else (end, start)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = {
        vertex
        for edge, count in edge_counts.items()
        if count == 1
        for vertex in edge
    }
    return adjacency, boundary

def _run_safe_topology_repair(
    mesh_result: Any,
    *,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    required = _postprocess_required(config)
    method = "topology_repair"
    if mesh_result is None:
        return mesh_result, {
            "method": method,
            "status": "skipped",
            "required": required,
            "message": "mesh extraction has not completed",
        }
    if not getattr(mesh_result, "available", False):
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "mesh_status": getattr(mesh_result, "status", "unknown"),
            "message": (
                "topology_repair requires a mesh, but mesh extraction status was "
                f"{getattr(mesh_result, 'status', 'unknown')!r}"
            ),
        }

    from metrics.topology import safe_topology_repair
    from metrics.topology_guard import evaluate_mesh_change, guard_policy_from_config
    from volume import MeshExtractionResult

    repair = safe_topology_repair(
        mesh_result.vertices,
        mesh_result.faces,
        keep_largest_component=bool(config.get("repair_keep_largest_component", True)),
    )
    before_score = float(repair.before.topology_score)
    after_score = float(repair.after.topology_score)
    guard = evaluate_mesh_change(
        before=repair.before,
        after=repair.after,
        before_vertex_count=int(len(mesh_result.vertices)),
        after_vertex_count=int(len(repair.vertices)),
        before_face_count=int(len(mesh_result.faces)),
        after_face_count=int(len(repair.faces)),
        policy=guard_policy_from_config(config),
    )
    if not guard.accepted:
        status = "failed" if required else "skipped"
        return mesh_result, {
            "method": method,
            "status": status,
            "required": required,
            "message": f"safe topology repair rejected: {guard.reason}",
            "repair": repair.to_dict(),
            "guard": guard.to_dict(),
        }

    metrics = {
        **dict(getattr(mesh_result, "metrics", {})),
        "postprocess": method,
        "postprocess_backend": "pure_python",
        "repair_changed": repair.changed,
        "repair_improved": repair.improved,
        "repair_before_topology_score": before_score,
        "repair_after_topology_score": after_score,
        "repair_guard": guard.to_dict(),
    }
    repair_faces = np.asarray(repair.faces, dtype=np.int64)
    if repair_faces.size == 0:
        repair_faces = np.empty((0, 3), dtype=np.int64)
    repaired = MeshExtractionResult(
        status="ok",
        method=f"{getattr(mesh_result, 'method', 'mesh')}_topology_repair",
        requested_method=getattr(mesh_result, "requested_method", mesh_result.method),
        method_aliases=tuple(getattr(mesh_result, "method_aliases", ()) or ()),
        vertices=repair.vertices,
        faces=repair_faces,
        normals=None,
        values=getattr(mesh_result, "values", None),
        message="safe topology repair completed",
        metrics=metrics,
        topology=repair.after.to_dict(),
    )
    return repaired, {
        "method": method,
        "status": "ok",
        "required": required,
        "message": "safe topology repair completed",
        "input_vertices": int(len(mesh_result.vertices)),
        "input_faces": int(len(mesh_result.faces)),
        "output_vertices": int(len(repaired.vertices)),
        "output_faces": int(len(repaired.faces)),
        "implementation": "metrics.topology.safe_topology_repair",
        "repair": repair.to_dict(),
        "guard": guard.to_dict(),
    }

def _postprocess_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("postprocess_required")
        or config.get("require_postprocess")
        or config.get("fail_on_postprocess_skip")
    )

def _mesh_required(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("require_mesh")
        or config.get("mesh_required")
        or config.get("fail_on_mesh_skip")
    )
