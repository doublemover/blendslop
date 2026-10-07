"""Structural solid diagnostics; watertight edges alone do not certify volume."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def solid_validity_report(vertices, faces, *, native_intersections=False,
                          max_intersection_faces=60000, max_contact_pairs=1000):
    from metrics.topology import mesh_topology_report

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("solid vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("solid faces must contain triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("solid face indices are outside the vertex array")

    if not np.isfinite(vertices).all():
        raise ValueError("solid vertices must be finite")

    report = {
        "topology": mesh_topology_report(vertices, faces).to_dict(),
        "finite": bool(np.isfinite(vertices).all()),
    }
    _, remap = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
    geometric_faces = np.sort(remap[faces], axis=1)
    report["duplicate_geometric_faces"] = int(
        len(faces) - len(np.unique(geometric_faces, axis=0)))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    report["zero_area_triangles"] = int(
        np.count_nonzero(np.linalg.norm(normals, axis=1) < 1e-12))
    directions = defaultdict(list)
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            directions[tuple(sorted((int(a), int(b))))].append(1 if a < b else -1)
    report["inconsistent_winding_edges"] = sum(
        len(values) == 2 and values[0] == values[1] for values in directions.values())
    report["self_intersection"] = {
        "status": "not_checked",
        "reason": "requires optional native triangle intersection check",
    }
    report["contained_component_surfaces"] = {
        "status": "not_checked",
        "reason": "requires closed component containment queries",
    }
    if native_intersections:
        import open3d as o3d

        if len(faces) > max_intersection_faces:
            report["self_intersection"] = {
                "status": "not_checked", "reason": "explicit triangle budget exceeded",
                "limit": max_intersection_faces,
            }
        else:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
            pairs = np.asarray(mesh.get_self_intersecting_triangles())
            report["self_intersection"] = {
                "status": "measured", "intersecting_triangle_pairs": len(pairs),
                "sample": pairs[:20].tolist(), "backend": "Open3D " + o3d.__version__,
            }
            from .triangle_contacts import verify_reported_pairs
            exact = verify_reported_pairs(vertices, faces, pairs, max_pairs=max_contact_pairs)
            report['self_intersection']['native_reported_triangle_pairs'] = len(pairs)
            report['self_intersection']['intersecting_triangle_pairs_scope'] = 'legacy raw native candidate count'
            report['self_intersection']['exact_pair_verification'] = exact
            report['self_intersection']['confirmed_boundary_contacts'] = exact['non_disjoint_pairs'] if exact['complete'] else None
            if exact['complete'] and exact['non_disjoint_pairs'] == 0:
                from .triangle_contacts import within_part_boundary_guard
                report['self_intersection']['within_part_boundary_guard'] = within_part_boundary_guard(vertices, faces)
            from .intersection_diagnostics import intersection_pair_diagnostics
            report["self_intersection"]["contact_diagnostics"] = intersection_pair_diagnostics(
                vertices, faces, pairs, max_pairs=max_contact_pairs)
            labels, counts, _ = mesh.cluster_connected_triangles()
            labels = np.asarray(labels)
            if len(counts) <= 32:
                records = []
                components = []
                for index in range(len(counts)):
                    component_faces = faces[labels == index]
                    points = vertices[component_faces].mean(axis=1)
                    sample_indices = np.linspace(0, len(points) - 1,
                                                 min(256, len(points))).astype(int)
                    component_vertices = vertices[component_faces].reshape((-1, 3))
                    components.append((component_faces, points[sample_indices], len(points),
                                       component_vertices.min(axis=0),
                                       component_vertices.max(axis=0)))
                for index, (_, sample, point_count, lower, upper) in enumerate(components):
                    for other, (other_faces, _, _, other_lower, other_upper) in enumerate(components):
                        if index == other or np.any(upper < other_lower) or np.any(other_upper < lower):
                            continue
                        component = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(vertices),
                            o3d.utility.Vector3iVector(other_faces))
                        component.remove_unreferenced_vertices()
                        if not component.is_watertight():
                            continue
                        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
                        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(component))
                        inside = scene.compute_occupancy(
                            o3d.core.Tensor(sample.astype(np.float32)), nthreads=4).numpy() > 0.5
                        if inside.any():
                            records.append({
                                "component": index, "inside_component": other,
                                "inside_centroids": int(inside.sum()), "sample_count": len(sample),
                                "all_sampled_inside": bool(inside.all()),
                                "complete_centroid_coverage": len(sample) == point_count,
                            })
                report["contained_component_surfaces"] = {
                    "status": "sampled_centroids", "records": records, "limit_per_component": 256,
                    "confidence_limit": "centroids reveal internal surfaces, not exact Boolean volume certification",
                }
            else:
                report["contained_component_surfaces"] = {
                    "status": "not_checked", "reason": "explicit component budget exceeded", "limit": 32,
                }
    report["single_solid_eligible"] = bool(
        report["finite"] and report["topology"]["watertight"]
        and report["topology"]["connected_components"] == 1
        and not report["duplicate_geometric_faces"] and not report["zero_area_triangles"]
        and not report["inconsistent_winding_edges"]
        and report["self_intersection"].get("status") == "measured"
        and report["self_intersection"].get("confirmed_boundary_contacts") == 0
        and report["self_intersection"].get("exact_pair_verification", {}).get("complete") is True
        and report["self_intersection"].get("within_part_boundary_guard", {}).get("passed") is True)
    report["representation_limit"] = (
        "disconnected editable assemblies may be useful; concatenation is not a Boolean union")
    return report
