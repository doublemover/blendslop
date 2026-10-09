"""Source-conditioned engineering limits, separate from artist acceptance.

The declaration consumes authored references and actual cameras only. Continuous
facet bounds are analytic certificates, not finer-mesh measurements. Existing
family gates remain authoritative; this supplementary policy cannot accept a
family whose artist, topology, silhouette or editability verdict is missing.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from blender_blocking.evaluation.canonical_artifacts import (
    CANONICAL_VIEWS, camera_frame_sha256,
)

PROTOCOL = "source_conditioned_family_surface_v1"
SUPPORTED_FAMILIES = ("sphere", "anisotropic_ellipsoid", "cylinder",
                      "tapered_frustum", "capsule", "thin_plate", "concave_arch",
                      "asymmetric_multipart_solid", "torus")
_MAX_ELEMENTS = 65536


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a positive finite number")
    return value


def _closed_oriented_sphere(data):
    """Screen source combinatorics, not candidate solid-boundary qualification."""
    vertices, faces = data.vertices, data.faces
    if len(vertices) > _MAX_ELEMENTS or len(faces) > _MAX_ELEMENTS:
        raise ValueError("source certificate element budget exceeded")
    edges, links = {}, [dict() for _ in vertices]
    for face in faces:
        a, b, c = map(int, face)
        if len({a, b, c}) != 3:
            raise ValueError("source has a degenerate index triangle")
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(1 if u < v else -1)
        for center, u, v in ((a, b, c), (b, c, a), (c, a, b)):
            links[center].setdefault(u, []).append(v)
            links[center].setdefault(v, []).append(u)
    if any(sorted(uses) != [-1, 1] for uses in edges.values()):
        raise ValueError("source must be closed with consistently oriented edges")
    if len(vertices) - len(edges) + len(faces) != 2:
        raise ValueError("source certificate requires a sphere-topology mesh")
    for link in links:
        if not link or any(len(neighbors) != 2 for neighbors in link.values()):
            raise ValueError("source vertex link must be a single cycle")
        seen, pending = set(), [next(iter(link))]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(link[current])
        if len(seen) != len(link):
            raise ValueError("source vertex link must be a single cycle")
    # Multiple disjoint spheres can satisfy an Euler count if mixed with other
    # components. Require connectivity explicitly, independently of that count.
    adjacency = [set() for _ in vertices]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(adjacency[current])
    if len(seen) != len(vertices):
        raise ValueError("source must have one connected component")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths <= np.finfo(float).eps * max(1., float(np.max(np.abs(vertices)))) ** 2):
        raise ValueError("source has a geometrically degenerate triangle")
    return triangles, cross / lengths[:, None]


def _minimum_triangle_radius(triangles):
    """Exact origin-to-filled-triangle distance, including face interiors."""
    a, b, c = np.moveaxis(np.asarray(triangles, float), 1, 0)
    edge_distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        denominator = np.einsum("ij,ij->i", edge, edge)
        t = np.divide(-np.einsum("ij,ij->i", start, edge), denominator,
                      out=np.zeros(len(a)), where=denominator > 0)
        point = start + np.clip(t, 0, 1)[:, None] * edge
        edge_distances.append(np.linalg.norm(point, axis=1))
    u, v = b - a, c - a
    uu, uv, vv = (np.einsum("ij,ij->i", x, y)
                  for x, y in ((u, u), (u, v), (v, v)))
    au, av = np.einsum("ij,ij->i", a, u), np.einsum("ij,ij->i", a, v)
    determinant = uu * vv - uv * uv
    good = determinant > np.finfo(float).eps * np.maximum(uu * vv, 1e-300)
    s = np.divide(uv * av - vv * au, determinant, out=np.zeros(len(a)), where=good)
    t = np.divide(uv * au - uu * av, determinant, out=np.zeros(len(a)), where=good)
    point = a + s[:, None] * u + t[:, None] * v
    interior = good & (s >= 0) & (t >= 0) & (s + t <= 1)
    face_distances = np.where(interior, np.linalg.norm(point, axis=1), np.inf)
    return np.minimum(np.min(edge_distances, axis=0), face_distances)


def _cone_angle(normals, directions):
    lengths = np.linalg.norm(directions, axis=2)
    if np.any(lengths <= 0):
        raise ValueError("analytic normal direction is undefined")
    cosines = np.einsum("ij,ikj->ik", normals, directions) / lengths
    if np.any(cosines <= 0):
        raise ValueError("source faces must point outward within a positive normal cone")
    return float(np.degrees(np.arccos(np.clip(cosines, -1, 1))).max())


def _radial_bounds(triangles, normals, radii, center):
    shifted = triangles - np.asarray(center, float)
    unit = shifted / radii
    nominal_error = float(np.max(np.abs(np.linalg.norm(unit, axis=2) - 1)))
    if nominal_error > 2e-6:
        raise ValueError("source vertices differ from the authored radial surface")
    unit_cross = np.cross(unit[:, 1] - unit[:, 0], unit[:, 2] - unit[:, 0])
    if np.any(np.einsum("ij,ij->i", unit_cross, unit[:, 0]) <= 0):
        raise ValueError("source faces do not preserve outward radial orientation")
    inner = _minimum_triangle_radius(unit)
    outer = np.linalg.norm(unit, axis=2).max(axis=1)
    distance = float(max(radii) * np.maximum(np.abs(inner - 1), np.abs(outer - 1)).max())
    # Ellipsoid gradients at barycentric positions are positive combinations of
    # vertex gradients. Their cone is bounded by the largest vertex angle.
    angle = _cone_angle(normals, shifted / (radii * radii))
    return distance, angle, nominal_error


def reference_facet_certificate(family, parameters, reference):
    """Bound continuous source facets against the fixed authored ideal.

Normal correspondence is radial on spheres/ellipsoids and axial/radial on
revolved profiles. It is explicitly not a nearest-analytic-normal theorem for
an anisotropic ellipsoid. The source screen does not certify self-intersections
or any candidate's native boundary.
"""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError("independent analytic certificate is unavailable for " + family)
    if family in ("thin_plate", "concave_arch", "asymmetric_multipart_solid"):
        from blender_blocking.synthetic.quality_contracts import quality_workload
        from blender_blocking.evaluation.rectilinear_reference import rectilinear_reference_certificate
        authored = next(row["parameters"] for row in quality_workload()["cases"] if row["name"] == family)
        if parameters != authored:
            raise ValueError("rectilinear parameters differ from the independently frozen authored case")
        proof = rectilinear_reference_certificate(family, reference)
        if proof["status"] != "certified":
            raise ValueError(proof["reason"])
        return {"protocol": "continuous_authored_facet_bound_v1", "family": family,
                "reference_geometry_hash": reference.content_hash,
                "distance_bound_world": proof["maximum_source_facet_distance_world"],
                "normal_correspondence_bound_degrees": proof["maximum_normal_angle_degrees"],
                "vertex_construction_error_world": proof["maximum_vertex_rounding_shift_world"],
                "normal_correspondence": proof["normal_correspondence"],
                "distance_scope": proof["distance_correspondence"],
                "candidate_boundary_qualified": False, "sampled": False,
                "rectilinear_exact_cover": proof}
    if family == "torus":
        from blender_blocking.synthetic.quality_contracts import quality_workload
        from blender_blocking.evaluation.torus_reference import torus_reference_certificate
        authored = next(row["parameters"] for row in quality_workload()["cases"] if row["name"] == family)
        if parameters != authored:
            raise ValueError("torus parameters differ from the independently frozen authored case")
        proof = torus_reference_certificate(reference)
        if proof["status"] != "certified":
            raise ValueError(proof["reason"])
        return {"protocol": "continuous_authored_facet_bound_v1", "family": family,
                "reference_geometry_hash": reference.content_hash,
                "distance_bound_world": proof["maximum_source_facet_distance_world"],
                "normal_correspondence_bound_degrees": proof["maximum_normal_angle_degrees"],
                "vertex_construction_error_world": proof["maximum_vertex_construction_shift_world"],
                "normal_correspondence": proof["normal_correspondence"],
                "distance_scope": proof["distance_correspondence"],
                "candidate_boundary_qualified": False, "sampled": False,
                "torus_parameter_cover": proof}
    triangles, normals = _closed_oriented_sphere(reference)
    distances, angles, errors = [], [], []
    if family in ("sphere", "anisotropic_ellipsoid"):
        raw = ([parameters["radius"]] * 3 if family == "sphere" else parameters["radii"])
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError("authored radii must contain three values")
        radii = np.array([_positive(value, "radius") for value in raw])
        distance, angle, error = _radial_bounds(triangles, normals, radii, [0, 0, 0])
        distances.append(distance); angles.append(angle); errors.append(error * max(radii))
    else:
        if family == "capsule":
            radius = _positive(parameters["radius"], "radius")
            segment = _positive(parameters["segment_height"], "segment_height")
            lower, upper, rb, rt = -segment / 2, segment / 2, radius, radius
            tolerance = 2e-6 * max(1., radius, segment)
            north = np.all(triangles[:, :, 2] >= upper - tolerance, axis=1)
            south = np.all(triangles[:, :, 2] <= lower + tolerance, axis=1)
            for selected, center in ((north, [0, 0, upper]), (south, [0, 0, lower])):
                if not selected.any():
                    raise ValueError("capsule source is missing an authored hemisphere")
                distance, angle, error = _radial_bounds(triangles[selected], normals[selected],
                                                      np.full(3, radius), center)
                distances.append(distance); angles.append(angle); errors.append(error * radius)
            side = ~(north | south)
        else:
            height = _positive(parameters["height"], "height")
            lower, upper = -height / 2, height / 2
            rb = _positive(parameters["radius"] if family == "cylinder" else parameters["radius_bottom"], "radius_bottom")
            rt = _positive(parameters["radius"] if family == "cylinder" else parameters["radius_top"], "radius_top")
            tolerance = 2e-6 * max(1., rb, rt, height)
            north = np.all(np.abs(triangles[:, :, 2] - upper) <= tolerance, axis=1)
            south = np.all(np.abs(triangles[:, :, 2] - lower) <= tolerance, axis=1)
            if not north.any() or not south.any():
                raise ValueError("revolved source must retain both authored flat caps")
            for selected, sign in ((north, 1), (south, -1)):
                directions = np.zeros_like(triangles[selected]); directions[:, :, 2] = sign
                angles.append(_cone_angle(normals[selected], directions))
                errors.append(float(np.max(np.abs(triangles[selected, :, 2] - sign * height / 2))))
                radii = np.linalg.norm(triangles[selected, :, :2], axis=2)
                if np.any(radii > (rt if sign == 1 else rb) + tolerance):
                    raise ValueError("source cap exceeds the authored disk")
            side = ~(north | south)
        points = triangles[side]
        if not len(points) or np.any(points[:, :, 2] < lower - tolerance) or np.any(points[:, :, 2] > upper + tolerance):
            raise ValueError("source triangles cross an authored profile join")
        slope = (rt - rb) / (upper - lower)
        authored_radius = rb + slope * (points[:, :, 2] - lower)
        xy = points[:, :, :2] / authored_radius[:, :, None]
        error = float(np.max(np.abs(np.linalg.norm(xy, axis=2) - 1)))
        if error > 2e-6:
            raise ValueError("source side vertices differ from the authored revolved profile")
        # Positive radial weights preserve normalized azimuth intervals. Each
        # face must use at most two rays, so its arc lies in the positive cone;
        # the outward dot product reaches its minimum at an arc endpoint.
        ray_dots = np.einsum("ikj,ilj->ikl", xy, xy)
        if np.any(ray_dots <= 0):
            raise ValueError("source side azimuth interval exceeds the certificate cone")
        determinant = xy[:, :, 0] * np.roll(xy[:, :, 1], 1, axis=1) - xy[:, :, 1] * np.roll(xy[:, :, 0], 1, axis=1)
        if np.any(np.min(np.abs(determinant), axis=1) > 1e-5):
            raise ValueError("source side triangle must use at most two azimuth rays")
        radial3 = np.concatenate((xy, np.zeros((*xy.shape[:2], 1))), axis=2)
        inner = _minimum_triangle_radius(radial3)
        outer = np.linalg.norm(xy, axis=2).max(axis=1)
        distances.append(float(max(rb, rt) * np.maximum(np.abs(inner - 1), np.abs(outer - 1)).max()))
        directions = np.concatenate((xy, np.full((*xy.shape[:2], 1), -slope)), axis=2)
        angles.append(_cone_angle(normals[side], directions))
        errors.append(error * max(rb, rt))
    return {"protocol": "continuous_authored_facet_bound_v1", "family": family,
            "reference_geometry_hash": reference.content_hash,
            "distance_bound_world": max(distances + errors),
            "normal_correspondence_bound_degrees": max(angles),
            "vertex_construction_error_world": max(errors),
            "normal_correspondence": "radial/axial analytic correspondence; not nearest-point correspondence",
            "distance_scope": "continuous source facets to authored ideal; radial covering for convex sphere-topology references",
            "candidate_boundary_qualified": False, "sampled": False}


def freeze_family_surface_contract(family, parameters, reference, cameras, *,
                                   reconstruction_pixels=.5, normal_allowance_degrees=1.):
    """Freeze a source-conditioned policy without accepting candidate inputs."""
    pixels = _positive(reconstruction_pixels, "reconstruction_pixels")
    normal_allowance = _positive(normal_allowance_degrees, "normal_allowance_degrees")
    if pixels > 1 or normal_allowance > 5:
        raise ValueError("engineering allowance exceeds the bounded policy")
    if set(cameras) != set(CANONICAL_VIEWS):
        raise ValueError("all five actual source cameras are required")
    camera_rows, pixel_sizes = {}, []
    for view in CANONICAL_VIEWS:
        record = cameras[view]
        if record.get("projection") != "ORTHO" or any(key not in record for key in
                ("resolution", "pixel_aspect", "shift_x", "shift_y", "clip_start", "clip_end")):
            raise ValueError("actual source camera provenance is incomplete")
        start, end = float(record["clip_start"]), float(record["clip_end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 < start < end:
            raise ValueError("actual source camera clipping is invalid")
        digest = camera_frame_sha256(record)
        matrix = np.asarray(record["matrix_world"], float)
        if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0) or
                not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-6, rtol=0) or
                np.linalg.det(matrix[:3, :3]) <= 0):
            raise ValueError("source camera frame must be rigid and right-handed")
        width, height = record["resolution"]
        ax, ay = record["pixel_aspect"]
        aspect = width * ax / (height * ay)
        scale = float(record["ortho_scale"])
        horizontal, vertical = (scale, scale / aspect) if aspect >= 1 else (scale * aspect, scale)
        pitch = max(horizontal / width, vertical / height)
        pixel_sizes.append(pitch)
        camera_rows[view] = {"frame_sha256": digest, "clip_start": start, "clip_end": end,
                             "max_pixel_pitch_world": pitch}
    certificate, reason = None, None
    try:
        certificate = reference_facet_certificate(family, parameters, reference)
    except ValueError as exc:
        reason = str(exc)
    limits = (None if certificate is None else {
        "symmetric_mean_distance_world_max": certificate["distance_bound_world"] + pixels * max(pixel_sizes),
        "normal_angle_p95_degrees_max": min(180., certificate["normal_correspondence_bound_degrees"] + normal_allowance)})
    declaration = {"protocol": PROTOCOL, "family": family,
                   "reference_geometry_hash": reference.content_hash,
                   "authored_parameters": deepcopy(parameters), "cameras": camera_rows,
                   "reference_certificate": certificate, "engineering_limits": limits,
                   "engineering_status": "frozen" if limits else "unqualified",
                   "reason": reason,
                   "reconstruction_allowance": {"source_pixels": pixels,
                       "world": pixels * max(pixel_sizes), "normal_degrees": normal_allowance,
                       "basis": "explicit engineering allowance; not an observed artist tolerance"},
                   "metric": {"protocol": "shared_world_area_sample_to_triangle_v1",
                              "sample_count_per_direction": 4096, "seed": 61007},
                   "artist_acceptance_limits": None,
                   "artist_acceptance_status": "unqualified; supplementary engineering policy does not replace required family gates",
                   "historical_candidate_exposure": "not claimed blind; algorithm consumes source inputs only"}
    declaration["contract_sha256"] = _digest(declaration)
    return declaration


def evaluate_family_surface_contract(contract, observation, *, candidate_geometry_hash):
    """Bind raw observations to a frozen policy; preserve independent failures."""
    declaration = deepcopy(contract)
    digest = declaration.pop("contract_sha256", None)
    if declaration.get("protocol") != PROTOCOL or digest != _digest(declaration):
        raise ValueError("frozen family contract digest is invalid")
    if (not isinstance(candidate_geometry_hash, str) or len(candidate_geometry_hash) != 64 or
            any(c not in "0123456789abcdef" for c in candidate_geometry_hash)):
        raise ValueError("candidate geometry identity must be a SHA-256")
    expected = {**contract["metric"], "reference_geometry_hash": contract["reference_geometry_hash"],
                "candidate_geometry_hash": candidate_geometry_hash,
                "normal_orientation": "oriented; opposite normals are 180 degrees",
                "normal_interpretation": "geometric face normals, not shading normals",
                "units": "unchanged Blender world coordinates; no fitting or normalization"}
    if any(observation.get(key) != value or isinstance(observation.get(key), bool)
           for key, value in expected.items()):
        raise ValueError("raw observation protocol or source/candidate binding differs")
    values = {}
    for name in ("symmetric_mean_distance_world", "distance_p95_world", "sampled_max_distance_world",
                 "normal_angle_mean_degrees", "normal_angle_p95_degrees"):
        value = observation.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("raw observation contains invalid distances or oriented angles")
        if "degrees" in name and value > 180:
            raise ValueError("oriented normal angle exceeds 180 degrees")
        values[name] = value
    if values["symmetric_mean_distance_world"] > values["sampled_max_distance_world"] or values["distance_p95_world"] > values["sampled_max_distance_world"]:
        raise ValueError("raw distance summary is inconsistent")
    limits = contract["engineering_limits"]
    distance = None if limits is None else values["symmetric_mean_distance_world"] <= limits["symmetric_mean_distance_world_max"]
    normal = None if limits is None else values["normal_angle_p95_degrees"] <= limits["normal_angle_p95_degrees_max"]
    return {"protocol": "source_conditioned_surface_observation_v1",
            "family": contract["family"], "contract_sha256": digest,
            "reference_geometry_hash": contract["reference_geometry_hash"],
            "candidate_geometry_hash": candidate_geometry_hash, "raw_metrics": values,
            "engineering_limits": deepcopy(limits), "engineering_distance_passed": distance,
            "engineering_normal_passed": normal,
            "engineering_surface_passed": None if limits is None else bool(distance and normal),
            "artist_surface_passed": None, "aggregate_passed": False,
            "independent_verdicts": {"silhouette": "unassessed", "topology": "unassessed", "editability": "unassessed"},
            "scope": "supplementary engineering evaluation; cannot replace artist or existing family acceptance"}
