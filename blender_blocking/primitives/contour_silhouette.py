"""Continuous pixel-center distance to a declared-precision projected union.

This smooth proposal surrogate is not a final opaque raster or an exact pixel
coverage filter. Resolved holes and disconnected loops above its precision
contract are retained. Only
circular-frustum chord error has an analytic representation bound here; other
families explicitly retain their finite-resolution mesh approximation.
"""
import time

import numpy as np


def projection_resolution(part, camera, resolution=None, chord_error_px=.25):
    if type(part).__name__ == "SuperFrustum":
        dimensions = np.asarray([part.radius_bottom, part.radius_top, part.height], float)
        if (not np.isfinite(dimensions).all() or np.any(dimensions[:2] < 0)
                or dimensions[2] <= 0):
            raise ValueError("frustum contour requires nonnegative radii and positive height")
    if resolution is not None:
        return max(8, int(resolution)), {"policy": "explicit_mesh_resolution",
                                       "geometric_error_bound_px": None}
    if type(part).__name__ == "SuperFrustum":
        width, height = camera.image_size
        u0, u1, v0, v1 = camera.world_bounds
        scale = max(width/(u1-u0), height/(v1-v0))
        radius = max(float(part.radius_bottom), float(part.radius_top))*scale
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("frustum radii must be finite and nonnegative")
        angle = np.arccos(np.clip(1-chord_error_px/max(radius, 1e-12), -1, 1))
        count = min(128, max(16, int(np.ceil(np.pi/max(angle, 1e-12)))))
        error = radius*(1-np.cos(np.pi/count))
        return count, {"policy": "circular_frustum_projected_chord",
                       "requested_chord_error_px": chord_error_px,
                       "geometric_error_bound_px": float(error),
                       "within_requested_chord_error": bool(error <= chord_error_px+1e-12),
                       "bound_scope": "circular-frustum mesh chord approximation before union precision"}
    return 48, {"policy": "declared_finite_mesh_approximation",
                "geometric_error_bound_px": None}


def projected_union(part, camera, resolution):
    from shapely import Polygon, MultiPoint, union_all

    mesh = part.to_mesh_data(resolution)
    vertices = np.asarray(mesh.vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("contour projection requires finite Nx3 mesh vertices")
    width, height = camera.image_size
    u0, u1, v0, v1 = camera.world_bounds
    xy = np.column_stack([
        (vertices[:, camera.axes[0]]-u0)/(u1-u0)*width-.5,
        (v1-vertices[:, camera.axes[1]])/(v1-v0)*height-.5])
    if not np.isfinite(xy).all():
        raise ValueError("contour projected vertices must be finite")
    convex = ((type(part).__name__ == "SuperFrustum"
               and part.radius_bottom >= 0 and part.radius_top >= 0 and part.height > 0)
              or (type(part).__name__ == "SuperquadricPrimitive"
                  and 0 < part.epsilon1 <= 2 and 0 < part.epsilon2 <= 2))
    triangles = []
    triangle_count = 0
    for face in mesh.faces:
        ids = np.asarray(face, np.int64)
        if np.any(ids < 0) or np.any(ids >= len(vertices)):
            raise ValueError("contour projection contains invalid mesh indices")
        for index in range(1, len(ids)-1):
            points = xy[ids[[0, index, index+1]]]
            first, second = points[1]-points[0], points[2]-points[0]
            if first[0]*second[1]-first[1]*second[0] != 0:
                triangle_count += 1
                if not convex:
                    triangles.append(Polygon(points))
            if triangle_count > 16384:
                raise ValueError("contour projection triangle allowance exceeded")
    if not triangle_count:
        raise ValueError("contour projection has no nonzero area")
    # Exact-double GEOS unions can leave machine-scale internal sliver holes
    # between projected triangles. Their distance field is not a valid smooth
    # contour surrogate. Declare a fixed 1e-9-pixel precision ONLY for this 2D
    # proposal operator; actual mesh coordinates and exact contact guards stay
    # unchanged. Features below this precision are not promised by the surrogate.
    precision_px = 1e-9
    # These declared families are convex (superquadric norm exponents >= 1).
    # Their projection is convex, so resolve its actual finite mesh vertices
    # directly. This avoids GEOS phantom interior slivers between redundant
    # projected triangles without inventing holes or applying a hull to a
    # concave/unknown family.
    shape = MultiPoint(xy).convex_hull if convex else union_all(triangles, grid_size=precision_px)
    if shape.is_empty or not shape.is_valid or shape.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("contour projected union is invalid")
    components = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    if len(components) > 32:
        raise ValueError("contour component allowance exceeded")
    # A distance surrogate must not amplify a sub-grid point-hole into a
    # pixel-wide low-confidence crater. Explicitly leave holes smaller than
    # 1e-7 pixels unresolved by this proposal operator. Large or long thin
    # holes remain, regardless of area. Final mesh/ray geometry is unchanged.
    minimum_hole_diameter_px = 1e-7
    suppressed = 0
    suppressed_components = 0
    resolved_components = []
    for polygon in components:
        if np.ptp(np.asarray(polygon.exterior.coords), axis=0).max() <= minimum_hole_diameter_px:
            suppressed_components += 1
            continue
        interiors = []
        for ring in polygon.interiors:
            coordinates = np.asarray(ring.coords, float)
            if np.ptp(coordinates, axis=0).max() <= minimum_hole_diameter_px:
                suppressed += 1
            else:
                interiors.append(coordinates)
        resolved_components.append(Polygon(polygon.exterior.coords, interiors))
    if not resolved_components:
        raise ValueError("contour is unresolved at declared proposal precision")
    shape = union_all(resolved_components)
    loops = []
    for polygon in resolved_components:
        loops.append(np.asarray(polygon.exterior.coords, float))
        loops.extend(np.asarray(ring.coords, float) for ring in polygon.interiors)
    if sum(len(loop)-1 for loop in loops) > 4096:
        raise ValueError("contour boundary allowance exceeded")
    return shape, loops, {"triangles": triangle_count, "components": len(resolved_components),
                          "projection_representation": "convex_primitive_vertex_hull" if convex else "mesh_triangle_union",
                          "unresolved_small_components": suppressed_components,
                          "loops": len(loops), "union_precision_grid_px": 0. if convex else precision_px,
                          "unresolved_hole_diameter_px": minimum_hole_diameter_px,
                          "unresolved_small_holes": suppressed,
                          "precision_scope": "proposal contours only; topology below grid unresolved"}


def contour_footprint(part, camera, softness=24., resolution=None, *, return_metadata=False):
    try:
        from shapely import contains_xy
    except ImportError as exc:
        raise RuntimeError("continuous nonellipse contours require Shapely 2.x; raster fallback disabled") from exc

    if not np.isfinite(softness) or softness <= 0:
        raise ValueError("contour softness must be finite and positive")
    width, height = camera.image_size
    if width*height > 262144:
        raise ValueError("contour pixel allowance exceeded")
    resolution, approximation = projection_resolution(part, camera, resolution)
    shape, loops, counts = projected_union(part, camera, resolution)
    starts = np.concatenate([loop[:-1] for loop in loops])
    vectors = np.concatenate([np.diff(loop, axis=0) for loop in loops])
    lengths = np.einsum('ij,ij->i', vectors, vectors)
    keep = lengths > 0
    starts, vectors, lengths = starts[keep], vectors[keep], lengths[keep]
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    distance = np.full(len(points), np.inf)
    started = last_progress = time.perf_counter()
    # Bounded work arrays: pixel tiles x boundary segments, never pixel x all
    # mesh triangles. Geometry is evaluated at the original subpixel positions.
    for offset in range(0, len(points), 256):
        tile = points[offset:offset+256]
        minimum = np.full(len(tile), np.inf)
        for edge in range(0, len(starts), 256):
            delta = tile[:, None, :]-starts[None, edge:edge+256]
            direction = vectors[edge:edge+256]
            fraction = np.clip(np.einsum('pqi,qi->pq', delta, direction)/lengths[edge:edge+256], 0, 1)
            residual = delta-fraction[:, :, None]*direction
            minimum = np.minimum(minimum, np.einsum('pqi,pqi->pq', residual, residual).min(axis=1))
        distance[offset:offset+len(tile)] = np.sqrt(minimum)
        now = time.perf_counter()
        if now-last_progress >= 10:
            print(f"contour pixels={offset+len(tile)}/{len(points)} elapsed={now-started:.2f}s", flush=True)
            last_progress = now
    signed = np.where(contains_xy(shape, points[:, 0], points[:, 1]), -distance, distance)
    opacity = float(getattr(part, 'opacity', getattr(part, 'density', 1.)))
    opacity *= float(getattr(part, 'confidence', 1.))
    if not np.isfinite(opacity):
        raise ValueError("contour opacity/confidence must be finite")
    mask = np.clip(opacity, 0, 1)/(1+np.exp(np.clip(signed*max(1., softness/12.), -60, 60)))
    mask = mask.reshape(height, width)
    report = {**counts, **approximation, "resolution": resolution,
              "operator": "continuous_projected_union_signed_distance_v1",
              "pixel_convention": "cell centers; image-space indices",
              "final_opaque_admission": False}
    return (mask, report) if return_metadata else mask
