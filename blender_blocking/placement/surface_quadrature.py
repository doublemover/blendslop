"""Deterministic area weights for exact analytic surface samples."""
from __future__ import annotations
import numpy as np


def surface_area_weights(primitive, samples, *, mesh_resolution=12):
    """Assign coarse mesh area to nearest analytic samples without moving them.

    The quadrature is an explicit finite tessellation approximation, not a claim
    of uniform transformed-sphere sampling or exact analytic surface area.
    """
    if not len(samples):
        return np.empty(0)
    if not hasattr(primitive, "to_mesh_data"):
        raise TypeError("area-aware objective requires a primitive mesh contract")
    from scipy.spatial import cKDTree
    mesh = primitive.to_mesh_data(mesh_resolution)
    faces = np.asarray([(f[0], f[i], f[i+1]) for f in mesh.faces
                        for i in range(1, len(f)-1)], int)
    if not len(faces):
        raise ValueError("area-aware surface quadrature requires nonempty faces")
    a, b, c = (mesh.vertices[faces[:, i]] for i in range(3))
    areas = .5 * np.linalg.norm(np.cross(b-a, c-a), axis=1)
    if not np.isfinite(areas).all() or not areas.sum() > 0.:
        raise ValueError("surface quadrature requires finite positive mesh area")
    _, nearest = cKDTree(samples).query((a+b+c)/3., workers=1)
    return np.bincount(nearest, weights=areas, minlength=len(samples))
