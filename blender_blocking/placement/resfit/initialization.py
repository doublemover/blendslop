from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_from_profile_bands,
    initialize_gaussians_from_points,
    initialize_superfrusta_from_points,
    initialize_superquadrics_from_points,
)
from ..resfit_objective import (
    PenaltyHook,
    ResFitLossWeights,
    ResFitObjectiveResult,
    SilhouetteHook,
    evaluate_resfit_objective,
)
from ..resfit_optimizer import (
    CoordinateDescentConfig,
    OptimizationRecord,
    coordinate_descent_optimize,
)

from .config import ResFitPipelineConfig, ResFitPipelineResult

InitializerFn = Callable[[np.ndarray, PrimitiveInitializationConfig], Sequence[object]]


def get_initializer(family: str) -> InitializerFn:
    """Return the initializer for a primitive family."""
    normalized = family.lower().strip()
    if normalized in ("superfrustum", "superfrusta", "frustum", "capsule", "capsules"):
        return initialize_superfrusta_from_points
    if normalized in ("ellipsoid", "ellipsoids"):
        return initialize_ellipsoids_from_points
    if normalized in ("superquadric", "superquadrics", "boxy_superquadric"):
        return initialize_superquadrics_from_points
    if normalized in ("gaussian", "gaussians", "anisotropic_gaussian"):
        return initialize_gaussians_from_points
    raise ValueError(f"unknown primitive family: {family}")


def _family_supports_profile_init(primitive_family: str) -> bool:
    return str(primitive_family).lower().strip() in {
        "superfrustum",
        "superfrusta",
        "frustum",
        "capsule",
        "capsules",
    }


def _first_family(config: Mapping[str, object]) -> str:
    family = config.get("primitive_family")
    if family:
        return str(family)
    families = config.get("primitive_families")
    if isinstance(families, (list, tuple)) and families:
        return str(families[0])
    return "superfrustum"


def _candidate_families(config: Mapping[str, object]) -> tuple[str, ...]:
    family = config.get("primitive_family")
    if family:
        return (str(family),)
    families = config.get("primitive_families")
    if isinstance(families, str):
        raw = tuple(item.strip() for item in families.split(","))
    elif isinstance(families, Sequence):
        raw = tuple(str(item).strip() for item in families)
    else:
        raw = ("superfrustum",)
    deduped: list[str] = []
    for item in raw:
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped or ("superfrustum",))


def whole_support_seed(target, points, family, *, max_elapsed_s=.5):
    """A family-compatible, single oriented seed from direct mask supports."""
    from blender_blocking.reconstruction.spatial_regions import pca_box
    from blender_blocking.reconstruction.oriented_support import support_evidence, fit_whole_support
    from blender_blocking.reconstruction.projection_contract import observed_holes
    from blender_blocking.primitives.analytic_primitives import EllipsoidPrimitive, SuperquadricPrimitive, AnisotropicGaussianPrimitive
    from blender_blocking.primitives.superfrustum import SuperFrustum
    if len(points) < 8 or any(n >= 4 for n in observed_holes(target).values()):
        return None, {"status":"incompatible_known_hole_or_insufficient_points"}
    center, radii, frame = pca_box(points)
    evidence = support_evidence(target)
    normalized = family.lower().strip()
    if normalized in {"superfrustum", "superfrusta", "frustum", "capsule", "capsules"}:
        support_family,dimensions,rotation = "frustum", [np.sqrt(radii[1]*radii[2])]*2+[radii[0]], frame[:,[1,2,0]]
    elif normalized in {"superquadric", "superquadrics", "boxy_superquadric"}:
        support_family,dimensions,rotation = "box", radii, frame
    else:
        support_family,dimensions,rotation = "ellipsoid", radii, frame
    fit = fit_whole_support(support_family,evidence,center=center,dimensions=dimensions,rotation=rotation,
                            max_evaluations=64,max_elapsed_s=max_elapsed_s)
    center,dimensions,rotation = fit['center'],fit['dimensions'],fit['rotation']
    if support_family == "frustum":
        axis = rotation[:,2]
        part = SuperFrustum(position=center,
            orientation=(np.arctan2(axis[1],axis[0]),np.arccos(np.clip(axis[2],-1.,1.))),
            radius_bottom=dimensions[0],radius_top=dimensions[1],height=2.*dimensions[2])
    elif support_family == "box":
        # This is an explicitly approximate family-compatible proposal; the
        # actual superquadric field/mesh is subsequently scored, not box supports.
        part = SuperquadricPrimitive(center=center,radii=dimensions,rotation=rotation,epsilon1=.15,epsilon2=.15)
    elif normalized in {"gaussian", "gaussians", "anisotropic_gaussian"}:
        part = AnisotropicGaussianPrimitive(center=center,covariance=rotation@np.diag(dimensions**2)@rotation.T)
    else:
        part = EllipsoidPrimitive(center=center,radii=dimensions,rotation=rotation)
    report = {k:v for k,v in fit.items() if k not in {'center','dimensions','rotation'}}
    report['status']='proposal_only'
    return (part,),report
