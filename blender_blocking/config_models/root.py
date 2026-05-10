from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *

from .backends import (
    CandidateConfig,
    ConstraintConfig,
    DifferentiableRenderConfig,
    EnsembleConfig,
    GaussianEllipsoidConfig,
    PrimitiveFitConfig,
    ReconstructionConfig,
    ShapeProgramConfig,
    SilhouetteIntersectionConfig,
    VisualHullConfig,
    VolumeConfig,
)
from .budgets import QualityBudgetConfig
from .e2e import CanonicalizeConfig, LoftMeshOptions, MeshJoinConfig, ProfileSamplingConfig, RenderConfig, SilhouetteExtractConfig
from .refinement import RefinementLabConfig
from .synthetic import SyntheticFactoryConfig

@dataclass
class BlockingConfig:
    """Root configuration for the blocking workflow."""

    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh_join: MeshJoinConfig = field(default_factory=MeshJoinConfig)
    silhouette_extract_ref: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_extract_render: SilhouetteExtractConfig = field(
        default_factory=SilhouetteExtractConfig
    )
    silhouette_intersection: SilhouetteIntersectionConfig = field(
        default_factory=SilhouetteIntersectionConfig
    )
    profile_sampling: ProfileSamplingConfig = field(
        default_factory=ProfileSamplingConfig
    )
    mesh_from_profile: LoftMeshOptions = field(default_factory=LoftMeshOptions)
    render_silhouette: RenderConfig = field(default_factory=RenderConfig)
    canonicalize: CanonicalizeConfig = field(default_factory=CanonicalizeConfig)
    visual_hull: VisualHullConfig = field(default_factory=VisualHullConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    primitive_fit: PrimitiveFitConfig = field(default_factory=PrimitiveFitConfig)
    gaussian_ellipsoid: GaussianEllipsoidConfig = field(
        default_factory=GaussianEllipsoidConfig
    )
    differentiable_render: DifferentiableRenderConfig = field(
        default_factory=DifferentiableRenderConfig
    )
    shape_program: ShapeProgramConfig = field(default_factory=ShapeProgramConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    synthetic_factory: SyntheticFactoryConfig = field(default_factory=SyntheticFactoryConfig)
    quality_budget: QualityBudgetConfig = field(default_factory=QualityBudgetConfig)
    refinement_lab: RefinementLabConfig = field(default_factory=RefinementLabConfig)

    def validate(self) -> None:
        """Validate configuration values across groups."""
        self.reconstruction.validate()
        self.mesh_join.validate()
        self.silhouette_extract_ref.validate()
        self.silhouette_extract_render.validate()
        self.silhouette_intersection.validate()
        self.profile_sampling.validate()
        self.mesh_from_profile.validate()
        self.render_silhouette.validate()
        self.canonicalize.validate()
        self.visual_hull.validate()
        self.volume.validate()
        self.ensemble.validate()
        self.primitive_fit.validate()
        self.gaussian_ellipsoid.validate()
        self.differentiable_render.validate()
        self.shape_program.validate()
        self.constraints.validate()
        self.synthetic_factory.validate()
        self.quality_budget.validate()
        self.refinement_lab.validate()

        # Placeholder for mutually exclusive scale policies if added later.

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict matching the canonical schema."""
        return {
            "reconstruction": self.reconstruction.to_dict(),
            "mesh_join": self.mesh_join.to_dict(),
            "silhouette_extract_ref": self.silhouette_extract_ref.to_dict(),
            "silhouette_extract_render": self.silhouette_extract_render.to_dict(),
            "silhouette_intersection": self.silhouette_intersection.to_dict(),
            "profile_sampling": self.profile_sampling.to_dict(),
            "mesh_from_profile": self.mesh_from_profile.to_dict(),
            "canonicalize": self.canonicalize.to_dict(),
            "render_silhouette": self.render_silhouette.to_dict(),
            "visual_hull": self.visual_hull.to_dict(),
            "volume": self.volume.to_dict(),
            "ensemble": self.ensemble.to_dict(),
            "primitive_fit": self.primitive_fit.to_dict(),
            "gaussian_ellipsoid": self.gaussian_ellipsoid.to_dict(),
            "differentiable_render": self.differentiable_render.to_dict(),
            "shape_program": self.shape_program.to_dict(),
            "constraints": self.constraints.to_dict(),
            "synthetic_factory": self.synthetic_factory.to_dict(),
            "quality_budget": self.quality_budget.to_dict(),
            "refinement_lab": self.refinement_lab.to_dict(),
        }
