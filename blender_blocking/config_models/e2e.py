from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *


@dataclass
class SilhouetteExtractConfig:
    """Configuration for silhouette extraction from images."""

    prefer_alpha: bool = True
    alpha_threshold: int = 127
    alpha_min_coverage: float = 0.001
    gray_threshold: Optional[int] = None
    invert_policy: str = "auto"
    polarity: str = "auto"
    min_area_frac: float = 0.0001
    max_area_frac: float = 0.98
    max_border_contact_frac: float = 0.95
    morph_close_px: int = 0
    morph_open_px: int = 0
    fill_holes: bool = False
    largest_component_only: bool = False
    min_component_area_px: int = 0
    candidate_scoring: bool = True
    emit_uncertainty: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if not (0 <= self.alpha_threshold <= 255):
            raise ValueError("alpha_threshold must be in [0, 255]")
        if not (0.0 <= self.alpha_min_coverage <= 1.0):
            raise ValueError("alpha_min_coverage must be in [0, 1]")
        if self.gray_threshold is not None and not (0 <= self.gray_threshold <= 255):
            raise ValueError("gray_threshold must be in [0, 255] when provided")
        if self.invert_policy not in _VALID_INVERT_POLICIES:
            raise ValueError(f"invert_policy must be one of {_VALID_INVERT_POLICIES}")
        if self.polarity not in _VALID_POLARITIES:
            raise ValueError(f"polarity must be one of {_VALID_POLARITIES}")
        if not (0.0 <= self.min_area_frac <= 1.0):
            raise ValueError("min_area_frac must be in [0, 1]")
        if not (0.0 <= self.max_area_frac <= 1.0):
            raise ValueError("max_area_frac must be in [0, 1]")
        if self.min_area_frac > self.max_area_frac:
            raise ValueError("min_area_frac must be <= max_area_frac")
        if not (0.0 <= self.max_border_contact_frac <= 1.0):
            raise ValueError("max_border_contact_frac must be in [0, 1]")
        if self.morph_close_px < 0 or self.morph_open_px < 0:
            raise ValueError("morph_close_px and morph_open_px must be >= 0")
        if self.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be >= 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "prefer_alpha": self.prefer_alpha,
            "alpha_threshold": self.alpha_threshold,
            "alpha_min_coverage": self.alpha_min_coverage,
            "gray_threshold": self.gray_threshold,
            "invert_policy": self.invert_policy,
            "polarity": self.polarity,
            "min_area_frac": self.min_area_frac,
            "max_area_frac": self.max_area_frac,
            "max_border_contact_frac": self.max_border_contact_frac,
            "morph_close_px": self.morph_close_px,
            "morph_open_px": self.morph_open_px,
            "fill_holes": self.fill_holes,
            "largest_component_only": self.largest_component_only,
            "min_component_area_px": self.min_component_area_px,
            "candidate_scoring": self.candidate_scoring,
            "emit_uncertainty": self.emit_uncertainty,
        }

@dataclass
class ProfileSamplingConfig:
    """Configuration for sampling silhouettes into profiles."""

    adaptive_sections: bool = False
    max_sections: int = 64
    section_tolerance: float = .012
    contour_sections: bool = False
    section_resolution: int = 64
    adaptive_sections: bool = False
    max_sections: int = 64
    section_tolerance: float = .012
    contour_sections: bool = False
    section_resolution: int = 64
    num_samples: int = 100
    sample_policy: str = "endpoints"
    fill_strategy: str = "interp_linear"
    smoothing_window: int = 3

    def validate(self) -> None:
        """Validate configuration values."""
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self.sample_policy not in _VALID_SAMPLE_POLICIES:
            raise ValueError(f"sample_policy must be one of {_VALID_SAMPLE_POLICIES}")
        if self.fill_strategy not in _VALID_FILL_STRATEGIES:
            raise ValueError(f"fill_strategy must be one of {_VALID_FILL_STRATEGIES}")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "num_samples": self.num_samples,
            "adaptive_sections": self.adaptive_sections,
            "max_sections": self.max_sections,
            "section_tolerance": self.section_tolerance,
            "contour_sections": self.contour_sections,
            "section_resolution": self.section_resolution,
            "adaptive_sections": self.adaptive_sections,
            "max_sections": self.max_sections,
            "section_tolerance": self.section_tolerance,
            "contour_sections": self.contour_sections,
            "section_resolution": self.section_resolution,
            "sample_policy": self.sample_policy,
            "fill_strategy": self.fill_strategy,
            "smoothing_window": self.smoothing_window,
        }

@dataclass
class LoftMeshOptions:
    """Configuration for loft mesh generation."""

    radial_segments: int = 24
    cap_mode: str = "fan"
    min_radius_u: float = 0.0
    merge_threshold_u: float = 0.0
    recalc_normals: bool = True
    shade_smooth: bool = True
    weld_degenerate_rings: bool = True
    adaptive_radial_segments: bool = False
    min_adaptive_radial_segments: int = 12
    max_adaptive_radial_segments: int = 96
    topology_strict: bool = True
    research_allow_low_radial_segments: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        min_segments = 3 if self.research_allow_low_radial_segments else 12
        if self.radial_segments < min_segments:
            raise ValueError(f"radial_segments must be >= {min_segments}")
        if self.cap_mode not in _VALID_CAP_MODES:
            raise ValueError(f"cap_mode must be one of {_VALID_CAP_MODES}")
        if self.min_radius_u < 0 or self.merge_threshold_u < 0:
            raise ValueError("min_radius_u and merge_threshold_u must be >= 0")
        if self.min_adaptive_radial_segments < 3:
            raise ValueError("min_adaptive_radial_segments must be >= 3")
        if self.max_adaptive_radial_segments < self.min_adaptive_radial_segments:
            raise ValueError(
                "max_adaptive_radial_segments must be >= min_adaptive_radial_segments"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "radial_segments": self.radial_segments,
            "cap_mode": self.cap_mode,
            "min_radius_u": self.min_radius_u,
            "merge_threshold_u": self.merge_threshold_u,
            "recalc_normals": self.recalc_normals,
            "shade_smooth": self.shade_smooth,
            "weld_degenerate_rings": self.weld_degenerate_rings,
            "adaptive_radial_segments": self.adaptive_radial_segments,
            "min_adaptive_radial_segments": self.min_adaptive_radial_segments,
            "max_adaptive_radial_segments": self.max_adaptive_radial_segments,
            "topology_strict": self.topology_strict,
            "research_allow_low_radial_segments": self.research_allow_low_radial_segments,
        }

@dataclass
class RenderConfig:
    """Configuration for rendering orthographic silhouettes."""

    resolution: Tuple[int, int] = (512, 512)
    engine: str = "BLENDER_EEVEE"
    transparent_bg: bool = True
    samples: int = 1
    margin_frac: float = 0.08
    color_mode: str = "RGBA"
    force_material: bool = False
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    camera_distance_factor: float = 2.0
    party_mode: bool = False
    view_calibration: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values."""
        from reconstruction.projection_contract import validate_view_calibration
        validate_view_calibration(self.view_calibration)
        if len(self.resolution) != 2:
            raise ValueError("resolution must be a (width, height) tuple")
        if any(val < 32 for val in self.resolution):
            raise ValueError("resolution values must be >= 32")
        if self.engine not in _VALID_RENDER_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_RENDER_ENGINES}")
        if self.samples < 1:
            raise ValueError("samples must be >= 1")
        if not (0.0 <= self.margin_frac <= 1.0):
            raise ValueError("margin_frac must be in [0, 1]")
        if self.color_mode not in _VALID_COLOR_MODES:
            raise ValueError(f"color_mode must be one of {_VALID_COLOR_MODES}")
        if len(self.background_color) != 4:
            raise ValueError("background_color must be RGBA with 4 values")
        if len(self.silhouette_color) != 4:
            raise ValueError("silhouette_color must be RGBA with 4 values")
        if any(not (0.0 <= val <= 1.0) for val in self.background_color):
            raise ValueError("background_color values must be in [0, 1]")
        if any(not (0.0 <= val <= 1.0) for val in self.silhouette_color):
            raise ValueError("silhouette_color values must be in [0, 1]")
        if self.camera_distance_factor <= 0:
            raise ValueError("camera_distance_factor must be > 0")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "resolution": list(self.resolution),
            "engine": self.engine,
            "transparent_bg": self.transparent_bg,
            "samples": self.samples,
            "margin_frac": self.margin_frac,
            "color_mode": self.color_mode,
            "force_material": self.force_material,
            "background_color": list(self.background_color),
            "silhouette_color": list(self.silhouette_color),
            "camera_distance_factor": self.camera_distance_factor,
            "party_mode": self.party_mode,
            "view_calibration": self.view_calibration,
        }

@dataclass
class CanonicalizeConfig:
    """Configuration for canonicalizing silhouette masks."""

    output_size: int = 256
    padding_frac: float = 0.1
    anchor: str = "bottom_center"
    interp: str = "nearest"
    use_cache: bool = True
    digest_algorithm: str = "sha256"
    fill_holes: Optional[bool] = None
    largest_component_only: Optional[bool] = None

    def validate(self) -> None:
        """Validate configuration values."""
        if self.output_size < 32:
            raise ValueError("output_size must be >= 32")
        if not (0.0 <= self.padding_frac <= 1.0):
            raise ValueError("padding_frac must be in [0, 1]")
        if self.anchor not in _VALID_CANON_ANCHORS:
            raise ValueError(f"anchor must be one of {_VALID_CANON_ANCHORS}")
        if self.interp not in _VALID_CANON_INTERP:
            raise ValueError(f"interp must be one of {_VALID_CANON_INTERP}")
        if self.digest_algorithm not in {"sha256"}:
            raise ValueError("digest_algorithm must be sha256")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "output_size": self.output_size,
            "padding_frac": self.padding_frac,
            "anchor": self.anchor,
            "interp": self.interp,
            "use_cache": self.use_cache,
            "digest_algorithm": self.digest_algorithm,
            "fill_holes": self.fill_holes,
            "largest_component_only": self.largest_component_only,
        }

@dataclass
class MeshJoinConfig:
    """Configuration for mesh join behavior."""

    mode: str = "boolean"
    boolean_solver: str = "auto"
    allow_degraded_simple_join: bool = True
    record_attempts: bool = True
    balanced_boolean_tree: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.mode not in _VALID_JOIN_MODES:
            raise ValueError(f"mode must be one of {_VALID_JOIN_MODES}")
        if self.boolean_solver not in _VALID_BOOLEAN_SOLVERS:
            raise ValueError(f"boolean_solver must be one of {_VALID_BOOLEAN_SOLVERS}")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dict."""
        return {
            "mode": self.mode,
            "boolean_solver": self.boolean_solver,
            "allow_degraded_simple_join": self.allow_degraded_simple_join,
            "record_attempts": self.record_attempts,
            "balanced_boolean_tree": self.balanced_boolean_tree,
        }
