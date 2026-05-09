"""Canonical evaluation metric names and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str | None
    higher_is_better: bool | None
    description: str
    required_metadata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
            "required_metadata": list(self.required_metadata),
        }


_DEFINITIONS = (
    MetricDefinition("silhouette.min_view_iou", None, True, "Minimum required-view area IoU."),
    MetricDefinition("silhouette.average_iou", None, True, "Mean required-view area IoU."),
    MetricDefinition("silhouette.mean_boundary_iou", None, True, "Mean required-view Boundary IoU."),
    MetricDefinition("silhouette.min_boundary_iou", None, True, "Minimum required-view Boundary IoU."),
    MetricDefinition("silhouette.mean_signed_distance_loss", None, False, "Mean normalized signed-distance silhouette loss."),
    MetricDefinition("silhouette.required_view_count", "view", None, "Required reference-view count."),
    MetricDefinition("silhouette.failed_required_view_count", "view", False, "Required reference views that failed their local gate."),
    MetricDefinition("silhouette.missing_required_metric_count", "metric", False, "Missing required per-view silhouette metric fields."),
    MetricDefinition("geometry.chamfer_l1", "normalized_distance", False, "Symmetric nearest-neighbor L1 Chamfer distance."),
    MetricDefinition("geometry.chamfer_l2", "normalized_distance_squared", False, "Symmetric nearest-neighbor squared Chamfer distance."),
    MetricDefinition("geometry.fscore_tau", None, True, "Surface F-score at configured tolerance.", ("tau",)),
    MetricDefinition("geometry.fscore_tolerance", "world_units", None, "Tolerance used for surface F-score."),
    MetricDefinition("geometry.volumetric_iou", None, True, "Occupancy volume intersection over union."),
    MetricDefinition("geometry.normal_consistency", None, True, "Nearest-neighbor normal consistency."),
    MetricDefinition("geometry.surface_coverage", None, True, "Fraction of ground-truth surface covered by candidate samples."),
    MetricDefinition("geometry.ambiguity_gap", None, False, "Estimated information gap caused by silhouette-only ambiguity."),
    MetricDefinition("geometry.true.chamfer_l1", "normalized_distance", False, "Distance to true synthetic/captured geometry."),
    MetricDefinition("geometry.true.chamfer_l2", "normalized_distance_squared", False, "Squared distance to true synthetic/captured geometry."),
    MetricDefinition("geometry.true.fscore_tau", None, True, "Surface F-score against true geometry."),
    MetricDefinition("geometry.true.volumetric_iou", None, True, "Volume IoU against true occupancy."),
    MetricDefinition("geometry.recoverable.chamfer_l1", "normalized_distance", False, "Distance to silhouette-recoverable envelope."),
    MetricDefinition("geometry.recoverable.chamfer_l2", "normalized_distance_squared", False, "Squared distance to silhouette-recoverable envelope."),
    MetricDefinition("geometry.recoverable.fscore_tau", None, True, "Surface F-score against silhouette-recoverable envelope."),
    MetricDefinition("geometry.recoverable.volumetric_iou", None, True, "Volume IoU against silhouette-recoverable envelope."),
    MetricDefinition("geometry.ambiguity_gap_chamfer_l1", "normalized_distance", False, "True-vs-recoverable Chamfer L1 gap."),
    MetricDefinition("geometry.ambiguity_gap_chamfer_l2", "normalized_distance_squared", False, "True-vs-recoverable Chamfer L2 gap."),
    MetricDefinition("geometry.ambiguity_gap_volume_iou", None, False, "Recoverable-vs-true volume IoU gap."),
    MetricDefinition("geometry.sample_count_ref", "points", None, "Ground-truth surface sample count used by geometry metrics."),
    MetricDefinition("geometry.sample_count_candidate", "points", None, "Candidate surface sample count used by geometry metrics."),
    MetricDefinition("recoverability.ambiguity_gap_chamfer_l1", "normalized_distance", False, "Silhouette ambiguity gap in Chamfer L1."),
    MetricDefinition("recoverability.ambiguity_gap_chamfer_l2", "normalized_distance_squared", False, "Silhouette ambiguity gap in Chamfer L2."),
    MetricDefinition("recoverability.ambiguity_gap_volume_iou", None, False, "Silhouette ambiguity gap in volume IoU."),
    MetricDefinition("recoverability.true.fscore_tau", None, True, "Recoverability report true-geometry F-score."),
    MetricDefinition("recoverability.recoverable.fscore_tau", None, True, "Recoverability report envelope F-score."),
    MetricDefinition("novel_view.psnr", "dB", True, "Novel-view peak signal-to-noise ratio."),
    MetricDefinition("novel_view.ssim", None, True, "Novel-view structural similarity."),
    MetricDefinition("novel_view.lpips", None, False, "Novel-view learned perceptual patch distance."),
    MetricDefinition("novel_view.mse", None, False, "Novel-view mean squared pixel error."),
    MetricDefinition("novel_view.image_count", "image", None, "Novel-view image pair count."),
    MetricDefinition("topology.connected_components", "count", False, "Mesh connected component count."),
    MetricDefinition("topology.boundary_edges", "count", False, "Mesh boundary edge count."),
    MetricDefinition("topology.non_manifold_edges", "count", False, "Mesh non-manifold edge count."),
    MetricDefinition("topology.degenerate_faces", "count", False, "Mesh degenerate face count."),
    MetricDefinition("topology.loose_vertices", "count", False, "Mesh loose vertex count."),
    MetricDefinition("topology.watertight", None, True, "Whether the mesh is watertight."),
    MetricDefinition("editability.editable_reconstruction_index", None, True, "Composite Blender editability score."),
    MetricDefinition("editability.object_hierarchy_score", None, True, "Object hierarchy and part separability score."),
    MetricDefinition("editability.primitive_score", None, True, "Primitive/parametric editability score."),
    MetricDefinition("editability.modifier_score", None, True, "Non-destructive modifier usage score."),
    MetricDefinition("editability.mesh_density_score", None, True, "Mesh density suitability for editing."),
    MetricDefinition("editability.semantic_part_score", None, True, "Semantic part naming/tagging score."),
    MetricDefinition("editability.export_roundtrip_score", None, True, "Export/reimport editability preservation score."),
    MetricDefinition("appearance.report_count", "report", None, "Texture/material appearance report count."),
    MetricDefinition("appearance.has_uv_map", None, True, "Whether the asset exposes UV coordinates."),
    MetricDefinition("appearance.uv_valid", None, True, "Whether UV layout checks pass."),
    MetricDefinition("appearance.uv_island_count", "island", None, "UV island count."),
    MetricDefinition("appearance.uv_overlap_ratio", None, False, "Fraction of UV area with overlapping islands."),
    MetricDefinition("appearance.uv_out_of_bounds_ratio", None, False, "Fraction of UV coordinates outside the 0-1 tile."),
    MetricDefinition("appearance.uv_stretch_mean", None, False, "Mean UV stretch estimate."),
    MetricDefinition("appearance.texel_density_cv", None, False, "Coefficient of variation for texel density."),
    MetricDefinition("appearance.missing_uv_faces", "face", False, "Faces without UV assignment."),
    MetricDefinition("appearance.texture_resolution", "pixel", None, "Largest evaluated texture pixel area."),
    MetricDefinition("appearance.texture_file_count", "file", None, "Number of texture files referenced by the asset."),
    MetricDefinition("appearance.texture_memory_mb", "MiB", False, "Estimated texture memory footprint."),
    MetricDefinition("appearance.reprojection_psnr", "dB", True, "Appearance reprojection PSNR."),
    MetricDefinition("appearance.reprojection_ssim", None, True, "Appearance reprojection SSIM."),
    MetricDefinition("appearance.reprojection_lpips", None, False, "Appearance reprojection LPIPS."),
    MetricDefinition("appearance.seam_visibility_score", None, False, "Texture seam visibility score."),
    MetricDefinition("appearance.material_slot_count", "material", None, "Material slot count."),
    MetricDefinition("appearance.named_material_ratio", None, True, "Fraction of material slots with non-generic names."),
    MetricDefinition("appearance.pbr_channel_coverage_ratio", None, True, "Fraction of required PBR channels present."),
    MetricDefinition("appearance.duplicate_material_count", "material", False, "Duplicate or redundant material count."),
    MetricDefinition("appearance.orphan_texture_count", "texture", False, "Texture files not referenced by material slots."),
    MetricDefinition("appearance.attribution_boundary_geometry_fidelity", None, True, "Boundary fidelity attributable to geometry rather than appearance."),
    MetricDefinition("appearance.attribution_texture_only_detail_score", None, False, "Detail score that exists only in texture/image space."),
    MetricDefinition("appearance.attribution_geometry_detail_score", None, True, "Detail score represented by editable geometry."),
    MetricDefinition("appearance.image_space_hallucination_warning", None, False, "Whether appearance appears to hide missing geometry."),
    MetricDefinition("export.qa_score", None, True, "Composite export and round-trip QA score."),
    MetricDefinition("export.status_ok", None, True, "Whether every configured export target completed successfully."),
    MetricDefinition("export.reimport_ok", None, True, "Whether every checked export target reimported successfully."),
    MetricDefinition("export.target_count", "target", None, "Number of export targets checked."),
    MetricDefinition("export.object_count", "object", None, "Total exported/reimported object count."),
    MetricDefinition("export.vertex_count", "vertex", None, "Total exported/reimported vertex count."),
    MetricDefinition("export.face_count", "face", None, "Total exported/reimported face count."),
    MetricDefinition("export.material_count", "material", None, "Total exported/reimported material count."),
    MetricDefinition("cost.total_wall_ms", "ms", False, "End-to-end wall-clock time."),
    MetricDefinition("cost.time_to_first_preview_ms", "ms", False, "Time until first renderable preview."),
    MetricDefinition("cost.peak_memory_mb", "MiB", False, "Peak memory usage."),
    MetricDefinition("cost.throughput.voxels_per_ms", "voxels/ms", True, "Voxel throughput for volume-backed candidates."),
    MetricDefinition("cost.throughput.active_voxels_per_ms", "voxels/ms", True, "Active occupied-voxel throughput for volume-backed candidates."),
)


METRIC_REGISTRY: Mapping[str, MetricDefinition] = {item.name: item for item in _DEFINITIONS}


def metric_definition(name: str) -> MetricDefinition | None:
    return METRIC_REGISTRY.get(name)


def registry_payload() -> dict[str, object]:
    return {name: definition.to_dict() for name, definition in METRIC_REGISTRY.items()}
