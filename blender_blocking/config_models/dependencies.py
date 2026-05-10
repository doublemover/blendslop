from __future__ import annotations

_VALID_RECON_MODES = {
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "shape_program",
    "ensemble",
}

_VALID_JOIN_MODES = {"auto", "boolean", "voxel", "simple"}

_VALID_CAP_MODES = {"fan", "none", "ngon"}

_VALID_SAMPLE_POLICIES = {"endpoints", "cell_centers"}

_VALID_FILL_STRATEGIES = {"interp_linear", "interp_nearest", "constant"}

_VALID_CANON_ANCHORS = {"center", "bottom_center"}

_VALID_CANON_INTERP = {"nearest"}

_VALID_RENDER_ENGINES = {"BLENDER_EEVEE", "WORKBENCH"}

_VALID_COLOR_MODES = {"BW", "RGBA"}

_VALID_CONTOUR_MODES = {"external", "ccomp", "tree", "hierarchy"}

_VALID_BOOLEAN_SOLVERS = {"auto", "EXACT", "MANIFOLD", "FLOAT", "FAST"}

_VALID_INVERT_POLICIES = {"auto", "invert", "no_invert"}

_VALID_POLARITIES = {"auto", "dark_foreground", "light_foreground", "alpha_foreground"}

_VALID_MESH_METHODS = {"marching_cubes", "lewiner", "dual_contouring", "points"}

_VALID_VOLUME_BACKENDS = {"dense", "chunked", "sparse_hash", "openvdb"}

_VALID_SELECTION_POLICIES = {
    "best_score",
    "balanced",
    "fidelity",
    "quality_first",
    "editable",
    "editability_first",
    "printable",
    "fast_preview",
    "pareto",
    "research_fidelity",
}

_VALID_POSTPROCESS = {
    "none",
    "poisson",
    "screened_poisson",
    "smooth_guarded",
    "topology_repair",
}

_VALID_REFINEMENT_SEARCH = {"grid", "random", "coordinate", "successive_halving"}

_VALID_REFINEMENT_OBJECTIVES = {
    "quality_win",
    "min_view_iou",
    "mean_iou",
    "profile_editable",
    "visual_hull_alignment",
    "fast_preview",
    "human_adjusted",
}

_VALID_REFINEMENT_REPORT_FAILURES = {"top", "all", "none"}

__all__ = [
    "_VALID_BOOLEAN_SOLVERS",
    "_VALID_CANON_ANCHORS",
    "_VALID_CANON_INTERP",
    "_VALID_CAP_MODES",
    "_VALID_COLOR_MODES",
    "_VALID_CONTOUR_MODES",
    "_VALID_FILL_STRATEGIES",
    "_VALID_INVERT_POLICIES",
    "_VALID_JOIN_MODES",
    "_VALID_MESH_METHODS",
    "_VALID_POLARITIES",
    "_VALID_POSTPROCESS",
    "_VALID_RECON_MODES",
    "_VALID_REFINEMENT_OBJECTIVES",
    "_VALID_REFINEMENT_REPORT_FAILURES",
    "_VALID_REFINEMENT_SEARCH",
    "_VALID_RENDER_ENGINES",
    "_VALID_SAMPLE_POLICIES",
    "_VALID_SELECTION_POLICIES",
    "_VALID_VOLUME_BACKENDS",
]
