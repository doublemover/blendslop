"""Explicit quality preset: defaults retain their bounded original budgets."""

def quality_config(config):
    config = dict(config)
    if config.get('quality_preset') != 'quality': return config
    for key, value in {'program_search_candidates': 24, 'program_refinement_steps': 2,
        'program_refinement_trials': 12, 'cuboid_search': True, 'convex_proxy_search': True,
        'structural_search': True, 'subtractive_search': True, 'adaptive_sections': True,
        'max_sections': 64, 'contour_sections': True, 'adaptive_hull': True,
        'preserve_thin_features': True, 'max_residual_proposals': 8, 'residual_rounds': 2,
        'max_primitives': 16, 'max_objective_evaluations': 768, 'max_multistart_attempts': 4,
        'finest_finishing': True, 'native_batch_queries': True,
        'native_union_execution': True, 'native_union_solver': 'MANIFOLD',
        'native_sdf_fallback': True}.items():
        config[key] = value
    for key, value in {'whole_support_search': True,'generalized_sweep_search':True, 'planar_extrusion_search': True, 'retessellate_coplanar': True, 'objective_mode': 'normalized_area_v1',
                       'routing_policy': 'structure_v1', 'hybrid_mode': 'structure_first',
                       'refinement_strategy': 'coupled_blocks', 'proxy_variant': 'fitted_opaque_union_v1',
                       'pixel_evidence_mode':'pixel_reliability_v1',
                       'dvx_objective': 'observed_projected_rays',
                       'dvx_parameterization': 'cage', 'dvx_warm_helper': True,
                       'dvx_grid_levels': [n for n in (16,32,64) if n<=int(config.get('dvx_resolution',32))]}.items():
        if config.get(key) is None:
            config[key] = value
    return config
