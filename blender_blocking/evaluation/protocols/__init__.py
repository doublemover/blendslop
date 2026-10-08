"""Versioned metric contracts. Legacy evaluation remains in comparable_geometry."""
from .core import PROFILES, shared_transform, evaluate_points, macro_cases
from .bundles import evaluate_mesh_profile, save_sample_bundle, reload_sample_bundle
from .dtu import DTU_MODES, dtu_directed_metrics, dtu_native_adapter
from .dtu_preparation import prepare_dtu_prediction
