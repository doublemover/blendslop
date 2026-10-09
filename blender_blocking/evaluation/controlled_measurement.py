"""Controlled occupancy is linear alpha; display RGB is never measurement."""
from __future__ import annotations

import hashlib
import json
import numpy as np

PROTOCOL = 'opaque_linear_alpha_v1'


def coverage_and_mask(alpha):
    """Keep filtered AA coverage and derive a fixed half-coverage hard mask."""
    coverage = np.asarray(alpha, dtype=np.float32)
    if (coverage.ndim != 2 or not coverage.size or not np.isfinite(coverage).all()
            or coverage.min() < -1e-6 or coverage.max() > 1. + 1e-6):
        raise ValueError('controlled alpha must be a finite nonempty image in [0, 1]')
    # Only remove floating-point endpoint noise, never dilate/blur a boundary.
    coverage = np.clip(coverage, 0., 1.)
    return coverage, coverage >= .5


def measurement_signature(contract):
    if contract.get('protocol') != PROTOCOL or contract.get('hard_mask_threshold') != .5:
        raise ValueError('unknown controlled occupancy contract')
    required = ('camera', 'renderer', 'sampling', 'pixel_convention', 'surface_policy', 'encoding')
    if any(key not in contract for key in required):
        raise ValueError('incomplete controlled measurement contract')
    from .canonical_artifacts import camera_frame_sha256
    camera = contract['camera']
    camera_frame_sha256(camera)
    clips = np.asarray([camera.get('clip_start'), camera.get('clip_end')], float)
    sampling = contract['sampling']
    samples, width = sampling.get('taa_render_samples'), sampling.get('filter_size')
    if (not np.isfinite(clips).all() or not 0 < clips[0] < clips[1]
            or isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 256
            or not isinstance(width, (int, float)) or not np.isfinite(width) or not .01 <= width <= 2.
            or contract['renderer'].get('engine') != 'BLENDER_EEVEE'):
        raise ValueError('invalid controlled camera clipping/renderer/sampling')
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def compare_controlled_measurements(reference, candidate, *, view, config):
    """Require matched acquisition before applying the existing strict gates.

    This is a new protocol. Historical display-PNG verdicts are not replaced.
    Geometry, surface and editability verdicts remain separate.
    """
    from .silhouette_eval import evaluate_silhouette_pair
    if measurement_signature(reference['contract']) != measurement_signature(candidate['contract']):
        raise ValueError('source/candidate measurement contracts differ')
    ref, ref_mask = coverage_and_mask(reference['coverage'])
    cand, cand_mask = coverage_and_mask(candidate['coverage'])
    camera_size = tuple(reference['contract']['camera']['resolution'])
    if ref.shape[::-1] != camera_size or ref.shape != cand.shape:
        raise ValueError('source/candidate measurement resolutions differ')
    metrics = evaluate_silhouette_pair(ref_mask, cand_mask, view=view, config=config)
    metrics['measurement_protocol'] = PROTOCOL
    metrics['coverage_l1'] = float(np.mean(np.abs(ref-cand)))
    metrics['coverage_l1_scope'] = 'diagnostic only; no new acceptance cutoff'
    return metrics
