"""Roof-only arch refinement from partial observed coverage contours.

This follow-up keeps the selected exterior, opening, extrusion and rigid frame
fixed. The shared private solver is used only by this stage and the separate
three-control rounded-triangle follow-up; it never consumes reference meshes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import numbers
import time

import numpy as np

from .adaptive_family import (
    _json_hash, _model_camera, family_control_values, family_program_update,
)
from .adaptive_arch_exterior import arch_exterior_signed_distance, _geometry
from .frozen_family import coverage_contour_points

CONTROL = 'notch_height_from_bottom_world'


def arch_roof_controls(wire):
    _geometry(wire)
    value = family_control_values(wire, 'concave_arch')['cavity_roof_height_world']
    return {CONTROL: float(value)}


def arch_roof_update(wire, values):
    if not values or set(values) != {CONTROL}:
        raise ValueError('roof-only stage requires its one declared notch-height control')
    return family_program_update(wire, 'concave_arch',
        {'cavity_roof_height_world': values[CONTROL]})


def _scores(residual):
    absolute = np.abs(residual)
    return {'mean_absolute_source_pixels': float(absolute.mean()),
            'p95_absolute_source_pixels': float(np.quantile(absolute, .95)),
            'maximum_absolute_source_pixels': float(absolute.max())}


def _bounded_tail_update(wire, observations, *, controls, update, distance,
                         parameter_bounds, heldout_views, prior_view_exposure,
                         baseline_geometry_hash, baseline_program_sha256,
                         max_evaluations, max_elapsed_s, protocol, metadata_key):
    """Small bounded observed L4 search with conservative L1/P95/max retention."""
    from scipy.optimize import least_squares
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, numbers.Integral)
            or not 8 <= max_evaluations <= 96 or isinstance(max_elapsed_s, bool)
            or not isinstance(max_elapsed_s, numbers.Real) or not math.isfinite(max_elapsed_s)
            or not 0 < max_elapsed_s <= 1.):
        raise ValueError('tail update allows 8..96 residual calls and >0..1 second')
    for identity in (baseline_geometry_hash, baseline_program_sha256):
        if (not isinstance(identity, str) or len(identity) != 64
                or any(c not in '0123456789abcdef' for c in identity)):
            raise ValueError('tail stage requires exact frozen baseline geometry and recipe SHA256')
    if (not isinstance(parameter_bounds, dict) or set(parameter_bounds) != set(controls)
            or not 1 <= len(controls) <= 3):
        raise ValueError('tail stage requires intervals for exactly its declared controls')
    if (not isinstance(prior_view_exposure, dict)
            or any(view not in prior_view_exposure for view in heldout_views)):
        raise ValueError('held-out exposure history must remain explicit')
    if (not isinstance(observations, dict) or not observations
            or set(observations) & set(heldout_views)):
        raise ValueError('reserved held-out observations cannot enter tail fitting')
    names = tuple(controls)
    intervals = np.asarray([parameter_bounds[name] for name in names], float)
    initial = np.asarray([controls[name] for name in names], float)
    if (intervals.shape != (len(names), 2) or not np.isfinite(intervals).all()
            or (intervals[:, 1] <= intervals[:, 0]).any()
            or (initial < intervals[:, 0]).any() or (initial > intervals[:, 1]).any()):
        raise ValueError('finite increasing intervals must include the retained control')
    evidence, records = [], {}
    for view, record in sorted(observations.items()):
        coverage = np.asarray(record['coverage'], float)
        if (coverage.ndim != 2 or not np.isfinite(coverage).all()
                or (coverage < 0.).any() or (coverage > 1.).any()):
            raise ValueError('observed coverage must be finite linear alpha in [0, 1]')
        _, scale = _model_camera(record['camera'], coverage.shape)
        pixels = coverage_contour_points(coverage, allow_clipped=True)
        count = len(pixels)
        pixels = pixels[np.linspace(0, count-1, min(count, 384), dtype=int)]
        height, width = coverage.shape
        points = np.column_stack(((pixels[:, 0]/width-.5)*scale,
                                  (.5-pixels[:, 1]/height)*scale))
        # Missing outside-frame contours are never closed or synthesized.
        touched = bool((coverage[0] >= .5).any() or (coverage[-1] >= .5).any()
                       or (coverage[:, 0] >= .5).any() or (coverage[:, -1] >= .5).any())
        evidence.append((points, record['camera'], scale/width))
        records[view] = {'camera_sha256': _json_hash(record['camera']),
            'coverage_sha256': hashlib.sha256(coverage.astype('<f8').tobytes()).hexdigest(),
            'observed_crossings': count, 'fitted_crossings': len(points),
            'frame_censored': touched, 'complete_contour_claim': not touched,
            'frame_border_closure_added': False}
    spans = intervals[:, 1]-intervals[:, 0]
    x0 = (initial-intervals[:, 0])/spans
    def decode(x):
        if np.array_equal(x, x0):
            return dict(controls)
        return {name: float(value) for name, value in zip(names, intervals[:, 0]+x*spans)}
    # Validate endpoints using the real model before the bounded solver starts.
    for axis in range(len(names)):
        for endpoint in (0., 1.):
            vector = x0.copy(); vector[axis] = endpoint
            update(wire, decode(vector))
    calls, best, termination = 0, x0.copy(), 'solver_completed'
    best_residual = None
    started = time.perf_counter()
    baseline_scores = None
    best_key = None
    class AllowanceEnded(Exception):
        pass
    def residual(x, *, retain=True, fit=True):
        nonlocal calls, best, best_key, baseline_scores, termination, best_residual
        cap = max_evaluations-2*len(names) if fit else max_evaluations
        # Reserve the final 20% of the same wall allowance for rank evidence.
        phase_seconds = max_elapsed_s*.8 if fit else max_elapsed_s
        if calls >= cap or (calls and time.perf_counter()-started >= phase_seconds):
            termination = 'evaluation_allowance' if calls >= cap else 'elapsed_allowance'
            raise AllowanceEnded
        calls += 1
        updated = update(wire, decode(x)).to_dict()
        parts = [distance(points, updated, camera)/pixel_step
                 for points, camera, pixel_step in evidence]
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise ValueError('projected contour residual is unavailable')
        scores = _scores(result)
        key = (scores['maximum_absolute_source_pixels'],
               scores['p95_absolute_source_pixels'], scores['mean_absolute_source_pixels'])
        if baseline_scores is None:
            baseline_scores, best_key, best_residual = scores, key, result.copy()
        # Numerical slack only; this is not a reconstruction/artist cutoff.
        slack = 64*np.finfo(float).eps*max(1., *baseline_scores.values())
        conservative = all(scores[k] <= baseline_scores[k]+slack for k in scores)
        if retain and conservative and key < best_key:
            best, best_key, best_residual = x.copy(), key, result.copy()
        return result
    initial_residual = residual(x0)
    def l4_objective(x):
        raw = residual(x)
        # Equal per-view weight, source-pixel units; transform yields L4 cost.
        blocks, index = [], 0
        for points, _, _ in evidence:
            part = raw[index:index+len(points)]; index += len(points)
            blocks.append(part*np.abs(part)/math.sqrt(len(part)))
        return np.concatenate(blocks)
    try:
        least_squares(l4_objective, x0, bounds=(np.zeros(len(names)), np.ones(len(names))),
                      diff_step=1e-4, max_nfev=max_evaluations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    except AllowanceEnded:
        pass
    proposed = best.copy(); columns = []
    try:
        for axis in range(len(names)):
            lower, upper = proposed.copy(), proposed.copy()
            lower[axis] = max(0., lower[axis]-1e-3)
            upper[axis] = min(1., upper[axis]+1e-3)
            columns.append((residual(upper, retain=False, fit=False)
                            -residual(lower, retain=False, fit=False))/(upper[axis]-lower[axis]))
    except AllowanceEnded:
        columns = []
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False) if columns else np.array([])
    rank = int(np.count_nonzero(singular > max(1e-8, singular[0]*1e-5))) if len(singular) else None
    bound_active = bool((np.minimum(proposed, 1.-proposed) <= 1e-6).any())
    improved = bool(best_key < (baseline_scores['maximum_absolute_source_pixels'],
        baseline_scores['p95_absolute_source_pixels'], baseline_scores['mean_absolute_source_pixels']))
    admitted = improved and rank == len(names) and not bound_active
    selected = proposed if admitted else x0
    updated = update(wire, decode(selected))
    # Score the already-selected model without any additional solver calls.
    selected_residual = best_residual if admitted else initial_residual
    detail = {'protocol': protocol, 'free_controls': list(names),
        'original_controls': controls, 'proposed_controls': decode(proposed),
        'selected_controls': decode(selected), 'frozen_intervals': intervals.tolist(),
        'objective': 'equal-view observed contour L4 in source-pixel units',
        'retention': 'mean absolute, P95 and maximum observed contour errors all nonworse; full local rank and no active bound',
        'baseline_observed_tail': baseline_scores, 'selected_observed_tail': _scores(selected_residual),
        'observed_tail_admitted': admitted, 'residual_calls': calls,
        'elapsed_seconds': time.perf_counter()-started, 'termination': termination,
        'fit_time_share': .8, 'rank_time_reserved_share': .2,
        'observations': records, 'local_rank': rank,
        'local_sensitivity_singular_values': singular.tolist(), 'interval_bound_active': bound_active,
        'heldout_views': list(heldout_views), 'prior_view_exposure': deepcopy(prior_view_exposure),
        'heldout_fit_or_roi_used': False, 'global_uniqueness_established': False,
        'identifiability': 'locally_identified' if rank == len(names) and not bound_active else 'underconstrained',
        'baseline_geometry_hash': baseline_geometry_hash,
        'baseline_program_file_sha256': baseline_program_sha256,
        'native_silhouette': 'unrun', 'surface_tail': 'unrun', 'native_boundary': 'unrun',
        'artist_limits': None, 'aggregate_accepted': False}
    metadata = deepcopy(dict(updated.metadata))
    history = list(metadata.get('historical_adaptive_stages', []))
    for key in ('adaptive_detail', 'adaptive_arch_exterior'):
        if key in metadata:
            history.append({'source_metadata_key': key, 'record': metadata.pop(key),
                'baseline_geometry_hash': baseline_geometry_hash,
                'baseline_program_file_sha256': baseline_program_sha256,
                'observations_apply_to_current_geometry': False})
    if history: metadata['historical_adaptive_stages'] = history
    metadata[metadata_key] = detail
    return replace(updated, metadata=metadata)


def refine_arch_roof(wire, observations, *, parameter_bounds, heldout_views,
                     prior_view_exposure, baseline_geometry_hash,
                     baseline_program_sha256, max_evaluations=96, max_elapsed_s=1.):
    return _bounded_tail_update(wire, observations, controls=arch_roof_controls(wire),
        update=arch_roof_update, distance=arch_exterior_signed_distance,
        parameter_bounds=parameter_bounds, heldout_views=heldout_views,
        prior_view_exposure=prior_view_exposure, baseline_geometry_hash=baseline_geometry_hash,
        baseline_program_sha256=baseline_program_sha256, max_evaluations=max_evaluations,
        max_elapsed_s=max_elapsed_s, protocol='bounded_arch_roof_tail_v1',
        metadata_key='arch_roof_tail')
