"""Read-only additional-view requests from same-evidence retained hypotheses.

Rendered candidate guesses are planning diagnostics, never new observations.
No target, selector, training/evaluation label or artist scene is modified.
"""
from dataclasses import dataclass, replace
import itertools
import time

import numpy as np

from .planning_camera import PlanningOrthographicCamera


@dataclass(frozen=True)
class CaptureDirection:
    name: str
    azimuth_deg: float
    elevation_deg: float
    access_weight: float = 1.
    visible_mask: object = None

    def __post_init__(self):
        values = np.asarray([self.azimuth_deg, self.elevation_deg, self.access_weight], float)
        if (not isinstance(self.name, str) or not self.name.strip() or not np.isfinite(values).all()
                or not -90 <= values[1] <= 90 or not 0 <= values[2] <= 1):
            raise ValueError('capture direction requires a name, finite angles and access weight 0..1')
        for field, value in zip(('azimuth_deg', 'elevation_deg', 'access_weight'), values):
            object.__setattr__(self, field, float(value))


def default_capture_directions():
    return tuple(CaptureDirection(f'oblique_az{azimuth}_el{elevation}', azimuth, elevation)
                 for elevation in (20, 40) for azimuth in (-45, 45, 135, 225))


def filtered_candidate_projection(data, camera):
    """Opaque triangle-union area inside each whole original pixel cell."""
    from shapely import Polygon, area, box, intersection, union_all

    if len(data.faces) > 4096:
        raise ValueError('planning triangle allowance exceeded')
    xy, _ = camera.project_world(data.vertices)
    polygons = []
    for face in data.faces:
        points = xy[face]
        first, second = points[1]-points[0], points[2]-points[0]
        if first[0]*second[1]-first[1]*second[0] != 0:
            polygons.append(Polygon(points))
    width, height = camera.resolution
    if width*height > 16384:
        raise ValueError('planning pixel allowance exceeded')
    if not polygons:
        return np.zeros((height, width))
    shape = union_all(polygons)
    if shape.is_empty or not shape.is_valid:
        raise ValueError('planning projected triangle union is invalid')
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    cells = box(xx.ravel()-.5, yy.ravel()-.5, xx.ravel()+.5, yy.ravel()+.5)
    # The area operator does not amplify machine-scale projected-union holes
    # into artificial signed-distance boundaries. No source mesh is rounded.
    mask = np.asarray(area(intersection(cells, shape)), float).reshape(height, width)
    if not np.isfinite(mask).all() or np.any(mask < -1e-12) or np.any(mask > 1+1e-12):
        raise ValueError('planning pixel-cell coverage is invalid')
    return np.clip(mask, 0, 1)


def _valid_observed_gate(row, minimum_iou):
    passed = row.get('passed', False)
    if not isinstance(passed, (bool, np.bool_)) or not passed:
        return False
    try:
        iou = float(row.get('area_iou', -1))
    except (ValueError, TypeError):
        return False
    return bool(np.isfinite(iou) and minimum_iou <= iou <= 1.)


def _eligible_hypotheses(target, results, *, minimum_observed_iou, maximum_candidates):
    from .evidence_identity import target_evidence_hash
    from .native_geometry import geometry_arrays
    from .visibility import valid_evidence

    identity = target_evidence_hash(target)
    required_views = {constraint.view for constraint in target.constraints
                      if valid_evidence(constraint).any()}
    accepted, excluded, seen = [], [], set()
    entries = dict(results or {})
    if len(entries) > 64:
        raise ValueError('planning retained-candidate input allowance exceeded')
    for name, result in entries.items():
        reason = None
        if not result.succeeded or result.geometry is None:
            reason = 'no_successful_retained_geometry'
        elif result.metric_result.extras.get('input_evidence_hash') != identity:
            reason = 'same_fixed_evidence_not_established'
        else:
            rows = result.metric_result.per_view
            if not required_views:
                reason = 'no_observed_pixels'
            else:
                passed = all(view in rows and _valid_observed_gate(rows[view], minimum_observed_iou)
                             for view in required_views)
                if not passed:
                    reason = 'required_observed_view_gate_or_fit_failed'
        if reason is None:
            data = geometry_arrays(result.geometry)
            if len(data.faces) > 4096:
                reason = 'planning_triangle_allowance'
            elif data.content_hash in seen:
                reason = 'duplicate_retained_geometry'
            else:
                seen.add(data.content_hash)
                observed_quality = min(float(rows[view]['area_iou']) for view in required_views)
                accepted.append((observed_quality, str(name), result, data))
        if reason is not None:
            excluded.append({'candidate': str(name), 'reason': reason})
    accepted.sort(key=lambda row: (-row[0], row[1]))
    for row in accepted[maximum_candidates:]:
        excluded.append({'candidate': row[1], 'reason': 'planning_candidate_allowance'})
    return accepted[:maximum_candidates], excluded, identity


def suggest_additional_view(target, results, *, directions=None, held_out_directions=(),
                            resolution=(32, 32), minimum_observed_iou=.85,
                            minimum_disagreement=.08, maximum_candidates=4,
                            capture_available=False, max_elapsed_s=8., exclude_angle_deg=5.):
    """Return an optional capture request without modifying current evidence.

    Access/visibility inputs describe possible capture conditions, not observed
    silhouette pixels. Calibrated oblique evidence admission remains a separate
    integration requirement. A reserved evaluation camera is not reused here.
    """
    started = time.perf_counter()
    if (not np.isfinite([minimum_observed_iou, minimum_disagreement, max_elapsed_s,
                         exclude_angle_deg]).all()
            or not 0 <= minimum_observed_iou <= 1 or not 0 <= minimum_disagreement <= 1
            or max_elapsed_s <= 0 or not 0 <= exclude_angle_deg < 90
            or not isinstance(maximum_candidates, int) or not 2 <= maximum_candidates <= 4):
        raise ValueError('invalid bounded view-planning configuration')
    if not isinstance(capture_available, bool):
        raise ValueError('real capture availability must be explicitly Boolean')
    directions = tuple(default_capture_directions() if directions is None else directions)
    if len(directions) > 12 or len({item.name for item in directions}) != len(directions):
        raise ValueError('planning requires at most 12 uniquely named capture directions')
    accepted, excluded, identity = _eligible_hypotheses(
        target, results, minimum_observed_iou=minimum_observed_iou,
        maximum_candidates=maximum_candidates)
    report = {'status': 'insufficient_retained_hypotheses', 'request': None,
              'input_evidence_hash': identity, 'eligible_candidate_ids': [row[2].candidate_id for row in accepted],
              'excluded_candidates': excluded, 'ranked_directions': [], 'excluded_directions': [],
              'planning_only': True, 'observations_added': 0,
              'real_capture_available': capture_available,
              'access_weight_scope': 'caller-supplied practical access; physical availability not verified',
              'automatic_selection_changed': False,
              'requires_real_calibrated_capture': True,
              'requires_additional_input_track_and_distinct_held_out_view': True,
              'qualification_scope': 'retained observed-view gates; no new native-solid or oblique-render qualification'}
    if len(accepted) < 2:
        return report
    origin = np.asarray(target.bounds.center, float)
    points = np.concatenate([row[3].vertices for row in accepted])
    canonical_backwards = {'front': np.array([0., -1., 0.]),
                           'side': np.array([1., 0., 0.]), 'top': np.array([0., 0., 1.])}
    forbidden = []
    from .visibility import valid_evidence
    for constraint in target.constraints:
        if constraint.view not in canonical_backwards:
            raise ValueError('existing observation camera is not a supported canonical contract')
        if valid_evidence(constraint).any():
            forbidden.append(('already_observed', canonical_backwards[constraint.view]))
    for item in held_out_directions:
        camera = PlanningOrthographicCamera.from_direction(
            item.name, azimuth_deg=item.azimuth_deg, elevation_deg=item.elevation_deg,
            origin=origin, resolution=resolution)
        forbidden.append(('reserved_evaluation_camera', np.asarray(camera.backward)))
    cosine = np.cos(np.deg2rad(exclude_angle_deg))
    completed = 0
    for index, direction in enumerate(directions):
        print(f'view planning direction={index+1}/{len(directions)} candidates={len(accepted)} '
              f'elapsed={time.perf_counter()-started:.2f}s', flush=True)
        if time.perf_counter()-started >= max_elapsed_s:
            report['status'] = 'partial_planning_allowance'
            break
        camera = PlanningOrthographicCamera.from_direction(
            direction.name, azimuth_deg=direction.azimuth_deg, elevation_deg=direction.elevation_deg,
            origin=origin, resolution=resolution)
        blocked = next((reason for reason, axis in forbidden
                        if abs(np.dot(camera.backward, axis)) >= cosine), None)
        if blocked is not None or direction.access_weight == 0:
            report['excluded_directions'].append({'direction': direction.name,
                                                  'reason': blocked or 'camera_access_unavailable'})
            continue
        local = (points-origin)@camera.frame
        lo, hi = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        padding = np.maximum(hi-lo, 1e-9)*.05
        camera = replace(camera, viewport=(lo[0]-padding[0], hi[0]+padding[0],
                                           lo[1]-padding[1], hi[1]+padding[1]))
        visible = np.ones(camera.resolution[::-1], bool) if direction.visible_mask is None else np.asarray(direction.visible_mask)
        if visible.dtype != np.dtype(bool):
            raise ValueError('planning visible region must be an explicit Boolean mask')
        if visible.shape != camera.resolution[::-1]:
            raise ValueError('planning visible region must match capture resolution')
        if not visible.any():
            report['excluded_directions'].append({'direction': direction.name, 'reason': 'no_visible_capture_region'})
            continue
        masks = []
        for row in accepted:
            if time.perf_counter()-started >= max_elapsed_s:
                break
            masks.append(filtered_candidate_projection(row[3], camera))
        if len(masks) != len(accepted):
            report['status'] = 'partial_planning_allowance'
            break
        disagreements = []
        heat = np.zeros(camera.resolution[::-1])
        for first, second in itertools.combinations(masks, 2):
            difference = np.abs(first-second)*visible
            union = float(np.sum(np.maximum(first, second)*visible))
            disagreements.append(float(difference.sum()/union) if union > 0 else 0.)
            heat = np.maximum(heat, difference)
        mean = float(np.mean(disagreements))
        pixel_ids = np.argwhere(heat >= max(.05, float(heat.max())*.5)) if heat.any() else np.empty((0, 2), int)
        region = None if not len(pixel_ids) else [int(pixel_ids[:, 1].min()), int(pixel_ids[:, 0].min()),
                                               int(pixel_ids[:, 1].max()+1), int(pixel_ids[:, 0].max()+1)]
        row = {'direction': direction.name, 'azimuth_deg': direction.azimuth_deg,
               'elevation_deg': direction.elevation_deg, 'access_weight': direction.access_weight,
               'mean_pairwise_disagreement': mean, 'maximum_pairwise_disagreement': max(disagreements),
               'ranking_score': mean*direction.access_weight,
               'visible_capture_pixels': int(visible.sum()), 'disagreement_region_px': region,
               'camera': camera.to_dict(), 'projection_operator': 'opaque_triangle_union_pixel_cell_area_v1'}
        report['ranked_directions'].append(row)
        completed += 1
    report['ranked_directions'].sort(key=lambda row: (-row['ranking_score'], row['direction']))
    if report['ranked_directions']:
        winner = report['ranked_directions'][0]
        if winner['ranking_score'] > 0 and winner['ranking_score'] >= minimum_disagreement and capture_available:
            report['request'] = {**winner, 'reason': 'retained_hypotheses_disagree_from_requested_direction',
                                 'oblique_input_supported_by_current_fitters': False,
                                 'action_ready': False,
                                 'pending_requirement': 'general-camera evidence admission before an oblique observation can be fitted'}
            report['status'] = 'partial_capture_suggested' if report['status'] == 'partial_planning_allowance' else 'capture_suggested'
        elif report['status'] != 'partial_planning_allowance':
            report['status'] = ('capture_unavailable_diagnostic_only' if not capture_available
                                else 'no_material_projected_disagreement')
    elif report['status'] != 'partial_planning_allowance':
        report['status'] = 'no_accessible_new_direction'
    report['elapsed_s'] = time.perf_counter()-started
    report['directions_completed'] = completed
    return report
