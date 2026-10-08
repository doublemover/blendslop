"""One existing thin-hole fixture through the actual pinned helper and replay.

The legacy reference is generated from the original source projection. This is
an operator/selection counterexample, not held-out reconstruction or timing data.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking')]

from reconstruction.backends.implicit_residual import ImplicitResidualBackend
from reconstruction.implicit.helper import close_implicit_sessions
from reconstruction.implicit.pipeline import replay_implicit_artifacts
from reconstruction.types import CandidateRequest
from test_implicit_pipeline import hole_fixture, seed_result


def retain_original_pixel_evidence(target, source, selected, surrogate, root):
    from reconstruction.evidence_identity import target_evidence_hash
    from reconstruction.pixel_evidence import observed_pixel_evidence
    from reconstruction.projected_metrics import projected_mesh_masks
    from reconstruction.pixel_projection import canonical_mesh_projections
    arrays = {}; cameras = {}
    for constraint in target.constraints:
        evidence = observed_pixel_evidence(constraint)
        for name in ('foreground', 'valid', 'weights', 'hard_mask'):
            arrays[constraint.view+'_'+name] = getattr(evidence, name)
        cameras[constraint.view] = constraint.camera.to_dict()
    for name, geometry in (('source', source), ('selected', selected), ('surrogate_best', surrogate)):
        for view, mask in projected_mesh_masks(target, geometry.vertices, geometry.faces).items():
            arrays[name+'_'+view+'_legacy_mask'] = mask
        for view, values in canonical_mesh_projections(target, geometry.vertices, geometry.faces).items():
            arrays[name+'_'+view+'_continuous_box_coverage'] = values
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root/'original-pixel-evidence.npz', **arrays)
    (root/'camera-contract.json').write_text(json.dumps({'cameras': cameras,
        'input_evidence_hash': target_evidence_hash(target),
        'geometry_hashes': {'source':source.content_hash,'selected':selected.content_hash,'surrogate_best':surrogate.content_hash},
        'reference_origin': 'generated unchanged legacy source projection with an explicit observed central hole',
        'native_renderer_filter': 'unverified', 'box_coverage': 'independent continuous diagnostic; public metric unchanged'},indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-root', required=True)
    args = parser.parse_args()
    target, seed = hole_fixture()
    backend = ImplicitResidualBackend()
    request = CandidateRequest('existing-thin-hole-checkpoints', backend.name, target,
        config={'execution_approved': True, 'helper_python': args.python,
                'seed_results': {'source': seed_result(target, seed)}, 'steps': 16, 'resolution': 32,
                'timeout_s': 20., 'checkpoint_selection_timeout_s': 8., 'boundary_timeout_s': 8.},
        artifact_root=Path(args.artifact_root))
    try:
        result = backend.reconstruct(request)
        assert result.status == 'degraded', result.to_dict()
        assert result.geometry.content_hash == seed.content_hash
        report = result.metric_result.extras['implicit_residual']
        selection = report['output_checkpoint_selection']
        replay = replay_implicit_artifacts(report, include_output_checkpoints=True)
        scored = [row for row in selection['checkpoints'] if row['status'] == 'original_camera_opaque_scored']
        assert len(scored) == 17
        assert [row['geometry'].content_hash for row in replay['output_checkpoints']] == [row['geometry_hash'] for row in scored]
        selected = next(row for row in scored if row['evaluation'] == selection['selected_evaluation'])
        surrogate = next(row for row in scored if row['evaluation'] == selection['surrogate_best_evaluation'])
        selected_iou = selected['per_view']['top']['area_iou']
        surrogate_iou = surrogate['per_view']['top']['area_iou']
        assert selected_iou >= surrogate_iou
        geometries = {row['evaluation']:row['geometry'] for row in replay['output_checkpoints']}
        retain_original_pixel_evidence(target, seed, geometries[selected['evaluation']],
                                       geometries[surrogate['evaluation']], args.artifact_root)
        packet = {'passed': True, 'selected_evaluation': selected['evaluation'],
                  'surrogate_best_evaluation': surrogate['evaluation'],
                  'selected_legacy_iou': selected_iou, 'surrogate_best_legacy_iou': surrogate_iou,
                  'source_legacy_iou': selection['source_per_view']['top']['area_iou'],
                  'full_pool_geometry_replay': True, 'finite_scored_fields': len(scored),
                  'source_retained': True, 'resolution': 32, 'optimizer_updates_requested': 16,
                  'native_render': 'not_run',
                  'metric_identity': selection['metric_identity'],
                  'input_evidence_hash': report['input_evidence_hash'],
                  'scope': 'existing generated legacy-reference thin-hole implementation fixture; no quality/timing or held-out claim'}
        Path(args.output).write_text(json.dumps(packet, indent=2)+'\n')
        print(json.dumps(packet), flush=True)
    finally:
        close_implicit_sessions()


if __name__ == '__main__':
    main()
