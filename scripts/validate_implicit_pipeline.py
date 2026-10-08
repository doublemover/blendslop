"""Three bounded host-to-pinned-worker jobs, topology change and source retention.

This is a contract fixture, not a reconstruction campaign or timing benchmark.
Run with the host image/meshing dependencies and --python pinned CPU interpreter.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking')]

from reconstruction.backends.implicit_residual import ImplicitResidualBackend
from reconstruction.implicit.helper import implicit_session, close_implicit_sessions
from reconstruction.implicit.pipeline import replay_implicit_artifacts
from reconstruction.types import CandidateRequest, CandidateMetrics
from reconstruction.projected_metrics import projected_mesh_masks, projected_mesh_metrics
from test_quality_geometry import target_for_masks
from test_implicit_pipeline import fixture, hole_fixture, separated_fixture, seed_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-separated', action='store_true', help='add two bounded disjoint-assembly jobs at 16/32')
    parser.add_argument('--artifact-root', help='retain source/proposal/field replay and original-pixel evidence for renderer-contract inspection')
    args = parser.parse_args()
    target, seed = fixture()
    backend = ImplicitResidualBackend()
    records = []
    try:
        if args.artifact_root:
            Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
        context = nullcontext(args.artifact_root) if args.artifact_root else tempfile.TemporaryDirectory(prefix='blendslop-implicit-fixture-')
        with context as root:
            jobs = 5 if args.include_separated else 3
            for index in range(jobs):
                target, seed = fixture() if index < 2 else hole_fixture() if index == 2 else separated_fixture()
                if index < 2:
                    source_masks = projected_mesh_masks(target, seed.vertices, seed.faces)
                    target = target_for_masks(source_masks)
                source_result = seed_result(target, seed)
                source_result = replace(source_result, metric_result=CandidateMetrics(
                    per_view=projected_mesh_metrics(target, seed.vertices, seed.faces),
                    extras=source_result.metric_result.extras))
                assert all(row['passed'] for row in source_result.metric_result.per_view.values())
                print(f'implicit integration job={index+1}/{jobs} stage=actual owned pinned helper', flush=True)
                config = {'execution_approved': True, 'helper_python': args.python,
                          'seed_results': {'source': source_result}, 'steps': 1,
                          'timeout_s': 20., 'boundary_timeout_s': 8.}
                if index == 2:
                    config.update(resolution=32, steps=0)
                elif index >= 3:
                    config.update(resolution=16 if index == 3 else 32)
                request = CandidateRequest('fixture-'+str(index), backend.name, target,
                                           config=config, artifact_root=Path(root))
                result = backend.reconstruct(request)
                if result.status != 'degraded' or result.geometry.content_hash != seed.content_hash:
                    raise AssertionError(('unqualified seed retention failed', result.to_dict()))
                report = result.metric_result.extras['implicit_residual']
                replay = replay_implicit_artifacts(report, include_output_checkpoints=True)
                if replay['retained'].content_hash != seed.content_hash:
                    raise AssertionError('authoritative source replay failed')
                scored = [row for row in report['output_checkpoint_selection']['checkpoints']
                          if row.get('status') == 'original_camera_opaque_scored']
                assert [row['geometry'].content_hash for row in replay['output_checkpoints']] == [
                    row['geometry_hash'] for row in scored]
                optimizer = report['optimization']
                if optimizer['objective_evaluations'] != (1 if index == 2 else 2) or optimizer['optimizer_updates'] != (0 if index == 2 else 1):
                    raise AssertionError('bounded numerical work differs')
                if index == 2:
                    assert report['proposal_topology']['euler_characteristic'] == 0
                    assert report['opaque_proposal_per_view']['top']['area_iou'] < report['opaque_seed_per_view']['top']['area_iou']
                    assert not report['full_geometry_admitted']
                if index >= 3:
                    assert report['source_screen']['connected_components'] == 2
                    assert report['proposal_topology']['connected_components'] == 2
                    assert report['exact_source_boundary']['passed']
                    assert report['proposal_exact_boundary']['passed']
                if args.artifact_root:
                    from reconstruction.pixel_evidence import observed_pixel_evidence
                    from reconstruction.pixel_projection import canonical_mesh_projections
                    source_legacy=projected_mesh_masks(target,seed.vertices,seed.faces)
                    proposal=replay['proposal']
                    proposal_legacy=projected_mesh_masks(target,proposal.vertices,proposal.faces)
                    source_area=canonical_mesh_projections(target,seed.vertices,seed.faces)
                    proposal_area=canonical_mesh_projections(target,proposal.vertices,proposal.faces)
                    arrays={}
                    for constraint in target.constraints:
                        evidence=observed_pixel_evidence(constraint);view=constraint.view
                        arrays.update({view+'_foreground_probability':evidence.foreground,
                            view+'_valid':evidence.valid,view+'_weights':evidence.weights,
                            view+'_source_legacy':source_legacy[view],view+'_proposal_legacy':proposal_legacy[view],
                            view+'_source_area':source_area[view],view+'_proposal_area':proposal_area[view]})
                    np.savez_compressed(Path(root)/('job-'+str(index)+'-original-pixels.npz'),**arrays)
                records.append({'job': index, 'topology': report['proposal_topology'],
                                'opaque_proposal_per_view': report['opaque_proposal_per_view'],
                                'opaque_seed_per_view': report['opaque_seed_per_view'], 'helper_session': optimizer['helper_session'],
                                'best_evaluation': optimizer['best_evaluation'],
                                'objective_evaluations': optimizer['objective_evaluations'],
                                'output_source_hash': result.geometry.content_hash,
                                'field_replay_proposal_hash': replay['proposal'].content_hash,
                                'exact_proposal_boundary': report['proposal_exact_boundary']['passed'],
                                'native_qualification_status': report['proposal_native_qualification']['status'],
                                'source_component_count': report['source_screen']['connected_components'],
                                'proposal_admitted': report['full_geometry_admitted']})
        assert records[0]['helper_session']['pid'] == records[1]['helper_session']['pid']
        assert all(r['helper_session']['starts'] == 1 for r in records)
        assert records[2]['helper_session']['pid'] == records[0]['helper_session']['pid']
        assert all(r['helper_session']['pid'] == records[0]['helper_session']['pid'] for r in records)
        packet = {'passed': True, 'records': records, 'scope': 'bounded source-to-helper-to-field-to-mesh contract fixtures; one confirmed-empty through-hole; optional separated assemblies',
                  'native_blender_render': 'not_run', 'quality_and_speed_gains': 'unmeasured'}
        Path(args.output).write_text(json.dumps(packet, indent=2)+'\n')
        print(json.dumps(packet), flush=True)
    finally:
        close_implicit_sessions()


if __name__ == '__main__':
    main()
