"""Pinned numeric fixtures for formula compatibility, frames and native fail-closed modes."""
import tempfile
import unittest
import numpy as np
from blender_blocking.evaluation.protocols.core import (PROFILES, evaluate_points, metrics_from_distances,
    shared_transform, apply_transform, macro_cases, union_occupancy, common_grid_sdf_diagnostic)
from blender_blocking.evaluation.protocols.bundles import save_sample_bundle,reload_sample_bundle
from blender_blocking.evaluation.protocols.dtu import dtu_directed_metrics,dtu_native_adapter,DTU_MODES
from blender_blocking.evaluation.geometry import volumetric_iou


class MetricProtocolTests(unittest.TestCase):
    def test_exact_singletons_and_unequal_two_point_directions(self):
        r=evaluate_points([[0,0,0]],[[3,4,0]],matrix=np.eye(4),profile='superflex_formula_common_v1')
        self.assertEqual(r['metrics']['cd_l1'],5)
        self.assertEqual(r['metrics']['cd_l2'],25)
        r=evaluate_points([[0,0,0],[2,0,0]],[[0,0,0],[1,0,0],[10,0,0]],matrix=np.eye(4))
        np.testing.assert_array_equal(r['prediction_to_reference'],[0,1,8])
        np.testing.assert_array_equal(r['reference_to_prediction'],[0,1])
        self.assertEqual(r['metrics']['cd_l1'],1.75)
        self.assertAlmostEqual(r['metrics']['cd_l2'],(65/3+.5)/2)

    def test_threshold_is_inclusive_and_superflex_epsilon_is_exact(self):
        r=metrics_from_distances([.01,.015,.02,.020001],[.01,.015,.02],'superflex_formula_common_v1')
        for t,p,recall in ((.01,.25,1/3),(.015,.5,2/3),(.02,.75,1)):
            f=r['f_scores'][str(t)]
            self.assertEqual(f['precision'],p);self.assertEqual(f['recall'],recall)
            self.assertEqual(f['f'],2*p*recall/(p+recall+1e-6))

    def test_uniform_scaling_preserves_f_and_scales_both_chamfers(self):
        from dataclasses import replace
        c=PROFILES['superflex_formula_common_v1']
        p=np.array([.005,.03]);r=np.array([.01,.02])
        a=metrics_from_distances(p,r,c);b=metrics_from_distances(p*7,r*7,replace(c,thresholds=tuple(t*7 for t in c.thresholds)))
        self.assertAlmostEqual(b['cd_l1'],a['cd_l1']*7)
        self.assertAlmostEqual(b['cd_l2'],a['cd_l2']*49)
        for t in c.thresholds:self.assertAlmostEqual(a['f_scores'][str(t)]['f'],b['f_scores'][str(t*7)]['f'])

    def test_target_transform_is_shared_and_prediction_translation_worsens(self):
        ref=np.array([[-2,-1,0],[2,1,1]],float)
        matrix=shared_transform(ref,1.8)
        self.assertAlmostEqual(np.ptp(apply_transform(ref,matrix),axis=0).max(),1.8)
        a=evaluate_points(ref,ref,profile='superfit_formula_common_v1')
        b=evaluate_points(ref,ref+[5,0,0],profile='superfit_formula_common_v1')
        self.assertEqual(a['metrics']['reported_cd'],0)
        self.assertGreater(b['metrics']['reported_cd'],100)
        np.testing.assert_array_equal(a['transform']['shared_matrix'],b['transform']['shared_matrix'])

    def test_legacy_formula_sum_is_preserved(self):
        legacy=metrics_from_distances([1,2],[3],'legacy_bbox_v1')
        common=metrics_from_distances([1,2],[3],'common_surface_v1')
        self.assertEqual(legacy['cd_l1'],2*common['cd_l1'])
        self.assertEqual(legacy['cd_l2'],2*common['cd_l2'])

    def test_macro_is_per_object_and_keeps_failures(self):
        cases=[{'case_id':'small','status':'available','metrics':{'cd_l1':1}},
               {'case_id':'large','status':'available','metrics':{'cd_l1':3}}]
        self.assertEqual(macro_cases(cases)['macro']['cd_l1'],2)
        report=macro_cases(cases+[{'case_id':'missing','status':'unavailable','reason':'missing assets'}])
        self.assertIsNone(report['macro']);self.assertEqual(len(report['failures']),1)
        with self.assertRaises(ValueError):macro_cases(cases+cases[:1])

    def test_native_profiles_refuse_random_reference_fallback(self):
        ref=np.zeros((4096,3));pred=ref.copy()
        self.assertEqual(evaluate_points(ref,pred,profile='superflex_native_v1',matrix=np.eye(4))['status'],'unavailable')
        self.assertEqual(evaluate_points(ref[:2048],pred[:2048],profile='superfit_native_v1',matrix=np.eye(4))['status'],'unavailable')

    def test_native_superflex_cannot_silently_normalize_the_supplied_frame(self):
        ref = np.zeros((4096, 3))
        metadata = {"reference_sampling":"released_fps", "reference_sha256":"fixture",
                    "frame":"released_shapenet", "prediction_sampling":"area_weighted_triangles"}
        accepted = evaluate_points(ref, ref, profile="superflex_native_v1",
                                   matrix=np.eye(4), reference_metadata=metadata)
        self.assertEqual(accepted["status"], "available")
        changed = np.eye(4); changed[:3, :3] *= 2
        rejected = evaluate_points(ref, ref, profile="superflex_native_v1",
                                   matrix=changed, reference_metadata=metadata)
        self.assertEqual(rejected["status"], "unavailable")
        self.assertIn("normalize=False", rejected["reason"])

    def test_reproducible_bundle_keeps_raw_directions_units_and_hashes(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'case'
            r=evaluate_points([[0,0,0]],[[1,0,0]],matrix=np.eye(4))
            save_sample_bundle(out,r,case_id='fixture',seed=77,original_units='metres',source_hashes={'reference':'pinned_fixture'})
            loaded=reload_sample_bundle(out)
            np.testing.assert_array_equal(loaded['prediction_to_reference'],[1])
            self.assertEqual(loaded['original_units'],'metres')
            (out/'samples.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):reload_sample_bundle(out)

    def test_campaign_preserves_failed_cells_in_each_mode_macro(self):
        import importlib.util
        from pathlib import Path
        import json
        import sys
        from unittest.mock import patch
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_protocol_campaign.py"
        spec = importlib.util.spec_from_file_location("protocol_campaign_fixture", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty-result.json"; empty.write_text("{}")
            for group in ("paired", "heldout"):
                folder = root / group; folder.mkdir()
                rows = [{"case":"failed", "requested_mode":"ensemble"},
                        {"case":"empty", "requested_mode":"ensemble", "result_path":str(empty)}]
                (folder / "final.json").write_text(json.dumps({"seed":77, "rows":rows}))
            with patch.object(sys, "argv", ["campaign", "--phase", str(root), "--output", str(root / "bundles")]):
                module.main()
            report = json.loads((root / "bundles/summary.json").read_text())
            for aggregate in report["aggregates"].values():
                self.assertEqual(aggregate["case_count"], 2)
                self.assertEqual(len(aggregate["failures"]), 2)
                self.assertIsNone(aggregate["macro"])
            self.assertEqual(len(list((root / "bundles").rglob("manifest.json"))), 12)

    def test_analytic_cube_overlap_and_active_union_not_parity_xor(self):
        axis=(np.arange(64)+.5)/32-1
        x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
        a=(abs(x)<.5)&(abs(y)<.5)&(abs(z)<.5)
        b=(abs(x-.5)<.5)&(abs(y)<.5)&(abs(z)<.5)
        self.assertAlmostEqual(volumetric_iou(a,b),1/3)
        union=union_occupancy([a,b])
        self.assertTrue(np.all(union[a&b]))
        self.assertGreater(union.sum(),np.logical_xor(a,b).sum())
        inner=(abs(x)<.25)&(abs(y)<.25)&(abs(z)<.25)
        shell=a&~inner
        self.assertAlmostEqual(volumetric_iou(shell,a),7/8)
        self.assertNotEqual(shell.sum(),a.sum())

    def test_common_sdf_grid_is_labeled_diagnostic_and_bounded(self):
        a=np.ones((128,)*3,dtype=np.float32);a[32:96,32:96,32:96]=-1
        r=common_grid_sdf_diagnostic([a],[a])
        self.assertEqual(r['volumetric_iou'],1)
        self.assertEqual(r['compatibility'],'common_grid_diagnostic')
        with self.assertRaises(ValueError):common_grid_sdf_diagnostic([a],[a],resolution=512)

    def test_dtu_strict_20mm_boundary_and_asymmetric_sets(self):
        r=dtu_directed_metrics([[19.999,0,0],[20,0,0],[20.001,0,0]],[[0,0,0]],[[0,0,0]],[[1,0,0]])
        self.assertEqual(r['kept_prediction_count'],1)
        self.assertAlmostEqual(r['overall_mm'],(19.999+1)/2)
        self.assertEqual(r['accuracy_mm'],19.999);self.assertEqual(r['completeness_mm'],1)
        r=dtu_directed_metrics([[20,0,0]],[[0,0,0]],[[0,0,0]],[[1,0,0]])
        self.assertEqual(r['status'],'unavailable')

    def test_dtu_calibration_and_distinct_wrapper_culling_contracts(self):
        for mode in DTU_MODES:self.assertEqual(dtu_native_adapter(mode=mode)['status'],'unavailable')
        self.assertFalse(DTU_MODES['partgs_block']['silhouette_culling'])
        self.assertTrue(DTU_MODES['partgs_point']['silhouette_culling'])
        matrix=np.eye(4);matrix[:3,:3]*=1000
        np.testing.assert_array_equal(apply_transform([[.001,0,0]],matrix),[[1,0,0]])


if __name__=='__main__':unittest.main()
