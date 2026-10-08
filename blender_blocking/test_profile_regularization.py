"""Bounded profile smoothing and explicit legacy route contracts."""
import unittest
import numpy as np
from geometry.loft_surface import prepare_loft_surface
from geometry.profile_models import EllipticalSlice
from config import BlockingConfig, ReconstructionConfig
from e2e.cli_args import _parse_args, _apply_cli_args


class ProfileRegularizationTests(unittest.TestCase):
    def test_smooth_regularization_reduces_noise_within_source_budget(self):
        z=np.linspace(0,2.6,65)
        exact=.6+.2*np.cos(2*np.pi*z/2.6)
        noisy=exact+.004*np.sin(np.arange(len(z))*1.7)
        slices=[EllipticalSlice(float(t),float(r),float(r)) for t,r in zip(z,noisy)]
        out=prepare_loft_surface(slices,"smooth",1,9,.006)
        radii=np.asarray([s.rx for s in out])
        self.assertLess(np.mean((radii-exact)**2),np.mean((noisy-exact)**2))
        self.assertLessEqual(np.max(np.abs(radii-noisy)),.006+1e-12)
        self.assertEqual(radii[0],noisy[0]);self.assertEqual(radii[-1],noisy[-1])
        for mode in ("sharp","stepped"):
            baseline=prepare_loft_surface(slices,mode,1)
            self.assertEqual(prepare_loft_surface(slices,mode,1,9,.006),baseline)

    def test_regularization_rejects_unbounded_or_invalid_options(self):
        section=[EllipticalSlice(0,1,1)]
        for window,budget in ((8,.01),(True,.01),(9,0),(9,-1),(9,float("nan"))):
            with self.assertRaises(ValueError):prepare_loft_surface(section,"smooth",4,window,budget)
        config=BlockingConfig();config.mesh_from_profile.regularization_window=9
        with self.assertRaises(ValueError):config.mesh_from_profile.validate()

    def test_cli_and_config_expose_explicit_connected_repair(self):
        args=_parse_args(["--legacy-profile-geometry","connected","--mesh-regularization-window","9",
                          "--mesh-regularization-max-deviation","0.006"])
        config=BlockingConfig();_apply_cli_args(config,args);config.validate()
        self.assertEqual(config.reconstruction.legacy_profile_geometry,"connected")
        self.assertEqual(config.mesh_from_profile.regularization_window,9)
        self.assertAlmostEqual(config.mesh_from_profile.regularization_max_deviation_u,.006)
        self.assertEqual(ReconstructionConfig().legacy_profile_geometry,"auto")
        with self.assertRaises(ValueError):ReconstructionConfig(legacy_profile_geometry="guess").validate()


    def test_quality_legacy_routes_connected_and_stacked_override_stays_explicit(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import main_integration as main
        config = BlockingConfig()
        workflow = SimpleNamespace(config=config, context=SimpleNamespace(),
                                   views={"front": np.zeros((8,8,3))}, create_3d_blockout_loft=Mock(return_value="connected"))
        with patch.object(main,"BLENDER_AVAILABLE",True), patch.object(main,"setup_scene",side_effect=RuntimeError("stacked path"),create=True):
            config.reconstruction.quality_preset="quality"
            self.assertEqual(main.BlockingWorkflow.create_3d_blockout(workflow,65),"connected")
            config.reconstruction.legacy_profile_geometry="stacked"
            with self.assertRaisesRegex(RuntimeError,"stacked path"):
                main.BlockingWorkflow.create_3d_blockout(workflow,65)

    def test_calibrated_cell_edges_preserve_flat_caps_and_interior_empty_rows(self):
        from types import SimpleNamespace
        from reconstruction.types import Bounds2D
        from reconstruction.projection_contract import bounds_from_calibrated_masks, calibrated_profile
        from test_quality_geometry import target_for_masks
        from dataclasses import replace
        mask = np.zeros((20,20),bool)
        mask[2:18,5:15] = True
        records = {view: {"world_bounds": (-1.,1.,-1.,1.)} for view in ("front","side")}
        boxes = {view: Bounds2D(5,2,15,18) for view in records}
        bounds = bounds_from_calibrated_masks({v:mask for v in records},boxes,records)
        target = replace(target_for_masks({v:mask for v in records}), bounds=bounds)
        profile = calibrated_profile(SimpleNamespace(target=target,config={"num_samples":17}))
        np.testing.assert_allclose(profile.rx,.5)
        self.assertEqual(profile.rx[0],profile.rx[-1])
        mask[10,:] = False
        other = replace(target,constraints=tuple(replace(c,mask=mask) for c in target.constraints))
        profile = calibrated_profile(SimpleNamespace(target=other,config={"num_samples":17}))
        self.assertEqual(profile.rx[8],0.)
