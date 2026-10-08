# Improvement phase, 6 October 2026

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

Current accepted continuation: [native performance and metric phase](native-phase.md). This document retains its earlier snapshot and measurements.


The continuation and final implementation/validation status are in [final-phase.md](final-phase.md). This file preserves camera-stage evidence; its next-step notes are historical.

The starting source is `0757de8b872258c927f4ae874d74578ba220f740`, Blender 5.2.2 LTS (`d13f752e3b9c`, Python 3.13.13). The source freeze, original five failures, hypotheses and held-out reservation are in `temp/improvement-phase-20261006/baseline-freeze.json`. Original reports and an untouched detached checkout are preserved. See `docs/IMPLEMENTATION_SPEC.md`, lines 8-21 and 122-125, for orthogonal-input and configuration scope.

## Camera control

Canonical image axes are front (+X,+Z), side (+Y,+Z), top (+X,+Y), with image rows increasing downwards. Synthetic front/side previously used orbit angles that swapped their roles; validation top previously rolled 90 degrees. Future synthetic renders use the canonical convention. Saved historical images are not rewritten and scores across these different capture protocols are not solver deltas.

`reconstruction.view_calibration` accepts one record per supplied canonical view: `projection: orthographic`, `world_bounds: [horizontal_min,horizontal_max,vertical_min,vertical_max]`, optional matching `axes`, and `orientation: canonical_positive_axes`. Bounds describe viewport **edges**, not object bounds. The camera record is permissible input metadata; reference geometry and dimensions are not supplied to solvers. Recorded viewports propagate into validation; no candidate-specific reframing is allowed for calibrated evaluation. Bounds come from foreground pixel edges, with a stated three-pixel/5% extent consistency tolerance. Unsupported perspective, unknown axes/rotation/flips, incomplete or inconsistent metric views reject explicitly. This does not solve perspective cameras or estimate hidden shape/unknown absolute units.

Each calibrated profile is sampled at common world heights using its own horizontal pixel scale and image crop, so unequal known crop/scale does not distort proportions. Uncalibrated input retains the legacy scale assumption and emits an ambiguity warning. Gaussian mesh projection and differentiable fitting now expand object bounds through the foreground ROI instead of stretching object bounds across the entire padded canvas.

Validation: focused projection tests (padding, crop/scale, inconsistency, perspective rejection, observed holes) and installed-Blender quick run: **83 suites passed, 0 failed, 2 skipped**. Logs: `temp/improvement-phase-20261006/camera-quick-repaired.log`; the superseded failing quick log is retained. The first torus Gaussian control passes after camera correction; differentiable still fails, so formulation work remains. Discovery timing overlapped the quick tests and is not final runtime evidence.

## Reproduction

`python scripts/run_improvement_pass.py --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --output temp/improvement-phase-20261006/paired --arm camera_control --cases box,vase,torus --modes profile_loft,visual_hull_voxel,gaussian_ellipsoid_proxy,primitive_fit_refine,differentiable_refine,hybrid_loft_hull,shape_program,ensemble --timeout 100`

The runner starts fresh factory-settings Blender children, four threads, explicit isolated outputs and a 100-second child cap. It records source/config/input hashes, versions, actual rendered views, sampled 3D metrics, and an orbit-45 view never used for fitting or selection. Camera controls will be frozen at their own commit, then compared with solver/selection changes under the same inputs. Held-out shapes cylinder, bottle and chair use seed 77 with known per-view crop and scale changes. They are not tuning cases.
