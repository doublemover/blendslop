# Blender 5.2.2 verification — 2026-10-06

Current accepted continuation: [native performance and metric phase](native-phase.md). This document retains its earlier snapshot and measurements.


Current continuation: [frozen improvement phase](final-phase.md). The figures and open items below describe the earlier audit/compatibility snapshot; the new phase records changed defaults, held-out measurements, measured selection and the existing isolated Open3D helper separately.

Installed Blender **5.2.2 LTS**, build `d13f752e3b9c`, bundled Python **3.13.13** at
`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` was used.
The earlier “not installed” blocker is withdrawn: the user identified this existing installation.
No download, installation or system configuration changes were made. Reconstruction configuration defaults remain unchanged.

**82 quick groups pass, 0 fail, 2 skip** (full procedural/E2E tests outside quick scope).
All **24/24** fresh-process reconstruction checks completed; **19/24** pass the render quality gate.
Quality failures remain measurements and are distinct from process/API failures.
The evaluated-mesh identity/reference integration passes with Chamfer 0, F-score 1 and volume IoU 1.

This follows [IMPLEMENTATION_SPEC.md lines 280–296](../IMPLEMENTATION_SPEC.md#16-testing-strategy).
Optional dependency policy is specified at [lines 318–324](../IMPLEMENTATION_SPEC.md#181-ambitious-backend-completion-requirements).

## Runtime and protocol

| Dependency | Verified version/status |
|---|---|
| numpy | 2.3.4 |
| cv2 | 4.13.0 |
| PIL.Image | 11.2.1 |
| PIL._imaging | available |
| scipy | 1.16.0 |
| skimage | 0.26.0 |
| open3d | unavailable (ModuleNotFoundError) |

Dependencies were resolved through `configure_dependency_paths()`; no unrelated venv was inserted into Blender.
Open3D is absent, so optional Poisson reconstruction remains untested. Core matrix paths do not require it.
EEVEE RNA resolves to `BLENDER_EEVEE`; the actual 5.2.2 build still exposes `taa_render_samples`.
The matrix records requested/applied engine and one sample, Blender build, Python, configs and source hashes.

One repeat of the earlier candidate configuration: box, vase, torus; seed 1234; front/side/top, 512×512;
hull 96, profile samples 160/radial segments 48, Gaussian 32/kmeans, primitive budget 8 s, differentiable budget 20 s.
Fresh processes use factory startup, disabled autoexec, four threads and a 100 s child cap.
User config/scripts/data paths were redirected to the ignored versioned run directory.
This is compatibility evidence; no second optimization search or new within-5.2 baseline trial was performed.

All 24 config hashes match the retained 5.0 candidate run. All nine decoded RGBA reference images match exactly;
their PNG file hashes differ. All three ground-truth OBJ files match byte for byte.
At matrix execution, common production source hashes matched the 5.0 final measurement snapshot; the harness adds runtime/reference metadata
and `--disable-autoexec`. The original trial was a dirty-tree run followed by measurement/instrumentation repairs;
its notes and source paths remain in the JSON. Python/native dependencies changed between versions.

## Three-shape means

| Requested method | IoU 5.0 | IoU 5.2 | F-score 5.2 | Chamfer L1 5.2 | Gate passes | Workflow s 5.2 |
|---|---:|---:|---:|---:|---:|---:|
| profile_loft | 0.9359 | 0.9359 | 0.5937 | 0.0801 | 3/3 | 6.7328 |
| visual_hull_voxel | 0.9728 | 0.9728 | 0.3910 | 0.0952 | 3/3 | 19.8738 |
| gaussian_ellipsoid_proxy | 0.8597 | 0.8597 | 0.3004 | 0.1011 | 2/3 | 10.1817 |
| primitive_fit_refine | 0.7699 | 0.7651 | 0.2240 | 0.1237 | 2/3 | 27.0844 |
| differentiable_refine | 0.7768 | 0.7770 | 0.2129 | 0.1197 | 2/3 | 29.9014 |
| hybrid_loft_hull | 0.9359 | 0.9359 | 0.5937 | 0.0801 | 3/3 | 19.6783 |
| shape_program | 0.7248 | 0.7248 | 0.1920 | 0.1398 | 1/3 | 7.1362 |
| ensemble | 0.9728 | 0.9728 | 0.3910 | 0.0952 | 3/3 | 66.6303 |

18/24 rows reproduce mean/min IoU, Chamfer L1 and F-score within 1e-12.
Wall-budget optimizers can change iteration counts with runtime/environment changes. Their score differences
are retained in the per-case CSV; they are not attributed solely to Blender or interpreted as a general improvement.
Ensemble selects a constituent backend; its selected backend is recorded for each case.

Geometry uses 8192 area-weighted surface samples, seed 1234, individual bbox-center/uniform-longest-extent normalization,
no rotation or anisotropic fitting, and F-score tolerance .02. Chamfer L1 sums directional Euclidean means;
L2 sums directional squared-distance means. Volume is 24³ parity only for one closed component.
No solid-union or self-intersection certification is implied.
Workflow timings exclude Blender startup and geometry evaluation, on a shared Windows host with single samples.

## Actual export/reimport checks

Fixture: two mesh children, transformed/nonuniformly scaled parent, bevel modifiers, one PBR material and metric scale .01.
The existing OBJ/GLB QA adapter is exercised; STL/native blend operators are also executed.

The initial GLB check failed: coordinates expanded 100× after reimport and modifiers were not exported.
The adapter now explicitly applies modifiers and temporarily scales hierarchy roots to metres, restoring source
locations/scales even on failure. Reimport now retains all 216 evaluated triangles with world vertex agreement
within 1e-6 Blender units; exact source-transform restoration is asserted. Blender 5.2 unit scale 1.0 and
Blender 5.0 unit scale .01 also pass all four formats. The failing pre-repair artifacts remain separate.
This adapter repair followed the matrix; export QA was disabled in that matrix, so its measurements remain valid.

| Format | Result | Max world-bound error | Preserved evidence |
|---|---|---:|---|
| obj | pass | 7.1525574e-07 | evaluated bevel geometry, material slots; hierarchy flattened |
| glb | pass | 4.7683716e-07 | evaluated bevel geometry, material slots, parent hierarchy |
| stl | pass | 2.3841858e-07 | evaluated geometry in Blender units; format has no materials/hierarchy |
| blend | pass | 0 | modifiers, material slots, units, parent hierarchy |

Material-slot presence is tested; shader/texture fidelity, UVs, animation, external DCC imports and complete asset delivery remain outside this fixture.

## Saved evidence and reproduction

- [Full small verification JSON](blender52-verification.json)
- [24 paired cases](blender-version-comparison.csv)
- [Eight method summaries](blender52-techniques.csv)
- Ignored raw logs/results/assets: `temp/blender52-compat-20261006/`; earlier 5.0 reports remain unchanged.

Use a fresh output directory to rerun the matrix:

```powershell
$env:BLENDER_USER_CONFIG = '<fresh-output>/user-config'
$env:BLENDER_USER_SCRIPTS = '<fresh-output>/user-scripts'
$env:BLENDER_USER_DATAFILES = '<fresh-output>/user-data'
& '.\.venv312\Scripts\python.exe' scripts/run_comparable_pass.py --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --output '<fresh-output>' --arm candidate --timeout 100
```

Run `scripts/verify_blender_exports.py` through that Blender with `--background --factory-startup --disable-autoexec --threads 4 --python-exit-code 2 --python`,
then pass `-- --output <fresh-output>/exports`. Use the same flags for `scripts/finalize_comparable_pass.py -- --worker contract --output <fresh-output>`.
The summarizer accepts `--old`, `--new`, `--output` and reads saved artifacts without running reconstruction.

Camera/proportion recovery, candidate selection by fresh external render, calibrated 3D thresholds, overlapping-solid validity,
held-out datasets and matched external SOTA comparisons remain open. Full procedural, GPU/OpenVDB/Poisson certification remains unexecuted.
