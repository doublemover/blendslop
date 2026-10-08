# Blender Automated Blocking Tool

Automated tool for creating rough 3D blockouts from orthogonal reference images for sculpting workflows.

**Status:** 🚧 In progress (legacy pipeline stable; roadmap-driven improvements underway)

## Quick Start

**New users start here:**

1. **[BLENDER_SETUP.md](BLENDER_SETUP.md)** - Configure Blender's Python environment (one-time setup)
2. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to your first blockout

Already configured Blender? Jump straight to the [QUICKSTART](QUICKSTART.md).

## Documentation

- **[BLENDER_SETUP.md](BLENDER_SETUP.md)** - Blender Python configuration guide ⭐ Start here
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in minutes
- **[INTEGRATION.md](INTEGRATION.md)** - Detailed API and usage guide
- **[REFINEMENT_LAB.md](REFINEMENT_LAB.md)** - Run reconstruction experiments, rank candidates, and inspect failure diagnostics
- **Testing** - See the Testing section below for local and CI commands
- **[CI_CD.md](CI_CD.md)** - CI/CD testing with real Blender (GitHub Actions, Docker)
- **[../AGENTS.md](../AGENTS.md)** - Quality gates for agents/crews (pre-commit, testing requirements)
- **[E2E_VALIDATION_SUMMARY.md](E2E_VALIDATION_SUMMARY.md)** - Validation framework details
- **[../docs/QUALITY_PERF_GATES.md](../docs/QUALITY_PERF_GATES.md)** - Runner phases, synthetic matrix, budget JSON, and artifact policy

## What It Does

Takes orthogonal reference images (front, side, top views) and automatically generates a 3D blockout mesh ready for sculpting in Blender.

**Input:** Reference images (PNG/JPG with black silhouettes on white background)

**Output:** Joined mesh object in Blender, ready for sculpting

## Quick Example

```python
import sys
sys.path.insert(0, "/path/to/blendslop")

from blender_blocking.main_integration import example_workflow_with_images

# Create blockout from reference images
workflow = example_workflow_with_images(
    front_path="images/front.png",
    side_path="images/side.png",
    top_path="images/top.png"
)
```

## How It Works

1. **Load** orthogonal reference images
2. **Process** images (edge detection, normalization)
3. **Analyze** shapes (contours, circularity, aspect ratio)
4. **Generate** 3D primitives using slice-based reconstruction
5. **Join** primitives into single mesh using boolean operations
6. **Setup** camera and lighting

## Project Structure

```
blender_blocking/
├── README.md                   # This file
├── BLENDER_SETUP.md            # ⭐ Blender Python setup guide
├── QUICKSTART.md               # Quick start guide
├── INTEGRATION.md              # Detailed API guide
├── REFINEMENT_LAB.md           # Experiment and ranking workflow
├── CI_CD.md                    # CI/CD guide (GitHub Actions, Docker)
├── main_integration.py         # Main workflow
├── create_test_images.py       # Test image generator
├── verify_setup.py             # Dependency verification
├── test_runner.py              # Main CI/CD test runner
├── test_version_compatibility.py # Version detection & API compatibility tests
├── test_blender_boolean.py     # Blender API compatibility tests
├── test_integration.py         # Test suite
├── test_e2e_validation.py      # E2E validation with IoU
├── refinement_lab/             # Experiment plans, ranking, reports, and diagnostics
├── requirements.txt            # Dependencies
├── utils/                      # Utility modules
│   └── blender_version.py      # Version detection & compatibility
├── primitives/                 # Blender primitive spawning
│   └── primitives.py
├── shape_matching/             # Slice-based shape analysis
│   └── slice_shape_matcher.py
├── placement/                  # 3D placement & mesh joining
│   └── primitive_placement.py
└── integration/                # Integration modules
    ├── image_processing/       # Image loading & processing
    ├── shape_matching/         # Contour analysis
    └── blender_ops/            # Scene setup, rendering, mesh generation

.github/workflows/
└── blender-tests.yml           # GitHub Actions CI/CD workflow
```

## Requirements

- Python 3.8+
- Blender 4.2 LTS or 5.0 (tested)
- Dependencies: numpy, opencv-python, Pillow, scipy

See [QUICKSTART.md](QUICKSTART.md) for installation instructions.

## Testing

### Running Tests Locally

Blender-only tests must run in Blender; pure-Python tests can run outside Blender:

```bash
# Pure-Python tests + dependency check (outside Blender)
python test_runner.py

# Full test suite
blender --background --python test_runner.py

# Quick tests (for pre-commit)
blender --background --python test_runner.py -- --quick

# Phase-based runner entry points
python test_runner.py --phase pure
blender --background --python test_runner.py -- --phase quick
blender --background --python test_runner.py -- --phase blender
python test_runner.py --phase bench --bench-case quality-smoke --budget-json ../configs/quality_perf_budget-smoke.json
blender --background --python test_runner.py -- --phase quality-smoke --budget-json ../configs/quality_perf_budget-smoke.json

# Verbose output
blender --background --python test_runner.py -- --verbose
```

### Test Suite

The legacy test runner executes these suites:
1. **Pure Python** - Config, geometry, and image-processing tests (no Blender required)
2. **Version Compatibility** - Detects Blender version and validates API compatibility
3. **Boolean Solver Enum** - Validates Blender API enums for current version
4. **MeshJoiner Integration** - Tests mesh joining with actual Blender operations
5. **Full Workflow** - End-to-end procedural generation
6. **E2E Validation** - Complete pipeline with IoU comparison (reference → 3D → render → compare)
7. **Dependency Check** - Verifies all packages installed correctly

The phase runner adds named phases for `pure`, `quick`, `blender`, `bench`, `nightly`, and `quality-smoke`. Benchmark and quality-smoke phases write JSON artifacts and optional budget reports; see [../docs/QUALITY_PERF_GATES.md](../docs/QUALITY_PERF_GATES.md).

### Refinement Experiments

Use the refinement lab when tuning reconstruction output quality instead of changing one value at a time by hand:

```bash
python -m blender_blocking.refinement_lab.cli list-tracks
python -m blender_blocking.refinement_lab.cli plan --suite default-vase --track visual-hull-transform --max-runs 4 --out temp/refinement-runs/plan-smoke.json
```

Run actual reconstruction variants through Blender:

```bash
blender --background --python blender_blocking/test_e2e_validation.py -- --refinement-suite default-vase --refinement-track profile-loft-refinement --refinement-max-runs 1 --refinement-result-root temp/refinement-runs/smoke-profile --no-progress
```

Generated lab artifacts belong under ignored `temp/refinement-runs/`; see [REFINEMENT_LAB.md](REFINEMENT_LAB.md).

### Ambitious Quality/Refinement Smoke

Use the repo-level orchestration script when you want one repeatable pass that
compares the ambitious backends on synthetic fixtures, exercises LPIPS
novel-view scoring, and runs closed-loop refinement sweeps:

```powershell
python scripts\run_quality_refinement_smoke.py --dry-run
python scripts\run_quality_refinement_smoke.py --clean-first
```

The default matrix covers `visual_hull_voxel`, `primitive_fit_refine`,
`gaussian_ellipsoid_proxy`, and `differentiable_refine` across adversarial,
primitive-fit, visual-hull, and smoke synthetic suites. LPIPS runs in its own
Blender process so Torch/LPIPS never shares a process with Open3D-heavy
visual-hull postprocess work. Closed-loop refinement runs launch the planner
itself inside Blender so synthetic references are generated where `bpy` exists.
Outputs go under
`temp/quality-refinement-runs/<timestamp>/` and include `commands.md`,
`summary.md`, `summary.json`, matrix JSON, quality reports, cost reports, and
refinement-loop artifacts. Generated internals use compact directory names to
stay inside Blender and Windows path limits. These are diagnostics only and must
not be committed.

### Supported Blender Versions

Tested and supported versions:
- **Blender 5.0**: Uses EXACT boolean solver
- **Blender 4.2 (LTS)**: Uses FAST boolean solver

Older versions are unverified and may require compatibility updates.

Version detection is automatic - no configuration required.

### CI/CD Integration

See **[CI_CD.md](CI_CD.md)** for:
- GitHub Actions workflow example
- GitLab CI configuration
- Docker-based testing with Blender containers
- Multi-version testing strategy (Blender 4.2 LTS, 5.0)

### For Agents/Automated Workflows

**Pre-commit validation:**
```bash
blender --background --python test_runner.py -- --quick
```

**Quality gates before merge:**
- All tests pass in Blender 5.0: `blender --background --python test_runner.py`
- All tests pass in Blender 4.2 (LTS)
- Exit code 0 = pass, 1 = fail, 2 = runner error

See the Testing section in this README for detailed testing instructions.

## Development

### Created By

This tool was built through Gas Town's multi-agent workflow:
- **chrome** - Primitives library
- **nitro** - Shape matching algorithm
- **guzzle** - Primitive placement
- **witness** - Integration framework
- **sculptor** - Final integration and testing

**Convoy:** hq-cv-7sdbk

### Module Status

- 🚧 Primitives library - Stable; new profile/loft path in progress
- 🚧 Shape matching - Stable; canonical IoU updates in progress
- 🚧 Primitive placement - Stable; heuristics and join modes in progress
- 🚧 Integration framework - Stable; manifest/config updates in progress
- 🚧 Main integration - Stable; optional loft path in progress
- 🚧 Test suite - Stable; expanded pure-Python + Blender tests in progress
- 🚧 Documentation - Updating to match implementation spec

## License

See repository root for license information.
