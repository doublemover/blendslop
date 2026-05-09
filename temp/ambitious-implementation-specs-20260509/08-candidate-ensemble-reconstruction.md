# Spec 08: Candidate Ensemble Reconstruction And Selection

## Scope

This expands idea 2: do not force a single reconstruction path to be best for all inputs. Generate multiple plausible candidates, score them objectively, preserve the full evidence, and select or combine the best result. This turns ambitious modes into a controlled competition rather than a pile of unrelated flags.

Related findings: F01, F04, F05, F07, F08, F10, F14, F16, F20, F22, F25, F26, F28.

## Current Code To Modify

- `blender_blocking/config.py:9`: current valid modes are single-choice.
- `blender_blocking/main_integration.py:593-731`: loft path returns one mesh and one manifest.
- `blender_blocking/main_integration.py:733-1042`: silhouette intersection returns one mesh and one manifest.
- `blender_blocking/main_integration.py:1051-1092`: top-level dispatch selects one mode.
- `blender_blocking/utils/generation_context.py:39-94`: run context can time stages but not candidate branches.
- `blender_blocking/utils/manifest.py:43-60`: manifest has outputs/warnings/errors but no candidate table.
- `blender_blocking/validation/silhouette_iou.py:212-228`: current metric is one IoU result.
- `blender_blocking/integration/blender_ops/render_utils.py:29-129`: render candidate outputs for scoring.
- `docs/IMPLEMENTATION_SPEC.md:281-292`: integration and acceptance should be updated when implementing ensemble mode.

## New Mode

Add:

```python
reconstruction_mode = "ensemble"
```

This mode does not replace individual backends. It orchestrates them:

- `profile_loft`,
- `silhouette_intersection`,
- `visual_hull_voxel`,
- `hybrid_loft_hull`,
- `primitive_fit_refine`,
- `gaussian_ellipsoid_proxy`,
- any plugin backend from Spec 15.

## Candidate Contract

```python
@dataclass(frozen=True)
class CandidateRequest:
    candidate_id: str
    backend_name: str
    config: Mapping[str, object]
    target: ReconstructionTarget
    budget: CandidateBudget

@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    backend_name: str
    status: Literal["success", "degraded", "failed", "skipped"]
    mesh_path: Path | None
    primitive_path: Path | None
    volume_path: Path | None
    render_paths: Mapping[str, Path]
    metric_result: CandidateMetrics
    artifacts: Mapping[str, Path]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
```

Every backend must return a `CandidateResult`; none may silently fall back without marking `degraded`.

## Scoring Function

Create `reconstruction/candidate_scoring.py`.

Primary score:

- required-view pass first,
- per-view area IoU,
- Boundary IoU,
- topology validity,
- manifest completeness,
- reconstruction time,
- mesh complexity,
- editability score,
- uncertainty agreement,
- human constraint satisfaction.

Suggested formula:

```text
score =
  1000 * required_view_all_pass
  + 200 * min_view_area_iou
  + 150 * mean_boundary_iou
  + 100 * topology_score
  + 80  * uncertainty_consistency
  + 60  * editability_score
  - 50  * degraded_penalty
  - 30  * complexity_penalty
  - 20  * time_budget_penalty
```

Do not hide the formula. Store every term in `candidate_scores.json`.

## Candidate Generation Matrix

Generate candidates by input condition:

- front only: profile-based candidates plus primitive fitting; visual hull skipped with reason.
- front+side: profile loft, silhouette intersection, hybrid loft/intersection.
- front+side+top: profile loft, silhouette intersection, visual hull, hybrid loft/hull.
- turntable views: visual hull, sparse volume, Gaussian/ellipsoid proxy, primitive fit.
- noisy/uncertain masks: generate conservative and aggressive mask-policy candidates.

Each candidate gets:

- config hash,
- backend version,
- input target hash,
- budget,
- expected risk label.

## Selection Policies

Support:

- `best_score`: return highest score.
- `quality_first`: prioritize minimum required-view IoU and topology.
- `editability_first`: prefer primitive/procedural outputs when quality is close.
- `fast_preview`: choose low-cost candidate within acceptable metrics.
- `pareto`: return multiple non-dominated candidates.

The output manifest should include selected policy and all candidate scores, not only the winner.

## Blending And Hybridization

Ambitious optional layer:

- Use visual hull as an occupancy constraint to clip or refine loft.
- Use profile loft as initialization for primitive fitting.
- Use silhouette intersection for holes/gaps and loft for smooth regions.
- Use Gaussian/ellipsoid proxy as an intermediate for point/volume refinement.

Hybrid candidates must still produce their own `CandidateResult`.

## Failure Handling

- A failed candidate should not fail the ensemble unless all required candidates fail or the selected policy demands it.
- A candidate failure must include stage, exception, artifacts available, and reproduction command.
- A degraded fallback counts as success only if the scoring function accepts it; it must never be represented as equivalent to the intended backend.

## Config

```python
@dataclass
class EnsembleConfig:
    candidates: tuple[CandidateConfig, ...]
    selection_policy: str
    max_parallel_candidates: int
    per_candidate_timeout_s: float | None
    total_timeout_s: float | None
    keep_all_artifacts: bool
    fail_if_no_candidate_passes_required_views: bool
```

Do not run Blender candidates in parallel unless Blender scene isolation is solved. Pure Python candidates can parallelize.

## Tests

Pure Python:

- scoring sorts candidates as expected,
- failed candidates do not erase successful results,
- degraded candidates receive penalty,
- policy tie-breaking is deterministic,
- mask-policy candidate generation covers noisy masks.

Blender:

- run front+side+top synthetic shape through at least three candidates,
- assert all candidate manifests exist,
- assert winner has best score,
- assert selected mesh is tagged with source candidate id.

## References

- Existing dispatch: `blender_blocking/main_integration.py:1051-1092`.
- Boundary IoU: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/boundary-iou-cvpr2021.pdf`.
- Generalized reprojection error: https://pmc.ncbi.nlm.nih.gov/articles/PMC4281271/.
- Tanks and Temples benchmark metadata: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/tanks-and-temples-2017.pdf`.
