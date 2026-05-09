# Spec 15: Reconstruction Backend Plugin Contract

## Scope

This expands idea 10. Every ambitious reconstruction approach should plug into one contract. This prevents future modes from becoming bespoke branches in `main_integration.py`.

Related findings: F08, F09, F16, F20, F25, F26, F28.

## Current Code To Modify

- `blender_blocking/config.py:9`: fixed mode set.
- `blender_blocking/main_integration.py:593-731`: loft branch.
- `blender_blocking/main_integration.py:733-1042`: silhouette intersection branch.
- `blender_blocking/main_integration.py:1051-1092`: mode dispatch.
- `blender_blocking/utils/manifest.py:43-60`: manifest should include backend metadata.
- `blender_blocking/benchmarks/benchmark_perf.py:1016-1251`: benchmark dispatch can become backend-aware.

## Backend Protocol

Create `blender_blocking/reconstruction/backend.py`:

```python
class ReconstructionBackend(Protocol):
    name: str
    version: str
    capabilities: BackendCapabilities

    def validate_config(self, config: Mapping[str, object]) -> list[str]: ...
    def estimate_budget(self, target: ReconstructionTarget, config: Mapping[str, object]) -> BackendBudget: ...
    def reconstruct(self, request: CandidateRequest) -> CandidateResult: ...
    def benchmark_cases(self) -> Sequence[BenchmarkCase]: ...
```

Capabilities:

- requires Blender,
- supports pure Python,
- supports multi-view,
- supports top view,
- supports uncertainty,
- supports human constraints,
- outputs mesh,
- outputs volume,
- outputs primitive set,
- supports gradients,
- supports editability score.

## Built-In Backends

Register:

- `LegacySliceBackend`,
- `ProfileLoftBackend`,
- `SilhouetteIntersectionBackend`,
- `VisualHullBackend`,
- `HybridLoftHullBackend`,
- `PrimitiveFitBackend`,
- `GaussianEllipsoidBackend`,
- `DifferentiableRefinementBackend`.

Each backend should own its implementation module and tests. `main_integration.py` should orchestrate, not contain hundreds of backend-specific lines.

## Registry

Create `reconstruction/registry.py`:

```python
def register_backend(backend: ReconstructionBackend) -> None: ...
def get_backend(name: str) -> ReconstructionBackend: ...
def list_backends() -> list[BackendInfo]: ...
```

Optional plugin discovery later:

- local Python entry points,
- config-specified module path,
- experimental backends under `blender_blocking/reconstruction/experimental/`.

Do not import optional heavy dependencies until the backend is selected.

## Request/Result Discipline

All backends receive the same:

- `ReconstructionTarget`,
- `GenerationContext`,
- config,
- constraints,
- budgets,
- artifact directory.

All backends return:

- status,
- artifacts,
- metrics,
- warnings/errors,
- selected/degraded fallback information,
- reproduction command.

No backend should write directly into arbitrary output locations without going through the request artifact root.

## Config

Move from one flat mode string to:

```json
{
  "reconstruction": {
    "mode": "ensemble",
    "backends": {
      "profile_loft": { "enabled": true },
      "visual_hull_voxel": { "enabled": true, "resolution": 128 },
      "primitive_fit_refine": { "enabled": false }
    }
  }
}
```

Legacy config files can be migrated by a config loader.

## Testing

- fake backend registration,
- missing backend error message,
- optional dependency skip behavior,
- backend request artifact root enforcement,
- built-in backend capability listing,
- ensemble uses registry rather than hardcoded branches.

## Migration Plan

1. Add protocol and registry.
2. Wrap existing loft branch as `ProfileLoftBackend`.
3. Wrap existing silhouette intersection branch as `SilhouetteIntersectionBackend`.
4. Wrap legacy branch as `LegacySliceBackend`.
5. Switch `main_integration.py:1051-1092` to registry dispatch.
6. Add ensemble orchestrator.
7. Move experimental backends behind registry.

## References

- Existing mode dispatch: `blender_blocking/main_integration.py:1051-1092`.
- Existing config mode set: `blender_blocking/config.py:9`.
- Candidate ensemble: `temp/ambitious-implementation-specs-20260509/08-candidate-ensemble-reconstruction.md`.
- Benchmark strategy: `temp/ambitious-implementation-specs-20260509/06-validation-benchmarks-artifacts.md`.
