# Spec 13: Human Correction And Constraint Hooks

## Scope

This expands idea 8. Ambitious reconstruction gets much more reliable when the pipeline can accept sparse human guidance: scribbles, known dimensions, symmetry hints, axis hints, centerline hints, and forced foreground/background. This spec defines the data model and integration points without requiring a GUI first.

Related findings: F02, F03, F07, F08, F22, F23, F24, F28.

## Current Code To Modify

- `blender_blocking/config.py:272-344`: `BlockingConfig` has no human constraint fields.
- `blender_blocking/main_integration.py:412-473`: loads silhouettes without user overrides.
- `blender_blocking/main_integration.py:473-488`: calculates bounds only from silhouettes.
- `blender_blocking/geometry/silhouette.py:37-113`: extraction has no scribble/polarity hints.
- `blender_blocking/geometry/dual_profile.py:132-250`: profile generation has optional offsets but no known-centerline constraints.
- `blender_blocking/integration/multi_view/visual_hull.py:210-231`: adds silhouette views without constraint metadata.
- `blender_blocking/placement/resfitting.py:190-738`: optimizer has no hard/soft user constraints.

## Constraint Model

Create `blender_blocking/constraints/`:

```text
constraints/
  __init__.py
  model.py
  apply_to_masks.py
  apply_to_profiles.py
  apply_to_volumes.py
  apply_to_objectives.py
  io.py
```

Data classes:

```python
@dataclass(frozen=True)
class ScribbleConstraint:
    view: str
    kind: Literal["foreground", "background", "unknown"]
    points_px: tuple[tuple[float, float], ...]
    brush_radius_px: float
    confidence: float

@dataclass(frozen=True)
class AxisConstraint:
    axis_name: str
    direction_world: tuple[float, float, float]
    confidence: float

@dataclass(frozen=True)
class DimensionConstraint:
    name: str
    value_u: float
    tolerance_u: float

@dataclass(frozen=True)
class SymmetryConstraint:
    plane: Literal["xz", "yz", "xy", "custom"]
    confidence: float

@dataclass(frozen=True)
class CenterlineConstraint:
    view: str
    points_px: tuple[tuple[float, float], ...]
    tolerance_px: float
```

## Supported Constraint Types

- foreground/background scribbles,
- known bbox,
- known height/width/depth,
- symmetry plane,
- axis orientation,
- front/side/top view role confirmation,
- centerline points,
- fixed contact plane or ground plane,
- preserve holes/components,
- force separate parts,
- ignore component or crop region,
- expected primitive family, e.g. "mostly cylindrical".

## File Format

JSON:

```json
{
  "version": 1,
  "constraints": [
    {
      "type": "scribble",
      "view": "front",
      "kind": "foreground",
      "points_px": [[120, 220], [122, 221]],
      "brush_radius_px": 4,
      "confidence": 1.0
    }
  ]
}
```

Config:

```python
@dataclass
class ConstraintConfig:
    constraint_files: tuple[str, ...]
    fail_on_unsatisfied_hard_constraints: bool
    use_constraints_for_candidate_scoring: bool
```

## Integration Points

Silhouette extraction:

- foreground scribble forces mask foreground with confidence 1.
- background scribble forces mask background with confidence 1.
- bbox hints constrain candidate scoring.
- polarity hint biases extraction.

Profile bands:

- centerline constraints adjust or validate profile centers.
- preserve-hole hints prevent largest-component cleanup from deleting holes/components.

Visual hull:

- dimension constraints set bounds.
- axis constraints set view transforms.
- scribbles update mask uncertainty before carving.

Primitive fitting:

- symmetry and dimension constraints become objective terms.
- primitive-family hints bias initialization.
- hard constraints can reject candidate results.

Candidate ensemble:

- constraint satisfaction is a score term.
- selected candidate must list satisfied and unsatisfied constraints.

## Non-GUI First

No GUI is required to start:

- constraints can be JSON files,
- scribbles can be stored as point arrays,
- future UI can write the same files.

Optional later:

- simple image overlay tool,
- Blender viewport annotation importer,
- web-based mask correction view.

## Tests

- foreground/background scribbles override noisy extraction.
- bbox constraint prevents full-canvas polarity failure.
- dimension constraint changes reconstruction bounds.
- unsatisfied hard constraint fails candidate.
- soft symmetry constraint affects scoring but does not hard-fail.

## References

- Uncertainty spec: `temp/ambitious-implementation-specs-20260509/09-uncertainty-aware-reconstruction.md`.
- Generalized reprojection error: https://pmc.ncbi.nlm.nih.gov/articles/PMC4281271/.
- Boundary IoU: `temp/perf-quality-audit-20260509/lane-e-validation-quality/paper-pdfs/boundary-iou-cvpr2021.pdf`.
