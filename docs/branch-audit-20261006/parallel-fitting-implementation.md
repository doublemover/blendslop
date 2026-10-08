# Parallel fitting implementation checkpoint

Historical checkpoint. Current implementation and validation status: [frozen final campaign results](quality-final-results.md). Measurements below retain their original source revisions and do not qualify the new candidate.

The implementation is committed locally at `cbcf14378b47db4c06aacb9f2d5a3baa54b33148`. This checkpoint integrates the approved persistent executor, parameter adapters, exterior objective, changed-part reuse, live analytic silhouette refinement and residual proposals. It starts from `29f40cea46f56d1a83141e9e6fb3eba86d17fdfb` on `endblay_opslay`. The broader improvement phase is still open. Its final matched quality/timing campaign has **not** started, and this checkpoint does not replace historical measurement receipts.

## Implemented and integrated

| Area | Previous behavior | Current implementation |
| --- | --- | --- |
| Normal ensemble and measured routing | Serial Blender path; total deadlines prevented parallel execution | Persistent process workers for both routes, one flat queue for candidates and independent fitting starts, at most four workers/native threads in total |
| Worker ownership and failure handling | Candidate execution could share the caller scene | Process-local scenes; numeric geometry/artifact transport; completed results retained independently; failed/deadline workers joined and recycled; selected geometry rehydrated in the parent |
| Geometry transport | NumPy pickle restored writable arrays with old hashes | Reconstruction through `GeometryArrays.capture` validates buffers, restores read-only flags and recomputes hashes |
| Family-specific fitting parameters | Missing rotations and Gaussian covariance; one scalar scale across unlike units | Local SO(3) rotation, frustum tangent orientation, logarithmic positive dimensions/exponents, extent-scaled translation, six SPD Cholesky Gaussian parameters and analytic pullbacks |
| Exterior fitting objective | Minimum absolute component field could reward buried surfaces | Signed union-field surrogate plus reverse target distances from non-buried component samples; explicitly not exact Chamfer |
| Fitting evaluation cost | Whole component data recomputed for each coordinate trial | Per-fit bounded reuse of unchanged target/occupancy rows, surface samples, nearest-target structures and pair fields; inactive terms skip numeric work and hooks |
| Live silhouette refinement | Coordinate trials repeatedly rendered the entire proxy assembly | Analytic projected-ellipse/Gaussian forward/backward path, covariance/parameter pullback, normalized view weights and explicit L2/product-IoU/fixed-target-distance proposal objective |
| Admission and fallback | Coarse refinement could be reported without a strict final-resolution improvement | Real configured geometric acceptance, retained best valid state, strict full-resolution admission, vectorized paired changed-footprint fallback and ordered one-winner admission |
| Residual structure | Fixed initialized assembly | Input-derived missing-region additions, worst-part splits/reseeding, replacement/reallocation at the part cap, bounded full-objective refinement and regression rejection |
| Live integration and artifacts | Local objective closures were not safe to transport | Rebuildable local hooks, persistent executor shared across family search, residual proposal history and execution evidence in existing result structures |

An automatically initialized primitive search can reserve one part slot for supported growth; an explicit primitive count is respected. Structural refinement can be disabled through its zero round/proposal settings. Tiny evaluation allowances are spent on coordinate fitting rather than an unusable structural reserve. Strong profile/hull/default-Gaussian geometry algorithms were not changed in this checkpoint.

## Focused evidence

The final coordinated focused pass ran eight affected suites: **118 passed, zero failed, zero errors, zero skipped**. Source hashes were unchanged throughout. These checks cover actual local process scheduling/IPC, total deadlines with parallel workers, worker failure retention, simultaneous nested fitting groups on the flat queue, native-array serialization, analytical derivatives, singular/opaque footprint handling, paired fallback admission and partial-winner retention, fitting budgets, configuration and existing differentiable regressions. All 25 changed Python files parsed, and `git diff --check` passed.

Two bounded installed-Blender 5.2.2 checks used the existing calibrated box:

- Profile candidate: process isolation, numeric-array transport, 482 vertices/960 triangles rehydrated into native owned geometry, matching arrays, OBJ presence, owner release and worker join.
- Primitive candidate: two families, four fitting starts, one two-worker queue; both families succeeded. Selected `superfrustum / default_seed_default_step`, **32 total objective evaluations**, 16 per family. Its finite fitting loss changed **0.0893722588 -> 0.0874431436**. A three-part residual split was scored/refined with two calls and rejected at 0.0875535118. Rebuildable hooks executed, primitive JSON/OBJ existed, selected geometry was rehydrated/released and both workers joined.

These are integration checks with deliberately small search settings, not paired benchmark rows, solid/export qualification, a runtime comparison or evidence of held-out generalization. The native smoke precedes the last numeric paired-fallback/inactive-term edits; the final affected pure checks cover those final edits. Blender 4.2/5.0 compatibility has not been exercised locally in this checkpoint.

Evidence is preserved under `temp/parallel-fitting-implementation-20261006`; source and evidence hashes are in [parallel-fitting-implementation.json](parallel-fitting-implementation.json). Original native artifacts and detailed worker logs remain in the owned Temp directories named by the copied receipts.

## Done and pending across the phase

| Phase item | Status at this checkpoint |
| --- | --- |
| Preserve earlier source/evidence and matching camera/unit/crop protocol | Preserved; earlier records retain their revisions and hashes |
| Persistent parallel ensemble/routing and independent fitting starts | Implemented; focused process/native checks passed |
| Scaled rotation/frustum/Gaussian covariance adapters | Implemented; numeric derivative and fitting checks passed |
| Exterior objective, changed-part SDF reuse and no-op term avoidance | Implemented; focused checks passed |
| Live analytic ellipse/Gaussian gradient, paired fallback, real acceptance | Implemented; focused live-path checks passed |
| Residual additions/splits/reallocation/refinement | Implemented and connected; focused tests and one native rejection check passed |
| Profile camera-basis interval-union objective and loft/shape-program changes | Remaining parent design/review; not implemented by this checkpoint |
| Shape-aware non-ellipse gradients, supported subtraction, merge/prune and partition/regrow/fill | Remaining broader technique design; not claimed complete |
| Adaptive hull/loft geometry and balanced native bake/fallback/fit-preservation changes | Remaining broader geometry design; not implemented by this checkpoint |
| DVX, new decomposition or other new dependencies | Not installed or newly implemented; no installation authorization |
| External semantic mesh handoff and native DTU observations/culling receipts | Existing preparation/transfer blockers retained; no substitute dataset scores |
| Optional existing Open3D Poisson qualification | Earlier bounded evidence retained; not repeated or relabeled for current source |
| Freeze one final agreed candidate/configuration | Pending completion of the remaining reviewed implementation |
| Matched training and held-out reconstruction matrix | Pending that final freeze; old affected primitive/differentiable/ensemble measurements do not qualify this source |
| Final full/native quick, solids, modifiers, editability and export campaign | Pending final freeze; two native smokes do not replace it |
| Final per-technique before/after, held-out metrics and exhaustive phase report | Pending the coordinated campaign; no new generalization or speed claim |
| Local implementation commit | Authorized; this checkpoint is committed locally only |
| Pushes, publication, installs and paid compute | None performed |

The `legacy_slice` backend requires the live workflow context and cannot currently cross the isolated numeric request boundary; default independent candidates exclude it and an explicitly requested unsupported route reports through the structured failure path. Custom candidate payloads/backends must be pickle-safe; rejection is job-local. Projection gradients describe ellipse proxies, so they do not establish shape-aware differentiability of a superquadric corner or subtractive CSG. The sampled exterior objective is approximate and remains input-derived.

## Agent closeout

Three bounded assignments were used: parameter adapters plus focused process/native checks; exterior objective plus residual proposals; analytic gradient plus paired fallback. All are completed, with no further delegation layers. Each inherited this task's model and reasoning settings without overrides. Agent status tools do not expose the exact inherited model IDs or reasoning levels, so those values are not guessed. The owner's new limit is in effect: no new agents, replacements, expanded assignments or delegation layers; any remaining approved implementation is handled by the root worker.
