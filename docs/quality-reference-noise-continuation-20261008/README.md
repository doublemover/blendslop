# Independent reference discretization — continuation evidence

The new `evaluation/reference_noise.py` contract fixes authored parameters, baseline/dense/finer tessellations, two comparison pairs, 4096 area samples per direction and seed 61007 before measuring. `scripts/run_reference_noise_check.py` accepts only the authored coverage-reference workload and two independently declared families. It hashes the source receipt, exact authored NPZ files, source code and workload before any metric; it never reads reconstructed candidates. Generated exact meshes and receipts use fresh manifest/lease ownership.

The [machine-readable evidence](reference-noise-evidence.json) retains the initial session and sphere-only continuation. Neither session rendered or launched topology helper children. Both used existing Blender 5.2.2 LTS/Python 3.13.13, two threads, a 120-second outer limit and an 8-GiB sampled RSS limit.

| Authored family | Reference tessellation comparison | Symmetric mean world distance | Oriented face normal p95, degrees |
| --- | --- | ---: | ---: |
| rounded_triangle_dot | 32 corner / 64 dome → 64 / 128 | 0.0000750105 | 1.744125 |
| rounded_triangle_dot | 64 / 128 → 128 / 256 | 0.0000573673 | 0.892333 |
| sphere, radius 0.8 | 64 segments / 32 rings → 128 / 64 | 0.0007974911 | 1.988422 |
| sphere, radius 0.8 | 128 / 64 → 256 / 128 | 0.0002012268 | 0.993639 |

These observations measure reference discretization against finer tessellations. They are not rigorous analytic upper bounds and do not declare acceptable reconstruction error. Family acceptance contracts remain absent, `qualified_limits` remains null, and `noise_qualified` remains false. The vase tolerance is not transferred. Independently authored approximation allowances are still needed before a non-vase candidate can receive a surface acceptance verdict.

The initial session completed the triangle comparisons, then safely stopped before sphere measurement because the reproduced native sphere's indexed byte hash differed. It joined in 7.098 seconds with sampled peak RSS 339,668,992 bytes. The actual retained sphere uses allocation-dependent native vertex/triangle order. A six-test reference contract suite now checks a strict reference-surface equivalence hash: exact binary64 vertex-coordinate multisets and exact oriented triangle-coordinate multisets, with counts, allowing only vertex/face permutation and cyclic triangle indexing. Coordinate changes, opposite winding and changed diagonals fail. This equivalence never qualifies topology identity; the original authored NPZ remains the metric input.

The approved sphere-only continuation confirmed that cause: indexed hashes `57b817e1…` and `bf505a0b…` differ, while both exact oriented geometry hashes equal `035f20b3…`. It completed in 6.006 seconds with sampled peak RSS 266,629,120 bytes. Its lease is released and ownership inspection reports no unknown files. The old failure remains intact, including its unregistered partial `results.json`, which conservatively blocks its read-only reclamation plan. New runs register partial receipts before measurement and save regenerated arrays and both identities if the guard fails.

No candidate-derived cutoff, fit, mask regeneration, full suite, original seven-control rerun, package change or publication was performed. Seventeen canonical integrity/replay/diagnostic tests and six independent reference-noise tests pass on the approved bundled Python. Native comparison capability was exercised only in the two bounded sessions above; other families remain unmeasured.
