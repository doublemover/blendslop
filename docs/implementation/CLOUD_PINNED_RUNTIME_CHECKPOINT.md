# Pinned CPU runtime numerical checkpoint

Official packages were installed only into the isolated cloud CPython 3.13.5
virtual environment: Torch 2.14.1+cpu from the official PyTorch CPU index,
dvx-python 0.1.1 from PyPI, NumPy 2.5.3 and SciPy 1.18.1. The environment occupies
approximately 1.1 GB; dependency trees and caches are excluded from recovery ZIPs.
The complete installed package receipt is `validation/pinned-dvx-packages.txt`.
No system or user-computer configuration changed.

`validate_dvx_cpu.py --approved` completed seven bounded synthetic contract
stages on Linux with two CPU threads:

- Actual resolution-16 voxelization plus both raw and spatially weighted backward
  passes produced finite, nonzero vertex gradients.
- Torch cage, sparse differential-coordinate and vertex pullbacks reproduced the
  independently checked CPU pullbacks with maximum error below 3e-8.
- One-update filtered-occupancy jobs completed for all three coordinate modes,
  preserving the best scored state and evaluating the final update.
- Two updates across resolution 16 then 32 completed with fixed physical transform
  and topology, distinct stage loss records and correctly reported finest evaluated
  resolution. Targets are explicitly synthetic operator fixtures, not captures.
- Probe and two actual optimization jobs shared one owned helper process and
  closed cleanly.

The actual helper test found that resolving a Linux venv interpreter symlink to
its system binary discarded the venv prefix. Both warm and one-shot invocation
now preserve the selected executable path. A separate regression constructs a
pipless venv and checks the subprocess prefix; source identity watches pyvenv.cfg
as well as the executable and helper sources. Exact dependency/execution guards
remain, and runtime reporting names the real platform rather than claiming Windows.

Full receipts are `validation/pinned-dvx-numeric.json` and its flushed progress log.
These are numerical/source contracts, not reconstruction or performance gains.
Actual voxel-target filter matching, original-camera renderer parity, final exact
fold admission in an integrated candidate, Windows runtime parity, Blender scene
integration and new quality measurements remain open. Historical saved-mesh results
and the separate scikit-image generated-winding evidence remain unchanged.
