# Read-only additional-view planning foundation

This is the first N4 source foundation, not completed general-camera fitting.
Two retained candidates can share all three canonical silhouettes and still
represent materially different 3D arrangements. The new planner ranks a small
set of possible capture directions by pairwise opaque silhouette coverage
disagreement, practical access weights and an explicit visible capture region.
It returns a calibrated angle/frame, viewport and disagreement region.

## Contracts

- Only successful retained geometry with the exact current evidence hash and
  passing high-quality gates for every observed view is eligible. Stale,
  incomplete, failed, low-fit and duplicate hypotheses are excluded.
- Up to four hypotheses and twelve proposed directions are ranked. Candidate
  and pixel/triangle counts are bounded. Cooperative runtime checks prevent
  starting another projection after the allowance expires. Progress includes
  direction, candidate count and elapsed time.
- The new orthographic planning camera has a proper world right/up/backward
  frame, explicit viewport, pixel-cell centers and invertible depth-aware
  projection. Viewports are shared across the retained hypotheses, rather than
  independently aligning away their geometric disagreement.
- The diagnostic renderer measures the actual opaque projected triangle union
  over whole pixel cells. Subpixel thin geometry remains measurable; confidence
  is not substituted for delivered shape. Original geometry is not modified.
- Observed directions, opposite equivalent silhouette directions and supplied
  reserved evaluation directions are excluded. Access and visibility describe
  proposed capture conditions; they are not new observed evidence.
- Real capture availability defaults false. Without it, only diagnostic rankings
  are returned. Even when a capture plan is present, its oblique action-ready
  flag is false and its pending general-camera evidence requirement is explicit.
  No user capture is initiated, no observation is appended and no automatic
  selector or training/evaluation label is changed.

## Bounded evidence

13 new contracts and 59 affected CPU tests passed. All 590 Python sources
parsed. The independent parity fixture contains two different four-cube
assemblies with exactly matching source-projected front/side/top binary masks.
A 16-pixel additional-view diagnostic identifies substantial oblique
disagreement. It is a synthetic source fixture, not reconstruction accuracy.
Tests cover arbitrary-frame/depth round trips including both poles, canonical
pixel parity, invalid reflected frames, subpixel cell-area preservation,
reserved/observed views, practical visibility, unavailable capture, stale and
failed gates, equivalent reindexed surfaces, expired allowances and unchanged
source/evidence identity. Production calibration still rejects oblique training
input, as an explicit pending contract rather than a silent conversion.

## Remaining acceptance

Production fitters, native capture/rendering and new real observations still
require larger generalized-camera integration. No new image was acquired, no
held-out view became training evidence, and no native Blender or reconstruction
campaign ran. Quality, information reduction and speed gains are unmeasured.
