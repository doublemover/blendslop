# Bounded cold helper cleanup

Cold DVX and native boundary qualification no longer use unbounded post-kill `communicate()` calls. One five-second final allowance covers pipe draining and an actual primary `wait()`. The wait consumes the remaining allowance rather than starting a second timer. Original timeout/cancellation errors retain their identity when cleanup or receipt publication also fails.

A primary can exit while an inherited output pipe remains open in a descendant. In that case the cleanup receipt records `primary_joined=true`, `pipes_drained=false`, and leaves the fresh artifact lease active. An unjoined primary also keeps its lease active. Native qualification no longer releases an incomplete lease merely because its context exits, and it never caches that failed admission. Partial scored cold-helper files remain retained.

This helper qualifies the primary and its output transport only. Complete ordinary descendant ownership is available through the separate explicit process-budget API; no universal tree claim or reclamation is added here.

Fourteen distinct focused checks passed using existing Python: eight existing cold transport/failure/partial/cancellation contracts, five cleanup guards, and the existing native timeout-release contract. A real stdlib parent exited while its fresh descendant held the output pipe for 0.6 seconds; cleanup returned within its 0.05-second allowance, reported the pipe incomplete, and the fixture then allowed the short-lived descendant to finish naturally. No DVX/Torch computation, Blender reconstruction, dependency change or cleanup of retained runs was performed.
