# Protected implicit coordinates: feasible-side follow-through

The first field prototype projected known-empty values above zero, but a latent
zero could remain below that projection. That left the optimizer inside a dead
clamp region. The actual pinned Torch fixture also reports zero clamp derivative
at equality. A protected point therefore needed a more explicit local contract.

Version2 centers protected coordinates on the feasible conditioned field and
reduces their remaining movement budget by the already-used conditioning shift.
Total signed-field change from the original seed remains bounded. The final
inequality projection deliberately retains the feasible-side derivative at
equality. Known-empty points cannot become material, but can now move farther
into the empty region to improve the declared distance/ray surrogate.

Eight actual pinned CPU numerical tests pass, including the new independent
one-sided derivative fixture. Eight host field tests still pass, including
through-hole exact boundary/replay, bridge protection and original-seed bounds.
The coordinate version is included in its constraint identity, so parameter
records cannot silently be interpreted under a different decoding rule.
Authoritative saved-field replay and all native/production acceptance limits
from the foundation checkpoint remain unchanged. No reconstruction or speed
gain is claimed.
