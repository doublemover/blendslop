# Cold helper ownership continuation

Cold DVX transport now creates a fresh project-owned run folder instead of automatically removing its temporary directory. Input packets, scored progress, result packets, command records and bounded log tails are explicitly registered. The interpreter and helper script remain external shared inputs. Existing warm-helper transport is unchanged.

Success, launch failure, timeout and cancellation retain their histories. The lease is released only after child termination is confirmed. An unconfirmed join leaves the lease active and blocks read-only reclamation planning. Secondary join/receipt errors annotate the primary exception instead of replacing it. Parent PYTHONPATH/PYTHONHOME settings are removed from the approved child interpreter environment, matching the existing warm transport.

Eight focused tests pass using standard-library fixture children and controlled failure mocks. They cover probe/fit transport, environment isolation, scored timeout recovery, timeout without a score, cancellation, unconfirmed termination, failed receipt publication, input byte preflight and launch failure. No DVX/Torch dependency execution, installation or GPU work was needed. The configured generated-artifact limit is checked before coordinator packet writes and when registering child outputs; it is not an operating-system disk quota on the helper.

This is producer adoption, not cleanup permission. No deletion executor, historical directory adoption, blanket reclamation, ACL change or security change is implemented. Render/variant producer adoption and promotion/crash/concurrent/locked-file lifecycle integration remain independent work. The changes stay on the isolated local quality branch pending reviewed publication approval.
