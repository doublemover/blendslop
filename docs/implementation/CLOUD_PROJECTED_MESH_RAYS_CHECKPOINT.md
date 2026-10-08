# Thickness-independent projected mesh rays

The named observed_projected_rays mode fixes the demonstrated D2 thin-depth
surrogate issue. It projects actual seed triangles, resolves their opaque 2D union
(including holes), and filters the resulting oriented boundary with the pinned DVX
2D one-voxel box integral. Each boundary vertex carries an original projected
vertex or a supported edge-intersection relationship. Torch differentiates those
relationships and the existing DVX filter. Invisible depth is not rewarded for
filling a volumetric cell.

Topology is recomputed each evaluation. Pullbacks describe the current active
projected-union topology; topology changes are nonsmooth and coincident ties use
deterministic subgradients. Unresolved carriers, invalid geometry or budget overflow
fail closed. Current bounds are 2,048 input triangles, 4,096 boundary vertices and
32 projected components. This is a bounded CPU prototype, not a promise to handle
arbitrary dense meshes. The selected helper must also have Shapely >=2.1,<3;
the exact tested cloud package is official Shapely 2.1.2.

The conditioned path preserves its seed transform/topology, cage/differential/vertex
choices, local area/edge/normal/anchor protection, staged best-state retention and
original exact folded-part guard. The new objective consumes original camera ray
evidence and avoids unused solid-hull construction. Quality chooses it only when
no explicit alternate objective was given. The old max-filtered-volume ray mode
and filtered/legacy occupancy remain separately available, with their original
surrogate/thin limitations recorded.

Four bounded actual pinned-runtime fixture stages pass:

- A closed subvoxel-thin box matches its projected box coverage with maximum
  error 1.2e-15 and exactly zero invisible-depth evidence gradient.
- Original-vertex pullbacks agree with independent numeric derivatives below
  2e-12; edge-intersection pullbacks agree below 1.1e-9 away from topology events.
- A two-update 16-to-32 conditioned projected-ray job completes and evaluates
  its final state. Within-stage synthetic objective losses improve; this is not
  a reconstruction gain, and losses across stages are not directly comparable.

`validation/projected-mesh-focused.json` records 66 passing affected CPU contracts
and 579 parsed Python files, including the original exact-contact/fold cases and
owned-helper lifecycle. New fixtures cover overlap, hole orientation, real gaps,
unsupported projections, explicit quality alternatives and high-level thin-seed
routing. `validation/projected-mesh-rays.json` retains the actual pinned receipts.

Dense/adaptive projected extraction, robust derivatives at topology events,
original renderer parity, native final qualification and all matched reconstruction
quality/speed measurements remain open. No paid compute, GPU, dataset download or
new user-computer task was used.
