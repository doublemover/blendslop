# N2 strictly separated assembly integration

The bounded residual path now accepts at most eight individually closed,
positively oriented, noncollapsed indexed parts within the unchanged total
512-triangle seed limit and 16/32 cell-center grids. Each component receives
its own topology/orientation/positive-volume screen. The existing exact
within-part triangle boundary guard is unchanged.

Every pair must have strictly disjoint immutable binary64 vertex AABBs on at
least one axis. Endpoint comparisons use no tolerance, welding, inflation or
minimum-gap guess. Since each triangle lies inside its vertex AABB, this
establishes disjointness for that restricted source representation. Touching,
overlapping, nested and AABB-overlapping-but-geometrically-disjoint components
are conservatively rejected. This does not implement a general union/cavity
qualifier, native solid qualification, or a rectangular pixel clearance bound.

For accepted separated parts, the existing nearest-triangle distance and
oriented winding produce negative material/positive exterior samples of their
union. Individual positive orientation prevents a large outward component from
masking a smaller inward one in the aggregate volume screen. An independently
computed analytic two-box union agrees at anisotropic 16 and 32 resolutions.

Three additional host checks exercise these field values, unsupported contacts
and cavities, independently inward parts, and the actual extraction/scoring/
source-retention path. The focused implicit/pixel scope passes 30 checks.
The consolidated P5/implicit/adjacent scope passes 132 checks.

The actual pinned CPU script passed five warm owned-helper jobs, including two
new separated-assembly cases. Both extracted assemblies retain two components
and pass exact boundary checks. Original-camera legacy IoUs for the 16-grid
proposal are approximately .796/.755/.796 versus a perfect source; the 32-grid
proposal reproduces the source masks. These are implementation fixtures, not
reconstruction improvements or independent performance measurements. Both
backend outputs retain the original assembly with native qualification absent.

Source, extracted proposal, retained mesh, seed/fitted fields, replay contracts,
and original probability/validity/reliability arrays are retained separately in
the recovery. Legacy source/proposal masks and continuous pixel-area projections
are also retained, with evidence identity and camera bounds independently
checked. The reference masks were generated with the unchanged legacy PIL
operator. No native renderer/filter contract is inferred from these arrays.

Native Blender rendering/export, arbitrary multipart overlap, enclosed shells,
larger grids and reconstruction quality acceptance remain open.
