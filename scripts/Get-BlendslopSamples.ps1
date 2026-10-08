#Requires -Version 7.2
<#
.SYNOPSIS
Fetch the curated Blendslop sample pack with PowerShell and .NET only.
.DESCRIPTION
One file, one folder. By default, saves under this repository's temp/sample-pack,
resolved from the script location rather than the caller's current directory.
Run it to fetch 16 PartObjaverse source GLBs, 12
PrimitiveAnything point clouds, and the 16 matching SuperFit fitted outputs
plus their configurations/notices. No Python, installs, external manifests,
full archives, logins, model execution, or pickle deserialization.

Expected payload: 36,246,898 bytes plus small ZIP local-header ranges.
PartObjaverse: 22,059,517 compressed bytes; PrimitiveAnything: 6,124,524 bytes;
SuperFit: 8,062,857 bytes. Extracted asset total: 43,472,361 bytes.
DTU is not included. The two cohorts have 16 and 12 distinct subjects;
fitted outputs are not additional subjects or ground truth.

The folder must be a private local absolute path. Existing verified copies
inside that folder may be reused. Nothing elsewhere is searched. Existing
mismatches are preserved and reported. Do not use a folder concurrently
modified by another program. Links/junctions are refused, not followed.

Successful files are reused on another run. Interrupted individual transfers
restart; they never trigger a whole-archive fallback. Checksums recorded after
download are clearly distinguished from independently pinned checksums.

Requires an already installed PowerShell 7.2+; Windows PowerShell 5.1 is not
supported. This script does not change execution policy or install runtimes.
.EXAMPLE
.\scripts\Get-BlendslopSamples.ps1 -Folder 'D:\BlendslopSamples'
.EXAMPLE
.\scripts\Get-BlendslopSamples.ps1
.EXAMPLE
.\scripts\Get-BlendslopSamples.ps1 -Folder 'D:\BlendslopSamples' -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Position = 0)]
    [Alias('Destination')]
    [string] $Folder
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Frozen revisions, object identities, sizes, CRCs, and available SHA256 pins.
$ManifestJson = @'
{"schemaVersion":1,"createdUtc":"2026-10-07","allowedHosts":["huggingface.co","us.aws.cdn.hf.co"],"partObjaverse":{"schemaVersion":1,"comparison":"16 source mesh objects shared with SuperFit; reusable for compatible geometry methods under a separately named common-object protocol","sourceUrl":"https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny/resolve/86509c37429a4df292d7f22f4d00f5cd7993a5a2/PartObjaverse-Tiny_mesh.zip","archiveBytes":482308457,"maximumObjects":16,"selectionBeforeScores":true,"objects":[{"name":"PartObjaverse-Tiny_mesh/3b4702ea84544ab5bf0cbfe91df1b789.glb","size":86644,"compressed":31010,"crc32":3101064032,"offset":148891807,"id":"3b4702ea84544ab5bf0cbfe91df1b789","label":"Electronics on support","rationale":"Compact body with slender supporting frame","sha256":"1d1a237963ff6c9ea7396a84a5251e1cb55109ea769de3d1048845566ff46b99","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Electronics","parts":["Base","Subject","Support Frame"]}],"superfitPath":"dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/963c5f8b81c045fcab90387bf3aa5143.glb","size":1947844,"compressed":1922483,"crc32":3879665647,"offset":208821469,"id":"963c5f8b81c045fcab90387bf3aa5143","label":"Kettle","rationale":"Round body, handle opening and spout","sha256":"4fd82e057a149ae5caad3d79fb39ba33091939d39e0df958aecab04d3435cd03","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Electronics","parts":["Bottom Of Pot","Pot Lid","Kettle Body","Kettle Handle"]}],"superfitPath":"dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/8a5c0fd0d4bc45128f3ee5c65a32e54f.glb","size":4080632,"compressed":3155204,"crc32":1841392971,"offset":46590155,"id":"8a5c0fd0d4bc45128f3ee5c65a32e54f","label":"Bottle assembly","rationale":"Axial body, small straw and ring","sha256":"28385090b47ae5cc79a9d66575e9fc444b7cb180e51f2a03c1c5db6473fbf4b7","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Daily-Used","parts":["Bottle","Bottle Cap","Straw","Ring"]}],"superfitPath":"dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/94f6eb48b8fe44c69dba4ef04cea53b4.glb","size":1333076,"compressed":949886,"crc32":1554517567,"offset":284673461,"id":"94f6eb48b8fe44c69dba4ef04cea53b4","label":"Carriage vehicle","rationale":"Boxy body, wheels and repeated window detail","sha256":"2d76c4669cfa65635148976a487afbe96790d121290f70ebf42b54bbe285471f","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Transportations","parts":["Carriage","Door","Window","Tire","Lights"]}],"superfitPath":"dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/1b41e3e63867499cada873215bf255ee.glb","size":642984,"compressed":238621,"crc32":1430870677,"offset":82514943,"id":"1b41e3e63867499cada873215bf255ee","label":"Truck-like vehicle","rationale":"Cab/chassis decomposition and separated wheels","sha256":"2573b2540c019a9f682f8b84fdc1232e7bbe5a13f69d5886c52db3a4cb199fc8","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Transportations","parts":["Base","Cab","Chassis","Tire","Window","Vitta"]}],"superfitPath":"dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/0c3ca2b32545416f8f1e6f0e87def1a6.glb","size":449772,"compressed":268029,"crc32":2684336588,"offset":349648159,"id":"0c3ca2b32545416f8f1e6f0e87def1a6","label":"Apples and tray","rationale":"Multiple smooth bodies on thin base","sha256":"66068796d701c502fe606a447f17567e841c069ed6faaa80b7e65cc047e957e7","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Food","parts":["Apple","Apple Stem","Tray"]}],"superfitPath":"dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/01fe14f50de443e4b11ad1fe033df666.glb","size":592816,"compressed":121402,"crc32":920366983,"offset":422745923,"id":"01fe14f50de443e4b11ad1fe033df666","label":"Pavilion/building","rationale":"Thin columns, layered roof and open architectural gaps","sha256":"922cec7bd0be66336fe1225176d6e48c1713717907b5dcdd92cc86ddc9093142","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Buildings&&Outdoor","parts":["Brackets","Door","Foundation","Column","Lantern","Railing","Roof","Support Frame","Wall","Wing Angles","Window"]}],"superfitPath":"dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/186ecaa38ee9468eb222328852afbd7c.glb","size":708524,"compressed":601075,"crc32":2330137340,"offset":424713036,"id":"186ecaa38ee9468eb222328852afbd7c","label":"Tree","rationale":"Branching trunk and layered foliage","sha256":"5329a47deefc3ae1b6ef99d3c6af49d14d4afdb857226f7b7ce3240aa9b5b012","status":"existing source rehashed; prior native preparation recorded","selectionEvidence":"publisher part labels and earlier source preparation; no new quality scores used","publisherPartLabels":[{"category":"Plants","parts":["Grassland","Leaf","Trunk"]}],"superfitPath":"dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/01b8043112e74366a21256d5e64398fb.glb","size":556812,"compressed":336973,"crc32":2002501362,"offset":357627562,"id":"01b8043112e74366a21256d5e64398fb","label":"Eyeglass frame","rationale":"Thin closed loops and open negative space","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Daily-Used","parts":["Frame","Glass","Rims"]}],"superfitPath":"dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/c93ee36deda14b4aa73c2b5a9d9e9c9f.glb","size":915360,"compressed":557223,"crc32":624606977,"offset":276673616,"id":"c93ee36deda14b4aa73c2b5a9d9e9c9f","label":"Headphones","rationale":"Bent open arch with bulky end pieces","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Electronics","parts":["Head Beam","Earphone","Sponge"]}],"superfitPath":"dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/dcf35ae14df947a9a700b8bcff8db57f.glb","size":475144,"compressed":247692,"crc32":2189989375,"offset":305265518,"id":"dcf35ae14df947a9a700b8bcff8db57f","label":"Electric fan","rationale":"Thin radial blades and enclosure gaps","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Electronics","parts":["Base","Body","Enclosure","Fan Blades","Motor","Plug","Wire"]}],"superfitPath":"dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/002e462c8bfa4267a9c9f038c7966f3b.glb","size":1477644,"compressed":512179,"crc32":622518899,"offset":120802370,"id":"002e462c8bfa4267a9c9f038c7966f3b","label":"Camera-like device","rationale":"Box body with round protruding lens and controls","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Electronics","parts":["Body","Button","Glass","Knob","Lens","Ornament","Screen"]}],"superfitPath":"dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/727370d6df3e47bfabc30ac1b10fb445.glb","size":6926924,"compressed":5113966,"crc32":2810291226,"offset":352513449,"id":"727370d6df3e47bfabc30ac1b10fb445","label":"Shoe","rationale":"Organic asymmetry, cavity and narrow laces","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Daily-Used","parts":["Body","Insole","Shoelace","Shoe Labels","Shoe Lining","Sole"]}],"superfitPath":"dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/674e65a1f526461781e59ec49e7200bc.glb","size":1989288,"compressed":1415638,"crc32":212810897,"offset":285623494,"id":"674e65a1f526461781e59ec49e7200bc","label":"Winged animal","rationale":"Thin wings, branching legs and tail","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Animals","parts":["Body","Head","Leg","Wing","Tail"]}],"superfitPath":"dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/053b8a4984e340ad94f3b563594ff09d.glb","size":187772,"compressed":47926,"crc32":812391917,"offset":81943435,"id":"053b8a4984e340ad94f3b563594ff09d","label":"Human figure","rationale":"Articulated limbs and self-occlusion","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Human-Shape","parts":["Body","Head","Knapsack","Arm","Foot","Hand","Leg","Neck"]}],"superfitPath":"dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d/primitive_assembly.pkl"},{"name":"PartObjaverse-Tiny_mesh/02e777bdc5114b148963ed3def6ad471.glb","size":6913744,"compressed":6540210,"crc32":2464599322,"offset":332558437,"id":"02e777bdc5114b148963ed3def6ad471","label":"Potted flowers","rationale":"Thin stems, leaves and layered branching","sha256":null,"status":"source ZIP entry and matching SuperFit directory verified; visual review pending","selectionEvidence":"publisher part labels; label is provisional descriptive shorthand, not verified semantic truth","publisherPartLabels":[{"category":"Plants","parts":["Leaf","Earth","Flower","Flowerpot","Grass","Petals","Stalks","Stones"]}],"superfitPath":"dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471/primitive_assembly.pkl"}],"nativePaperCompatibility":"Not claimed; source preparation and native-frame correspondence remain to validate."},"primitiveAnything":{"schemaVersion":1,"createdUtc":"2026-10-07T08:28:00Z","purpose":"Exactly12PrimitiveAnything point-cloud subjects, six already saved plus six missing; point-cloud comparison only, no exact mesh/solid-GT correspondence claimed.","allowedHosts":["huggingface.co","us.aws.cdn.hf.co"],"defaultTier":"tiny","selection":"One named example in each of twelve distinct challenge classes, chosen before scores. Existing matching files may be reused.","files":[{"id":"pa-points-a-chair_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/a%20chair_001.ply","fileName":"PrimitiveAnything/test_pc/a chair_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"582cd9ffc4779c108f102b27e4bb6b5c1dbf6055ef0dc700c480bfece2c604b3","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-a-helicopter_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/a%20helicopter_001.ply","fileName":"PrimitiveAnything/test_pc/a helicopter_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"38acd318a322b8429b8811929d82e28c3e6cd3ab2ebb7c1e6833d574786a7244","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-a-ladder_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/a%20ladder_001.ply","fileName":"PrimitiveAnything/test_pc/a ladder_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"f5994884c65814d04d5ba2666a778cb6e36e85bb2d211fb69aae52721ea29226","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-a-sports-car_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/a%20sports%20car_001.ply","fileName":"PrimitiveAnything/test_pc/a sports car_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"8ec329ed913b042b267a94002a748d51dbd9ed126c7426ca0995135206676e93","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-old-fighter_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/old%20fighter_001.ply","fileName":"PrimitiveAnything/test_pc/old fighter_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"1969fab397e9dabac3c57be33205b381b1cd7f9cad8de1e2c7cea4b05ce9aeaa","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-trees_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/trees_001.ply","fileName":"PrimitiveAnything/test_pc/trees_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"de5f6b05e54d789d83cd8a9a2515a616f9963a0a435f84ed8d0d2c7c61a633a6","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"not-retrieved","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-sofa_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/sofa_001.ply","fileName":"PrimitiveAnything/test_pc/sofa_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"592f8dfd336ecfbab107cdc0d93df5f65a67ec9acbc229e3a8bea229e29c65ae","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-the-bed_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/the%20bed_001.ply","fileName":"PrimitiveAnything/test_pc/the bed_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"04f133c51c40ff5b7ff19b4d08b4cd1313e1f85c1d2f4bb8501ebdc3f38c4efc","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-the-table_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/the%20table_001.ply","fileName":"PrimitiveAnything/test_pc/the table_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"6a85f0f6c1cfabedb1fa8f6508d236035498bf9d509e8a628a29e203e616f2f1","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-tv_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/tv_001.ply","fileName":"PrimitiveAnything/test_pc/tv_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"684c6563f1bba58fd32a65ddf66259356f0d275b1234114b165ac5f1240d99ed","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-the-stairs_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/the%20stairs_001.ply","fileName":"PrimitiveAnything/test_pc/the stairs_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"e09019c7b92e6b1fba95fd9a178c2d6b2556af53b5ce7537ce3fb5c6f624d7ca","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-points-a-submarine_001","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc/a%20submarine_001.ply","fileName":"PrimitiveAnything/test_pc/a submarine_001.ply","maxBytes":510265,"expectedBytes":510265,"sha256":"be2c79f98bd33a2cfe070ce3842ce17602ff150ca3efbfa142bef043508900e6","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/hyz317/PrimitiveAnything/tree/59606099595f9293fe5c8d05a4779ab95ac7bb69/test_pc","license":"Dataset card: GPL-3.0; retain source notices.","status":"already-retrieved-in-small-bundle","purpose":"10,000-point example, not a source mesh or exact solid-volume GT","archive":null},{"id":"pa-source-card","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/resolve/59606099595f9293fe5c8d05a4779ab95ac7bb69/README.md","fileName":"PrimitiveAnything/SOURCE_README.md","maxBytes":1344,"expectedBytes":1344,"sha256":"79341902b1bf63af0ae02fc7769b27f0e0250a81691f82364f4d00571dadf5d4","hashKind":"reference-artifact","hashSource":"https://huggingface.co/datasets/hyz317/PrimitiveAnything/raw/59606099595f9293fe5c8d05a4779ab95ac7bb69/README.md","license":"GPL-3.0 dataset declaration","archive":null}]},"superFit":{"schemaVersion":1,"createdUtc":"2026-10-07T08:29:00Z","purpose":"Optional fitted outputs for exactly16curatedPartObjaversesourceobjects; sixteen distinct subjects, not32. No automatic pickle deserialization.","allowedHosts":["huggingface.co","us.aws.cdn.hf.co"],"files":[{"id":"sf-readme-md","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/README.md","fileName":"SuperFit/README.md","expectedBytes":11985,"maxBytes":11985,"sha256":null,"hashKind":null,"hashSource":null,"license":"Release data CC BY-NC 4.0; retain notices.","archive":null},{"id":"sf-license","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/LICENSE","fileName":"SuperFit/LICENSE","expectedBytes":4852,"maxBytes":4852,"sha256":null,"hashKind":null,"hashSource":null,"license":"Release data CC BY-NC 4.0; retain notices.","archive":null},{"id":"sf-provenance-md","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/PROVENANCE.md","fileName":"SuperFit/PROVENANCE.md","expectedBytes":2956,"maxBytes":2956,"sha256":null,"hashKind":null,"hashSource":null,"license":"Release data CC BY-NC 4.0; retain notices.","archive":null},{"id":"sf-metadata-json","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/metadata.json","fileName":"SuperFit/metadata.json","expectedBytes":13167,"maxBytes":13167,"sha256":null,"hashKind":null,"hashSource":null,"license":"Release data CC BY-NC 4.0; retain notices.","archive":null},{"id":"sf-3b4702ea84544ab5bf0cbfe91df1b789-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789/config.json","expectedBytes":2693,"maxBytes":2693,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-3b4702ea84544ab5bf0cbfe91df1b789-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789/primitive_assembly.pkl","expectedBytes":361281,"maxBytes":361281,"sha256":"808b1f393f2f0100da0158616074e585d460672f50f892cbd9878a11639494d8","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/3b4702ea84544ab5bf0cbfe91df1b789","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-963c5f8b81c045fcab90387bf3aa5143-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143/config.json","expectedBytes":2693,"maxBytes":2693,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-963c5f8b81c045fcab90387bf3aa5143-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143/primitive_assembly.pkl","expectedBytes":241281,"maxBytes":241281,"sha256":"6b6ef2d8fafb9796332980e2afb319298f5582299fe710c6ab84d171415a7eaf","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/963c5f8b81c045fcab90387bf3aa5143","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-8a5c0fd0d4bc45128f3ee5c65a32e54f-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-8a5c0fd0d4bc45128f3ee5c65a32e54f-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f/primitive_assembly.pkl","expectedBytes":319424,"maxBytes":319424,"sha256":"ef30346603686d39b6d8d8750d79400c32cbd56b7cf632eb181c7c960b1e405e","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/8a5c0fd0d4bc45128f3ee5c65a32e54f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-94f6eb48b8fe44c69dba4ef04cea53b4-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-94f6eb48b8fe44c69dba4ef04cea53b4-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4/primitive_assembly.pkl","expectedBytes":259863,"maxBytes":259863,"sha256":"be6b5d2dee999d60ee9d71db6e8c5b2c64ff5767a438847d0447cd6d052da1d3","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/94f6eb48b8fe44c69dba4ef04cea53b4","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-1b41e3e63867499cada873215bf255ee-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-1b41e3e63867499cada873215bf255ee-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee/primitive_assembly.pkl","expectedBytes":455840,"maxBytes":455840,"sha256":"c1c74791cff1695fd9e919ef26d06556c06b062224869376938ac6ea041b7c64","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/1b41e3e63867499cada873215bf255ee","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-0c3ca2b32545416f8f1e6f0e87def1a6-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-0c3ca2b32545416f8f1e6f0e87def1a6-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6/primitive_assembly.pkl","expectedBytes":384704,"maxBytes":384704,"sha256":"6ef76762c2ef3d88fd19148e29e081d30dc80f6cbb16d3ed7a8a4df1afe2bd4a","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/0c3ca2b32545416f8f1e6f0e87def1a6","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-01fe14f50de443e4b11ad1fe033df666-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-01fe14f50de443e4b11ad1fe033df666-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666/primitive_assembly.pkl","expectedBytes":867845,"maxBytes":867845,"sha256":"f68a69873961fb21f26562b3d892a323c083dae9803d9717934210ef3d0f1e6a","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01fe14f50de443e4b11ad1fe033df666","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-186ecaa38ee9468eb222328852afbd7c-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c/config.json","expectedBytes":2693,"maxBytes":2693,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-186ecaa38ee9468eb222328852afbd7c-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c/primitive_assembly.pkl","expectedBytes":608638,"maxBytes":608638,"sha256":"5260283e058dfece53f485588e4e06522c67a6807cac1f980cc25a2203816016","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/186ecaa38ee9468eb222328852afbd7c","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-01b8043112e74366a21256d5e64398fb-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb/config.json","expectedBytes":2693,"maxBytes":2693,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-01b8043112e74366a21256d5e64398fb-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb/primitive_assembly.pkl","expectedBytes":618657,"maxBytes":618657,"sha256":"301f9c3b1fa9f1d6fd2f2c0485b857552eed7b84d19e88e13e6fa5f74f39f82e","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/01b8043112e74366a21256d5e64398fb","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-c93ee36deda14b4aa73c2b5a9d9e9c9f-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-c93ee36deda14b4aa73c2b5a9d9e9c9f-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f/primitive_assembly.pkl","expectedBytes":321409,"maxBytes":321409,"sha256":"671bc8fd852ce2521f4815ed7aa55ba824734fb1bd5f83d8152e92ae2aea8a62","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/c93ee36deda14b4aa73c2b5a9d9e9c9f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-dcf35ae14df947a9a700b8bcff8db57f-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-dcf35ae14df947a9a700b8bcff8db57f-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f/primitive_assembly.pkl","expectedBytes":969330,"maxBytes":969330,"sha256":"9e5ff65cf9a2e3cc22a819e3e8bcbc1a0522f81ac8e2d0911ba187c9b5998a07","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/dcf35ae14df947a9a700b8bcff8db57f","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-002e462c8bfa4267a9c9f038c7966f3b-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-002e462c8bfa4267a9c9f038c7966f3b-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b/primitive_assembly.pkl","expectedBytes":259560,"maxBytes":259560,"sha256":"4b80c445a1c56578f58f95e7746e6c4dc14d7d56cb60565ed81d1b40acb6191e","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/002e462c8bfa4267a9c9f038c7966f3b","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-727370d6df3e47bfabc30ac1b10fb445-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-727370d6df3e47bfabc30ac1b10fb445-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445/primitive_assembly.pkl","expectedBytes":603566,"maxBytes":603566,"sha256":"2e39d3f373db9c183571f6aaef369a456f1f11babbf91dba36d5b05142a2802d","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/727370d6df3e47bfabc30ac1b10fb445","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-674e65a1f526461781e59ec49e7200bc-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-674e65a1f526461781e59ec49e7200bc-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc/primitive_assembly.pkl","expectedBytes":669299,"maxBytes":669299,"sha256":"9d7bef0a941a588505f88581a917b75e7992ba4505c449d4bd86c52cdc415c68","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/674e65a1f526461781e59ec49e7200bc","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-053b8a4984e340ad94f3b563594ff09d-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d/config.json","expectedBytes":2694,"maxBytes":2694,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-053b8a4984e340ad94f3b563594ff09d-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d/primitive_assembly.pkl","expectedBytes":661116,"maxBytes":661116,"sha256":"d95b1fc3683868974c434754f1087f38830bf9ea52d6872298f2256c1241cea0","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/053b8a4984e340ad94f3b563594ff09d","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-02e777bdc5114b148963ed3def6ad471-config","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471/config.json","fileName":"SuperFit/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471/config.json","expectedBytes":2693,"maxBytes":2693,"sha256":null,"hashKind":null,"hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null},{"id":"sf-02e777bdc5114b148963ed3def6ad471-assembly","tier":"tiny","mode":"public","url":"https://huggingface.co/datasets/bardofcodes/superfit-primitive-assemblies/resolve/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471/primitive_assembly.pkl","fileName":"SuperFit/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471/primitive_assembly.pkl","expectedBytes":384985,"maxBytes":384985,"sha256":"c10b8f0ff94c5b13288923fbd1724f4096d1c168d7177e23ff447edb4e4c3005","hashKind":"published","hashSource":"https://huggingface.co/api/datasets/bardofcodes/superfit-primitive-assemblies/tree/4cf20ea71897d99b3b09063559a66afecf774b97/dataset/partobjaverse/superfrustum/02e777bdc5114b148963ed3def6ad471","license":"Release data CC BY-NC 4.0; source rights remain separate.","purpose":"Fitted output for one of the16curatedsourceobjects. Not GT. Never automatically deserialize pickle.","archive":null}]}}
'@
$Spec = $ManifestJson | ConvertFrom-Json -AsHashtable -Depth 40
$script:Root = $null
$script:Client = $null
$script:Transferred = 0L
$script:MaxTransfer = 115343360L # 110 MiB including any retried response bodies.
$script:MaxAsset = 16777216L     # 16 MiB hard bound per compressed/decoded asset.
$script:Hosts = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($h in $Spec['allowedHosts']) { [void]$script:Hosts.Add($h) }

function Assert-PlainPath([string] $Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if ($full.StartsWith('\\') -or $full.StartsWith('//')) { throw 'Use a local path; UNC/device paths are unsupported.' }
    $cursor = $full
    while ($cursor) {
        try { $attributes = [IO.File]::GetAttributes($cursor) }
        catch [IO.FileNotFoundException] { $attributes = $null }
        catch [IO.DirectoryNotFoundException] { $attributes = $null }
        if ($null -ne $attributes -and ($attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Link/junction/reparse point refused: $cursor"
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    return $full
}
function Get-InFolderPath([string] $Relative) {
    if ([string]::IsNullOrWhiteSpace($Relative) -or $Relative.Contains('\') -or $Relative.Contains(':') -or $Relative.StartsWith('/')) { throw 'Unsafe embedded relative path.' }
    foreach ($part in $Relative.Split('/')) {
        if ($part -in @('', '.', '..') -or $part -match '[<>:"|?*\x00-\x1f]' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\.|$)') { throw 'Unsafe embedded path component.' }
    }
    $full = [IO.Path]::GetFullPath([IO.Path]::Combine($script:Root,$Relative))
    $prefix = $script:Root.TrimEnd([IO.Path]::DirectorySeparatorChar,[IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if (-not $full.StartsWith($prefix,$comparison)) { throw 'Path leaves the chosen folder.' }
    return Assert-PlainPath $full
}
function Ensure-Directory([string] $Path) {
    [void](Assert-PlainPath $Path)
    [void][IO.Directory]::CreateDirectory($Path)
    [void](Assert-PlainPath $Path)
}
function Get-Sha([byte[]] $Bytes) {
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}
function Get-DefinitionSha([hashtable] $Entry) {
    $parts = foreach ($k in @('id','mode','url','fileName','bytes','sha256','entryName','offset','compressed','crc32','archiveBytes')) { [string]$Entry[$k] }
    return Get-Sha ([Text.Encoding]::UTF8.GetBytes(($parts -join "`n")))
}
function Read-Receipt([string] $Path, [hashtable] $Entry) {
    [void](Assert-PlainPath $Path)
    if ([IO.Directory]::Exists($Path)) { throw "Receipt path is a directory; preserved: $Path" }
    if (-not [IO.File]::Exists($Path)) { return @{} }
    if (([IO.FileInfo]::new($Path)).Length -gt 1048576) { throw "Oversized receipt preserved: $Path" }
    $receipt = [IO.File]::ReadAllText($Path) | ConvertFrom-Json -AsHashtable -Depth 20
    if ($receipt -isnot [hashtable] -or $receipt['schemaVersion'] -ne 1 -or $receipt['definitionSha256'] -ne (Get-DefinitionSha $Entry) -or $receipt['sha256'] -notmatch '^[0-9a-f]{64}$' -or [long]$receipt['bytes'] -ne [long]$Entry['bytes']) {
        throw "Receipt differs from this frozen selection; preserved: $Path"
    }
    return $receipt
}
function Write-NewBytes([string] $Path, [byte[]] $Bytes) {
    [void](Assert-PlainPath $Path)
    $stream = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $stream.Write($Bytes,0,$Bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
}
function Write-Json([string] $Path, [object] $Value, [bool] $Replace = $false) {
    [void](Assert-PlainPath $Path)
    $temp = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"
    Write-NewBytes $temp ([Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 40)))
    [void](Assert-PlainPath $Path)
    [IO.File]::Move($temp,$Path,$Replace)
}
function Assert-Url([string] $Value) {
    $uri = $null
    if (-not [Uri]::TryCreate($Value,[UriKind]::Absolute,[ref]$uri) -or $uri.Scheme -ne 'https' -or $uri.Port -ne 443 -or $uri.UserInfo -or $uri.Fragment -or -not $script:Hosts.Contains($uri.IdnHost)) {
        throw 'Only the exact embedded HTTPS hosts are allowed; no credential-bearing URLs.'
    }
    return $uri
}
function Get-SafeError([string] $Message) {
    return ($Message -replace '(https?://[^\s?]+)\?[^\s]+','$1?[redacted]')
}
function Get-Response([Uri] $Url, [long] $Start, [long] $Count, [Threading.CancellationToken] $Token) {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    for ($redirect = 0; $redirect -le 5; $redirect++) {
        [void](Assert-Url $Url.AbsoluteUri)
        if (-not $seen.Add($Url.AbsoluteUri)) { throw 'Redirect loop.' }
        $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get,$Url)
        $response = $null
        try {
            [void]$request.Headers.TryAddWithoutValidation('User-Agent','BlendslopSamples/1.0')
            [void]$request.Headers.TryAddWithoutValidation('Accept-Encoding','identity')
            if ($Start -ge 0) { $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new($Start,($Start+$Count-1)) }
            $response = $script:Client.SendAsync($request,[Net.Http.HttpCompletionOption]::ResponseHeadersRead,$Token).GetAwaiter().GetResult()
            if ([int]$response.StatusCode -in @(301,302,303,307,308)) {
                if ($redirect -eq 5 -or $null -eq $response.Headers.Location) { throw 'Redirect limit or missing Location.' }
                $Url = [Uri]::new($Url,$response.Headers.Location)
                [void](Assert-Url $Url.AbsoluteUri)
                $response.Dispose(); $response = $null
                continue
            }
            $result = $response
            $response = $null # Caller owns the response from this point.
            return $result
        } finally {
            $request.Dispose()
            if ($null -ne $response) { $response.Dispose() }
        }
    }
    throw 'Redirect limit.'
}
function Receive-Bytes([string] $Url, [long] $Count, [long] $Start = -1, [long] $ArchiveBytes = 0) {
    if ($Count -le 0 -or $Count -gt $script:MaxAsset) { throw 'Requested transfer is outside its fixed bound.' }
    if ($Start -ge 0 -and ($ArchiveBytes -le 0 -or $Start+$Count -gt $ArchiveBytes)) { throw 'Range exceeds pinned archive.' }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if ($script:Transferred+$Count+1 -gt $script:MaxTransfer) { throw '110 MiB run transfer budget would be exceeded.' }
        $cts = [Threading.CancellationTokenSource]::new([TimeSpan]::FromMinutes(5))
        $response = $null; $inputStream = $null
        $retry = $false; $why = $null
        try {
            $response = Get-Response (Assert-Url $Url) $Start $Count $cts.Token
            $status = [int]$response.StatusCode
            if ($status -in @(408,429,500,502,503,504)) {
                $retry = $true
                throw "Transient HTTP $status."
            }
            if ($status -in @(401,403)) { throw "HTTP ${status}: access denied; no login, credentials, or acceptance will be attempted." }
            $wanted = if ($Start -ge 0) { 206 } else { 200 }
            if ($status -ne $wanted) { throw "Expected HTTP $wanted, received $status. No whole-archive fallback." }
            $headers = $response.Content.Headers
            if ($headers.ContentType -and $headers.ContentType.MediaType -match '(?i)html') { throw 'HTML response refused.' }
            foreach ($coding in $headers.ContentEncoding) { if ($coding -ne 'identity') { throw 'Transformed HTTP content refused.' } }
            if (($Start -ge 0 -and $null -eq $headers.ContentLength) -or ($null -ne $headers.ContentLength -and [long]$headers.ContentLength -ne $Count)) { throw 'Content-Length does not equal the pinned request length.' }
            if ($Start -ge 0) {
                $range = $headers.ContentRange
                if ($null -eq $range -or $range.Unit -ne 'bytes' -or -not $range.HasRange -or -not $range.HasLength -or $range.From -ne $Start -or $range.To -ne ($Start+$Count-1) -or $range.Length -ne $ArchiveBytes) { throw 'Content-Range does not exactly match the pinned archive and request.' }
            } elseif ($null -ne $headers.ContentRange) { throw 'Unexpected partial-content header.' }
            $buffer = [byte[]]::new([int]$Count)
            $got = 0
            $inputStream = $response.Content.ReadAsStreamAsync($cts.Token).GetAwaiter().GetResult()
            while ($got -lt $buffer.Length) {
                $n = $inputStream.ReadAsync($buffer,$got,([Math]::Min(131072,$buffer.Length-$got)),$cts.Token).GetAwaiter().GetResult()
                if ($n -eq 0) { $retry = $true; throw 'Truncated response.' }
                $script:Transferred += $n
                $got += $n
            }
            # Ranges always have an exact Content-Length: never probe past them.
            # Small direct files may be chunked. One bounded EOF probe verifies
            # that their decoded body has exactly the pinned length.
            if ($Start -lt 0 -and $null -eq $headers.ContentLength) {
                $probe = [byte[]]::new(1)
                $extra = $inputStream.ReadAsync($probe,0,1,$cts.Token).GetAwaiter().GetResult()
                $script:Transferred += $extra
                if ($extra -ne 0) { throw 'Direct file exceeds its pinned size; refused.' }
            }
            return ,$buffer
        } catch {
            $why = Get-SafeError $_.Exception.Message
            if ($_.Exception -is [Net.Http.HttpRequestException] -or $_.Exception -is [IO.IOException] -or $_.Exception -is [OperationCanceledException] -or $_.Exception.InnerException -is [Net.Http.HttpRequestException] -or $_.Exception.InnerException -is [IO.IOException] -or $_.Exception.InnerException -is [OperationCanceledException]) { $retry = $true }
            if (-not $retry -or $attempt -eq 3) { throw $why }
        } finally {
            if ($null -ne $inputStream) { $inputStream.Dispose() }
            if ($null -ne $response) { $response.Dispose() }
            $cts.Dispose()
        }
        Write-Warning "Transfer attempt $attempt failed: $why Retrying this bounded request."
        Start-Sleep -Seconds (2*$attempt)
    }
    throw 'Transfer failed.'
}

# The helper uses only in-memory .NET operations. DeflateStream receives exactly
# one member's compressed bytes, never the entire ZIP or a live HTTP stream.
$BinaryHelper = @'
using System;
using System.IO;
using System.IO.Compression;
using System.Text;
namespace BlendslopSamplesV1 {
    public static class Binary {
        private static readonly uint[] Table = MakeTable();
        private static uint[] MakeTable() {
            var table = new uint[256];
            for (uint i = 0; i < table.Length; i++) {
                uint c = i;
                for (int j = 0; j < 8; j++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                table[i] = c;
            }
            return table;
        }
        public static uint Crc32(byte[] bytes) {
            uint c = 0xFFFFFFFFu;
            foreach (byte b in bytes) c = Table[(c ^ b) & 255] ^ (c >> 8);
            return c ^ 0xFFFFFFFFu;
        }
        public static ushort U16(byte[] b, int p) {
            if (p < 0 || p + 2 > b.Length) throw new InvalidDataException("Truncated ZIP field.");
            return (ushort)(b[p] | b[p+1] << 8);
        }
        public static uint U32(byte[] b, int p) {
            if (p < 0 || p + 4 > b.Length) throw new InvalidDataException("Truncated binary field.");
            return (uint)b[p] | (uint)b[p+1] << 8 | (uint)b[p+2] << 16 | (uint)b[p+3] << 24;
        }
        private static ulong U64(byte[] b, int p) { return U32(b,p) | (ulong)U32(b,p+4) << 32; }
        public static void ValidateGlb(byte[] bytes) {
            if (bytes.Length < 12 || U32(bytes,0) != 0x46546C67u || U32(bytes,4) != 2u || U32(bytes,8) != bytes.LongLength)
                throw new InvalidDataException("Invalid GLB magic, version, or declared length.");
        }
        public static int ValidateLocal(byte[] h, byte[] variable, string expectedName, long size, long compressed, uint crc) {
            if (h.Length != 30 || U32(h,0) != 0x04034B50u) throw new InvalidDataException("Invalid ZIP local header.");
            int flags = U16(h,6), method = U16(h,8), names = U16(h,26), extras = U16(h,28);
            if ((flags & ~0x080E) != 0 || (method != 0 && method != 8)) throw new InvalidDataException("Encrypted/unsupported ZIP flags or method.");
            if (method == 0 && (flags & 6) != 0) throw new InvalidDataException("Stored member has invalid compression flags.");
            if (variable.Length != names + extras) throw new InvalidDataException("Local variable-field length mismatch.");
            byte[] wanted = Encoding.UTF8.GetBytes(expectedName);
            if (wanted.Length != names) throw new InvalidDataException("Pinned ZIP entry name changed.");
            for (int i = 0; i < names; i++) if (variable[i] != wanted[i]) throw new InvalidDataException("Pinned ZIP entry name changed.");
            ulong localSize = U32(h,22), localCompressed = U32(h,18);
            uint localCrc = U32(h,14);
            bool size64 = localSize == uint.MaxValue, compressed64 = localCompressed == uint.MaxValue, zip64Seen = false;
            for (int p = names; p < variable.Length;) {
                if (variable.Length - p < 4) throw new InvalidDataException("Truncated ZIP extra field.");
                int tag = U16(variable,p), n = U16(variable,p+2); p += 4;
                if (n > variable.Length-p) throw new InvalidDataException("ZIP extra field exceeds header.");
                if (tag == 1 && (size64 || compressed64)) {
                    if (zip64Seen) throw new InvalidDataException("Duplicate ZIP64 extra field.");
                    zip64Seen = true;
                    int q = p;
                    if (size64) { if (q+8 > p+n) throw new InvalidDataException("Truncated ZIP64 size."); localSize=U64(variable,q); q+=8; }
                    if (compressed64) { if (q+8 > p+n) throw new InvalidDataException("Truncated ZIP64 compressed size."); localCompressed=U64(variable,q); }
                }
                p += n;
            }
            if ((size64 || compressed64) && !zip64Seen) throw new InvalidDataException("Missing ZIP64 size field.");
            bool descriptor = (flags & 8) != 0;
            if ((!descriptor || localCrc != 0) && localCrc != crc) throw new InvalidDataException("Local CRC differs from pinned ZIP metadata.");
            if ((!descriptor || localSize != 0) && localSize != (ulong)size) throw new InvalidDataException("Local size differs from pinned ZIP metadata.");
            if ((!descriptor || localCompressed != 0) && localCompressed != (ulong)compressed) throw new InvalidDataException("Local compressed size differs from pinned ZIP metadata.");
            if (method == 0 && compressed != size) throw new InvalidDataException("Stored member size mismatch.");
            return method;
        }
        public static byte[] Expand(byte[] compressed, int method, int size, uint crc) {
            if (size <= 0 || size > 16777216 || compressed.Length == 0 || compressed.Length > 16777216)
                throw new InvalidDataException("ZIP member exceeds fixed memory bound.");
            byte[] output;
            if (method == 0) {
                if (compressed.Length != size) throw new InvalidDataException("Stored member length mismatch.");
                output = compressed;
            } else if (method == 8) {
                output = new byte[size];
                using (var member = new MemoryStream(compressed, false))
                using (var inflater = new DeflateStream(member, CompressionMode.Decompress, false)) {
                    int got = 0;
                    while (got < size) {
                        int n = inflater.Read(output, got, Math.Min(131072, size-got));
                        if (n == 0) throw new InvalidDataException("Truncated deflated member.");
                        got += n;
                    }
                    if (inflater.ReadByte() != -1) throw new InvalidDataException("Deflated member exceeds pinned output size.");
                }
            } else throw new InvalidDataException("Unsupported ZIP compression method.");
            if (Crc32(output) != crc) throw new InvalidDataException("Pinned CRC32 mismatch.");
            ValidateGlb(output);
            return output;
        }
    }
}
'@

function Assert-Bytes([byte[]] $Bytes, [hashtable] $Entry, [hashtable] $Receipt, [bool] $Fresh = $false) {
    if ($Bytes.LongLength -ne [long]$Entry['bytes']) { throw 'Exact asset size mismatch; existing file preserved.' }
    $sha = Get-Sha $Bytes
    if ($Entry['sha256']) {
        if ($sha -ne $Entry['sha256']) { throw 'Pinned SHA256 mismatch; existing file preserved.' }
    } elseif (-not $Fresh) {
        if (-not $Receipt['sha256'] -or $Receipt['sha256'] -ne $sha -or $Receipt['definitionSha256'] -ne (Get-DefinitionSha $Entry)) { throw 'No matching prior SHA256 receipt for this unpinned existing file; preserved.' }
    }
    if ($Entry['mode'] -eq 'zip-member') {
        if ([BlendslopSamplesV1.Binary]::Crc32($Bytes) -ne [uint32]$Entry['crc32']) { throw 'Pinned CRC32 mismatch; existing file preserved.' }
        [BlendslopSamplesV1.Binary]::ValidateGlb($Bytes)
    } elseif ($Entry['fileName'].EndsWith('.ply')) {
        if ($Bytes.Length -lt 4 -or $Bytes[0] -ne 112 -or $Bytes[1] -ne 108 -or $Bytes[2] -ne 121 -or $Bytes[3] -notin @(10,13)) { throw 'Invalid PLY header.' }
    }
    return $sha
}
function Read-BoundedFile([string] $Path, [long] $ExpectedBytes) {
    [void](Assert-PlainPath $Path)
    $stream = [IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        if ($stream.Length -ne $ExpectedBytes -or $ExpectedBytes -le 0 -or $ExpectedBytes -gt $script:MaxAsset) { throw 'Existing file size mismatch; preserved.' }
        $buffer = [byte[]]::new([int]$ExpectedBytes); $got = 0
        while ($got -lt $buffer.Length) {
            $n = $stream.Read($buffer,$got,$buffer.Length-$got)
            if ($n -eq 0) { throw 'Existing file changed while being read; preserved.' }
            $got += $n
        }
        if ($stream.ReadByte() -ne -1) { throw 'Existing file changed while being read; preserved.' }
        return ,$buffer
    } finally { $stream.Dispose() }
}
function Get-CandidateIndex([Collections.Generic.HashSet[long]] $WantedSizes) {
    $index = @{}; $count = 0
    $pending = [Collections.Generic.Stack[string]]::new(); $pending.Push($script:Root)
    while ($pending.Count -gt 0) {
        $dir = $pending.Pop(); [void](Assert-PlainPath $dir)
        foreach ($path in [IO.Directory]::EnumerateFileSystemEntries($dir)) {
            $count++
            if ($count -gt 100000) { throw 'Chosen folder has over 100,000 entries. Use a smaller dedicated sample folder.' }
            $attrs = [IO.File]::GetAttributes($path)
            if ($attrs -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($attrs -band [IO.FileAttributes]::Directory) {
                if ([IO.Path]::GetFileName($path) -ne '.blendslop-samples') { $pending.Push($path) }
            } else {
                $length = ([IO.FileInfo]::new($path)).Length
                if ($WantedSizes.Contains($length)) {
                    $key = [string]$length
                    if (-not $index.ContainsKey($key)) { $index[$key] = [Collections.Generic.List[string]]::new() }
                    $index[$key].Add($path)
                }
            }
        }
    }
    return $index
}
function Receive-Asset([hashtable] $Entry) {
    if ($Entry['mode'] -eq 'direct') { return ,(Receive-Bytes $Entry['url'] ([long]$Entry['bytes'])) }
    $start = [long]$Entry['offset']; $archive = [long]$Entry['archiveBytes']
    $header = Receive-Bytes $Entry['url'] 30 $start $archive
    if ([BlendslopSamplesV1.Binary]::U32($header,0) -ne 0x04034b50) { throw 'Pinned offset is not a ZIP local header.' }
    $nameBytes = [int][BlendslopSamplesV1.Binary]::U16($header,26)
    $extraBytes = [int][BlendslopSamplesV1.Binary]::U16($header,28)
    if ($nameBytes -ne [Text.Encoding]::UTF8.GetByteCount($Entry['entryName']) -or $extraBytes -gt 4096) { throw 'Unexpected ZIP name/extra-field size; refusing unreviewed range.' }
    $variable = Receive-Bytes $Entry['url'] ($nameBytes+$extraBytes) ($start+30) $archive
    $method = [BlendslopSamplesV1.Binary]::ValidateLocal($header,$variable,$Entry['entryName'],[long]$Entry['bytes'],[long]$Entry['compressed'],[uint32]$Entry['crc32'])
    $compressed = Receive-Bytes $Entry['url'] ([long]$Entry['compressed']) ($start+30+$nameBytes+$extraBytes) $archive
    return ,([BlendslopSamplesV1.Binary]::Expand($compressed,$method,[int]$Entry['bytes'],[uint32]$Entry['crc32']))
}

# Normalize the three embedded manifests to the same fixed acquisition plan.
$Entries = [Collections.Generic.List[object]]::new()
foreach ($row in $Spec['partObjaverse']['objects']) {
    $Entries.Add(@{
        id=('partobj-'+$row['id']); mode='zip-member'; url=$Spec['partObjaverse']['sourceUrl'];
        fileName=('partobjaverse/'+$row['name']); bytes=[long]$row['size']; sha256=$row['sha256'];
        hashKind=$(if ($row['sha256']) { 'reference-artifact' } else { 'download-recorded' });
        entryName=$row['name']; offset=[long]$row['offset']; compressed=[long]$row['compressed'];
        crc32=[uint32]$row['crc32']; archiveBytes=[long]$Spec['partObjaverse']['archiveBytes'];
        license='Original/source-object rights apply; review the PartObjaverse source before redistribution.'
    })
}
foreach ($cohort in @('primitiveAnything','superFit')) {
    foreach ($row in $Spec[$cohort]['files']) {
        $Entries.Add(@{id=$row['id']; mode='direct'; url=$row['url']; fileName=('downloads/'+$row['fileName']);
            bytes=[long]$row['expectedBytes']; sha256=$row['sha256']; hashKind=$(if ($row['sha256']) { $row['hashKind'] } else { 'download-recorded' }); license=$row['license']})
    }
}
if ($Entries.Count -ne 65 -or $Spec['partObjaverse']['objects'].Count -ne 16 -or $Spec['primitiveAnything']['files'].Count -ne 13 -or $Spec['superFit']['files'].Count -ne 36) { throw 'Frozen cohort count changed.' }
if ([string]::IsNullOrWhiteSpace($Folder)) {
    $repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $Folder = [IO.Path]::Combine($repositoryRoot, 'temp', 'sample-pack')
}
if ([string]::IsNullOrWhiteSpace($Folder) -or -not [IO.Path]::IsPathFullyQualified($Folder)) { throw 'Provide one absolute local folder, for example D:\BlendslopSamples.' }
$script:Root = [IO.Path]::GetFullPath($Folder)
if ($script:Root -eq [IO.Path]::GetPathRoot($script:Root)) { throw 'Choose a sample folder, not a filesystem root.' }
$ids = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$totalOutput = 0L
foreach ($entry in $Entries) {
    if ($entry['id'] -notmatch '^[a-z0-9_-]+$' -or -not $ids.Add($entry['id']) -or -not $paths.Add($entry['fileName'])) { throw 'Invalid/duplicate frozen asset identity.' }
    [void](Assert-Url $entry['url'])
    if ([long]$entry['bytes'] -le 0 -or [long]$entry['bytes'] -gt $script:MaxAsset -or ($entry['sha256'] -and $entry['sha256'] -notmatch '^[a-f0-9]{64}$')) { throw 'Invalid frozen size/hash.' }
    if ($entry['mode'] -eq 'zip-member' -and ($entry['entryName'] -notmatch '^PartObjaverse-Tiny_mesh/[a-f0-9]{32}\.glb$' -or [long]$entry['compressed'] -le 0 -or [long]$entry['compressed'] -gt $script:MaxAsset -or [long]$entry['offset'] -lt 0)) { throw 'Invalid frozen ZIP member.' }
    $totalOutput += [long]$entry['bytes']
}
Write-Host "Folder: $script:Root"
Write-Host '16 source meshes + 12 point clouds + 16 matched fitted outputs/configurations/notices.'
Write-Host 'About 36.25 MB transferred on a fresh run; existing hash-verified copies reduce that.'
Write-Host "Asset output: $totalOutput bytes. No whole archive is downloaded."
Write-Host 'SuperFit release: CC BY-NC 4.0; PrimitiveAnything card: GPL-3.0; original object rights remain separate.'
if (-not $PSCmdlet.ShouldProcess($script:Root,'Fetch and verify the fixed 65-file sample pack')) { return }

# WhatIf returns above, before directory reads/writes, compilation, or network.
[void](Assert-PlainPath $script:Root)
foreach ($entry in $Entries) { [void](Get-InFolderPath $entry['fileName']) }
if (-not ('BlendslopSamplesV1.Binary' -as [type])) { Add-Type -TypeDefinition $BinaryHelper -Language CSharp }
# Known-answer self-checks run locally before the first network operation.
if ([BlendslopSamplesV1.Binary]::Crc32([Text.Encoding]::ASCII.GetBytes('123456789')) -ne 3421780262) { throw 'CRC32 self-test failed.' }
$fixtureGlb = [byte[]]@(103,108,84,70,2,0,0,0,24,0,0,0,4,0,0,0,74,83,79,78,123,125,32,32)
$fixtureDeflate = [byte[]]@(75,207,9,113,99,98,96,96,144,0,98,22,32,246,10,246,247,171,174,85,80,0,0)
$fixtureCrc = [uint32]3513908092
$fixtureHash = Get-Sha $fixtureGlb
if ((Get-Sha ([BlendslopSamplesV1.Binary]::Expand($fixtureGlb,0,24,$fixtureCrc))) -ne $fixtureHash -or (Get-Sha ([BlendslopSamplesV1.Binary]::Expand($fixtureDeflate,8,24,$fixtureCrc))) -ne $fixtureHash) { throw 'Stored/deflated ZIP member self-test failed.' }
Ensure-Directory $script:Root
$metaRoot = Get-InFolderPath '.blendslop-samples'; Ensure-Directory $metaRoot
$receiptRoot = Get-InFolderPath '.blendslop-samples/receipts'; Ensure-Directory $receiptRoot
$stagingRoot = Get-InFolderPath '.blendslop-samples/staging'; Ensure-Directory $stagingRoot
$lockPath = Get-InFolderPath '.blendslop-samples/run.lock'
$lock = [IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$started = [DateTimeOffset]::UtcNow.ToString('o')
$run = [Collections.Generic.List[object]]::new(); $failures = 0
try {
    $wantedSizes = [Collections.Generic.HashSet[long]]::new()
    foreach ($entry in $Entries) { [void]$wantedSizes.Add([long]$entry['bytes']) }
    $candidates = Get-CandidateIndex $wantedSizes
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false; $handler.UseCookies = $false
    $handler.UseDefaultCredentials = $false; $handler.PreAuthenticate = $false
    $handler.Credentials = $null; $handler.UseProxy = $false
    $handler.AutomaticDecompression = [Net.DecompressionMethods]::None
    $script:Client = [Net.Http.HttpClient]::new($handler,$true)
    $script:Client.Timeout = [Threading.Timeout]::InfiniteTimeSpan
    $number = 0
    foreach ($entry in $Entries) {
        $number++
        Write-Progress -Activity 'Blendslop sample pack' -Status "$number/65 $($entry['id'])" -PercentComplete (100*($number-1)/65)
        try {
            $path = Get-InFolderPath $entry['fileName']
            $receiptPath = Get-InFolderPath ('.blendslop-samples/receipts/'+$entry['id']+'.json')
            $old = Read-Receipt $receiptPath $entry
            $bytes = $null; $sha = $null; $status = $null; $reusedFrom = $null
            if ([IO.Directory]::Exists($path)) { throw 'Expected file path is an existing directory; preserved.' }
            if ([IO.File]::Exists($path)) {
                $bytes = Read-BoundedFile $path ([long]$entry['bytes'])
                $sha = Assert-Bytes $bytes $entry $old
                $status = 'verified-existing'
            } else {
                $reuseSha = if ($entry['sha256']) { $entry['sha256'] } else { $old['sha256'] }
                if ($reuseSha -and $candidates.ContainsKey([string]$entry['bytes'])) {
                    foreach ($candidate in $candidates[[string]$entry['bytes']]) {
                        # Only files with matching pinned/receipted SHA are candidates.
                        # Other same-size files are ignored and never changed.
                        try {
                            $candidateBytes = Read-BoundedFile $candidate ([long]$entry['bytes'])
                            if ((Get-Sha $candidateBytes) -ne $reuseSha) { continue }
                            $sha = Assert-Bytes $candidateBytes $entry $old
                            $bytes = $candidateBytes; $reusedFrom = [IO.Path]::GetRelativePath($script:Root,$candidate)
                            $status = 'copied-verified-existing'; break
                        } catch { Write-Verbose "Candidate skipped: $(Get-SafeError $_.Exception.Message)" }
                    }
                }
                if ($null -eq $bytes) {
                    $bytes = Receive-Asset $entry
                    $sha = Assert-Bytes $bytes $entry $old $true
                    if ($old['sha256'] -and $old['sha256'] -ne $sha) { throw 'Download differs from a prior receipt; no final file written.' }
                    $status = 'downloaded-verified'
                }
                Ensure-Directory ([IO.Path]::GetDirectoryName($path))
                $stage = Get-InFolderPath ('.blendslop-samples/staging/'+[Guid]::NewGuid().ToString('N')+'.asset')
                Write-NewBytes $stage $bytes
                # Persist the proof before the no-overwrite promotion. If interrupted,
                # a later run can still verify a promoted unpinned file from this proof.
                $proof = @{schemaVersion=1; id=$entry['id']; fileName=$entry['fileName']; url=$entry['url'];
                    definitionSha256=(Get-DefinitionSha $entry); bytes=$bytes.LongLength; sha256=$sha;
                    pinnedSha256=$entry['sha256']; hashKind=$entry['hashKind']; crc32=$entry['crc32'];
                    license=$entry['license']; verifiedAtUtc=[DateTimeOffset]::UtcNow.ToString('o')}
                Write-Json $receiptPath $proof ([IO.File]::Exists($receiptPath))
                [void](Assert-PlainPath $stage); [void](Assert-PlainPath $path)
                [IO.File]::Move($stage,$path) # Never replaces an existing file.
            }
            if ($old.Count -eq 0 -and $status -eq 'verified-existing') {
                Write-Json $receiptPath @{schemaVersion=1; id=$entry['id']; fileName=$entry['fileName']; url=$entry['url']; definitionSha256=(Get-DefinitionSha $entry);
                    bytes=$bytes.LongLength; sha256=$sha; pinnedSha256=$entry['sha256']; hashKind=$entry['hashKind']; crc32=$entry['crc32']; license=$entry['license']; verifiedAtUtc=[DateTimeOffset]::UtcNow.ToString('o')}
            }
            $run.Add(@{id=$entry['id']; fileName=$entry['fileName']; status=$status; bytes=$bytes.LongLength; sha256=$sha; hashKind=$entry['hashKind']; reusedFrom=$reusedFrom})
            Write-Host "$number/65 $status : $($entry['fileName'])"
        } catch {
            $failures++
            $message = Get-SafeError $_.Exception.Message
            $run.Add(@{id=$entry['id']; fileName=$entry['fileName']; status='failed'; reason=$message})
            Write-Warning "$($entry['id']): $message"
        }
    }
} finally {
    Write-Progress -Activity 'Blendslop sample pack' -Completed
    if ($null -ne $script:Client) { $script:Client.Dispose() }
    try {
        $report = Get-InFolderPath ('.blendslop-samples/run-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+[Guid]::NewGuid().ToString('N')+'.json')
        Write-Json $report @{schemaVersion=1; startedAtUtc=$started; finishedAtUtc=[DateTimeOffset]::UtcNow.ToString('o'); manifestSha256=(Get-Sha ([Text.Encoding]::UTF8.GetBytes($ManifestJson)));
            responseBodyBytes=$script:Transferred; expectedAssetCount=65; completed=($run.Count-$failures); failures=$failures; results=$run.ToArray()}
        Write-Host "Run receipt: $report"
    } finally { $lock.Dispose() }
}
if ($failures -gt 0) { throw "$failures asset(s) failed. Verified successes and mismatched existing files are preserved. Run again to reuse successes; see the run receipt for each blocker." }
Write-Host "Done. All 65 files verified in $script:Root ($script:Transferred response-body bytes transferred)."
Write-Host 'Hashes marked download-recorded are local receipts, not independently published checksums.'
