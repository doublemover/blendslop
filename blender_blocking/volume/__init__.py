"""Volume grid backends and interchange helpers."""

from .chunks import ChunkedVolumeGrid
from .chunk_cache import (
    ChunkCacheKey,
    ChunkCacheStats,
    VolumeChunkCache,
    chunk_cache_key,
)
from .contracts import (
    Bounds3D,
    Chunk,
    ChunkKey,
    MESH_EXTRACT_METHOD_ALIASES,
    MeshExtractionResult,
    SUPPORTED_VALUE_TYPES,
    normalize_mesh_extraction_method,
    VolumeGrid,
    VolumeMetadata,
    VolumeStats,
    VoxelTransform,
)
from .dense import DenseVolumeGrid
from .hierarchical import HierarchicalOccupancyGrid
from .meshing import extract_mesh, extract_surface_voxels, surface_points
from .openvdb_adapter import (
    OpenVDBStatus,
    OpenVDBVolumeGrid,
    detect_openvdb,
    export_to_openvdb,
    import_from_openvdb,
)
from .serialization import (
    FORMAT_VERSION,
    array_sha256,
    file_sha256,
    load_volume,
    save_volume,
    stable_json_hash,
)
from .sdf_projection import (
    SIGN_CONVENTION,
    SDFProjectionReport,
    SDFProjectionResult,
    occupancy_grid_from_signed_distance,
    occupancy_mask_from_volume,
    signed_distance_field_from_occupancy,
    signed_distance_grid_from_volume,
)
from .sparse_hash import SparseHashVolumeGrid

__all__ = [
    "Bounds3D",
    "Chunk",
    "ChunkCacheKey",
    "ChunkCacheStats",
    "ChunkKey",
    "ChunkedVolumeGrid",
    "DenseVolumeGrid",
    "HierarchicalOccupancyGrid",
    "FORMAT_VERSION",
    "MeshExtractionResult",
    "MESH_EXTRACT_METHOD_ALIASES",
    "OpenVDBStatus",
    "OpenVDBVolumeGrid",
    "SDFProjectionReport",
    "SDFProjectionResult",
    "SIGN_CONVENTION",
    "SUPPORTED_VALUE_TYPES",
    "SparseHashVolumeGrid",
    "VolumeGrid",
    "VolumeChunkCache",
    "VolumeMetadata",
    "VolumeStats",
    "VoxelTransform",
    "array_sha256",
    "chunk_cache_key",
    "detect_openvdb",
    "export_to_openvdb",
    "extract_mesh",
    "extract_surface_voxels",
    "file_sha256",
    "import_from_openvdb",
    "load_volume",
    "occupancy_grid_from_signed_distance",
    "occupancy_mask_from_volume",
    "save_volume",
    "signed_distance_field_from_occupancy",
    "signed_distance_grid_from_volume",
    "normalize_mesh_extraction_method",
    "stable_json_hash",
    "surface_points",
]
