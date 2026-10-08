"""Fixed-world narrow-band signed-field coordinates with hard empty protection."""
from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class NarrowBandField:
    seed_zyx: np.ndarray
    active_flat: np.ndarray
    protected_empty_zyx: np.ndarray
    voxel_size_xyz: np.ndarray
    origin_xyz: np.ndarray
    band_width: float
    maximum_displacement: float
    minimum_empty_distance: float
    seed_field_hash: str

    @classmethod
    def create(cls, seed_zyx, voxel_size_xyz, *, origin_xyz, band_width, maximum_displacement,
               known_empty_zyx=None, minimum_empty_distance=None):
        seed = np.array(seed_zyx, dtype=np.float32, order='C', copy=True)
        spacing = np.asarray(voxel_size_xyz, float).copy()
        origin = np.asarray(origin_xyz, float).copy()
        if (seed.ndim != 3 or len(set(seed.shape)) != 1 or seed.shape[0] not in (16, 32)
                or not np.isfinite(seed).all() or spacing.shape != (3,)
                or not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or origin.shape != (3,) or not np.isfinite(origin).all()):
            raise ValueError('implicit coordinates require a finite 16/32-cubed field and positive world spacing')
        floor = float(spacing.min()*.01 if minimum_empty_distance is None else minimum_empty_distance)
        if (not np.isfinite([band_width, maximum_displacement, floor]).all()
                or not 0 < maximum_displacement <= band_width or not 0 < floor <= band_width):
            raise ValueError('implicit band/displacement/empty-distance bounds are incompatible')
        known = np.zeros(seed.shape, bool) if known_empty_zyx is None else np.asarray(known_empty_zyx)
        if known.shape != seed.shape or known.dtype != np.dtype(bool):
            raise ValueError('known-empty field must be an explicit matching Boolean array')
        boundary = np.zeros(seed.shape, bool)
        for axis in range(3):
            low, high = [slice(None)]*3, [slice(None)]*3
            low[axis], high[axis] = 0, -1
            boundary[tuple(low)] = boundary[tuple(high)] = True
        if np.any(seed[boundary] <= 0):
            raise ValueError('source zero set is not contained inside the fixed field domain')
        protected = known | boundary
        if np.any(seed[protected] < floor-maximum_displacement):
            raise ValueError('known-empty evidence conflicts outside the admissible displacement band')
        active = np.flatnonzero(np.abs(seed.ravel()) <= band_width).astype(np.int64)
        if not len(active):
            raise ValueError('implicit seed has no resolved active boundary samples')
        if not np.any(seed < 0):
            raise ValueError('implicit seed material is unresolved at this physical grid spacing')
        for value in (seed, active, protected, spacing, origin):
            value.setflags(write=False)
        identity = hashlib.sha256(str(seed.shape).encode()+seed.tobytes()+spacing.tobytes()+origin.tobytes()).hexdigest()
        return cls(seed, active, protected, spacing, origin, float(band_width),
                   float(maximum_displacement), floor, identity)

    def decode(self, parameters):
        parameters = np.asarray(parameters, float)
        if parameters.shape != self.active_flat.shape or not np.isfinite(parameters).all():
            raise ValueError('implicit parameters must match the finite active field coordinates')
        seed = self.seed_zyx.astype(float)
        conditioned = seed.copy()
        conditioned[self.protected_empty_zyx] = np.maximum(
            conditioned[self.protected_empty_zyx], self.minimum_empty_distance)
        # Center protected coordinates on the feasible conditioned field. A
        # latent zero must not sit below a hard clamp with a dead derivative.
        # Remaining movement is reduced so total change from the original seed
        # still respects the same physical residual bound.
        capacity = np.maximum(self.maximum_displacement-np.abs(conditioned-seed), 0.)
        field = conditioned.ravel().copy()
        field[self.active_flat] += capacity.ravel()[self.active_flat]*np.tanh(parameters)
        field = field.reshape(self.seed_zyx.shape)
        field[self.protected_empty_zyx] = np.maximum(
            field[self.protected_empty_zyx], self.minimum_empty_distance)
        return field

    def torch_decode(self, parameters):
        import torch
        if (parameters.device.type != 'cpu' or parameters.shape != self.active_flat.shape
                or not bool(torch.isfinite(parameters).all())):
            raise ValueError('implicit parameters require finite CPU active coordinates')
        seed = torch.tensor(self.seed_zyx.copy(), dtype=parameters.dtype)
        indices = torch.tensor(self.active_flat.copy(), dtype=torch.int64)
        protected = torch.tensor(self.protected_empty_zyx.copy(), dtype=torch.bool)
        conditioned = torch.where(protected, torch.clamp(seed, min=self.minimum_empty_distance), seed)
        capacity = torch.clamp(self.maximum_displacement-torch.abs(conditioned-seed), min=0.)
        field = conditioned.reshape(-1).clone()
        values = field[indices]+capacity.reshape(-1)[indices]*torch.tanh(parameters)
        field = field.index_copy(0, indices, values).reshape(seed.shape)
        # Explicit feasible-side subgradient at equality. The pinned Torch
        # clamp kernel returns zero there, which would strand empty-region
        # coordinates on the boundary even after feasible initialization.
        projected = torch.where(field < self.minimum_empty_distance,
                                torch.full_like(field, self.minimum_empty_distance), field)
        return torch.where(protected, projected, field)

    def report(self):
        constraint_hash = hashlib.sha256(
            b"bounded_narrow_band_implicit_residual_v2"
            +self.seed_field_hash.encode()+self.protected_empty_zyx.tobytes()
            +np.asarray([self.band_width, self.maximum_displacement,
                         self.minimum_empty_distance], dtype=np.float64).tobytes()).hexdigest()
        return {'coordinate_model': 'bounded_narrow_band_implicit_residual_v2',
                'layout': 'z,y,x; world-increasing voxel-center axes',
                'shape': list(self.seed_zyx.shape), 'active_parameters': len(self.active_flat),
                'logical_cells': int(self.seed_zyx.size), 'seed_field_hash': self.seed_field_hash,
                'constraint_hash': constraint_hash,
                'voxel_size_xyz': self.voxel_size_xyz.tolist(), 'origin_xyz': self.origin_xyz.tolist(),
                'band_width_world': self.band_width,
                'maximum_displacement_world': self.maximum_displacement,
                'bound_scope': 'signed-field residual values; final surface error also depends on grid interpolation',
                'minimum_empty_distance_world': self.minimum_empty_distance,
                'known_empty_and_domain_boundary_cells': int(self.protected_empty_zyx.sum()),
                'storage_scope': 'dense base field; only residual coordinates are narrow-band',
                'topology_changes_possible': True, 'native_qualification': False}
