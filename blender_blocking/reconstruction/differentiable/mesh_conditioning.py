"""CPU deformation coordinates and reference regularizers, independent of DVX."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrilinearCage:
    seed: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    control_positions: np.ndarray

    def decode(self, displacements):
        values = np.asarray(displacements, float)
        if values.shape != self.control_positions.shape:
            raise ValueError("cage displacement shape differs from its controls")
        return self.seed + np.sum(values[self.indices] * self.weights[...,None], axis=1)

    def pullback(self, vertex_gradient):
        gradient = np.asarray(vertex_gradient, float)
        if gradient.shape != self.seed.shape:
            raise ValueError("vertex gradient shape differs from cage seed")
        controls = np.zeros_like(self.control_positions)
        np.add.at(controls,self.indices.ravel(),
                  (gradient[:,None,:]*self.weights[...,None]).reshape(-1,3))
        return controls


def trilinear_cage(vertices, *, side=4, lower=-1., upper=1.):
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("cage vertices must be a finite Nx3 array")
    if side < 2 or upper <= lower or np.any(vertices < lower) or np.any(vertices > upper):
        raise ValueError("cage grid must contain all seed vertices")
    coordinates = (vertices-lower)/(upper-lower)*(side-1)
    cell = np.minimum(np.floor(coordinates).astype(int),side-2)
    fraction = coordinates-cell
    offsets = np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)], int)
    corners = cell[:,None,:]+offsets[None,:,:]
    indices = (corners[:,:,0]*side+corners[:,:,1])*side+corners[:,:,2]
    weights = np.prod(np.where(offsets[None,:,:],fraction[:,None,:],1.-fraction[:,None,:]),axis=2)
    axis = np.linspace(lower,upper,side)
    controls = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    return TrilinearCage(vertices.copy(),indices,weights,controls)


def mesh_edges(faces):
    faces = np.asarray(faces,int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("conditioning requires triangular connectivity")
    edges = np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    return np.unique(np.sort(edges,axis=1),axis=0)


class DifferentialCoordinates:
    """Fixed uniform graph Laplacian and reusable CPU sparse linear solve."""
    def __init__(self, vertices, faces, *, strength=4.):
        from scipy.sparse import coo_matrix, diags, eye
        from scipy.sparse.linalg import splu
        vertices = np.asarray(vertices,float)
        if strength < 0. or not np.isfinite(strength):
            raise ValueError("differential strength must be finite and nonnegative")
        edges = mesh_edges(faces)
        row,col = edges.T
        adjacency = coo_matrix((np.ones(2*len(edges)),(np.r_[row,col],np.r_[col,row])),
                               shape=(len(vertices),len(vertices))).tocsr()
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
        self.matrix = (eye(len(vertices))+strength*laplacian).tocsc()
        self.factor = splu(self.matrix)
        self.seed = vertices.copy()
        self.seed_coordinates = self.matrix@vertices

    def decode(self, coordinates):
        return self.factor.solve(np.asarray(coordinates,float))

    def pullback(self, vertex_gradient):
        return self.factor.solve(np.asarray(vertex_gradient,float),trans='T')


def deformation_reference_terms(vertices, seed, faces, *, minimum_area_ratio=.1):
    """Scale-invariant edge distortion and an explicit local area barrier."""
    vertices,seed,faces = np.asarray(vertices,float),np.asarray(seed,float),np.asarray(faces,int)
    edges = mesh_edges(faces)
    old_lengths = np.linalg.norm(seed[edges[:,0]]-seed[edges[:,1]],axis=1)
    new_lengths = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    old_triangles,new_triangles = seed[faces],vertices[faces]
    old_cross = np.cross(old_triangles[:,1]-old_triangles[:,0],old_triangles[:,2]-old_triangles[:,0])
    new_cross = np.cross(new_triangles[:,1]-new_triangles[:,0],new_triangles[:,2]-new_triangles[:,0])
    old_areas,new_areas = np.linalg.norm(old_cross,axis=1),np.linalg.norm(new_cross,axis=1)
    if np.any(old_lengths <= 0.) or np.any(old_areas <= 0.):
        raise ValueError("deformation seed has collapsed edges or faces")
    cosine = np.einsum('ij,ij->i',old_cross,new_cross)/np.maximum(old_areas*new_areas,np.finfo(float).tiny)
    return {'edge_distortion':float(np.mean((new_lengths/old_lengths-1.)**2)),
            'area_barrier':float(np.mean(np.maximum(minimum_area_ratio-new_areas/old_areas,0.)**2)),
            'normal_change':float(np.mean((1.-np.clip(cosine,-1.,1.))**2))}


def torch_coordinates(seed,faces,*,mode='cage',strength=4.):
    """Torch pullbacks of the independently testable CPU coordinate operators."""
    import torch
    seed_array=np.asarray(seed,float)
    seed_tensor=torch.tensor(seed_array,dtype=torch.float32)
    if mode == 'vertices':
        parameters=seed_tensor.clone().requires_grad_()
        return parameters,lambda:parameters,{'parameterization':'vertices','variables':parameters.numel()}
    if mode == 'cage':
        cage=trilinear_cage(seed_array)
        indices=torch.tensor(cage.indices,dtype=torch.int64)
        weights=torch.tensor(cage.weights,dtype=torch.float32)
        parameters=torch.zeros((len(cage.control_positions),3),dtype=torch.float32,requires_grad=True)
        def decode():
            return seed_tensor+(parameters[indices]*weights[...,None]).sum(dim=1)
        return parameters,decode,{'parameterization':'trilinear_4x4x4_cage','variables':parameters.numel(),
                                 'seed_vertices':len(seed_array),'weights_per_vertex':8}
    if mode != 'differential':
        raise ValueError('unknown deformation parameterization')
    coordinates=DifferentialCoordinates(seed_array,faces,strength=strength)
    class SparseDecode(torch.autograd.Function):
        @staticmethod
        def forward(ctx,values):
            ctx.operator=coordinates
            return torch.as_tensor(coordinates.decode(values.detach().cpu().numpy()),dtype=values.dtype)
        @staticmethod
        def backward(ctx,gradient):
            return torch.as_tensor(ctx.operator.pullback(gradient.detach().cpu().numpy()),dtype=gradient.dtype)
    parameters=torch.tensor(coordinates.seed_coordinates,dtype=torch.float32,requires_grad=True)
    return parameters,lambda:SparseDecode.apply(parameters),{
        'parameterization':'uniform_laplacian_differential_coordinates','variables':parameters.numel(),
        'strength':strength,'factorization_reused':True,'topology_fixed':True}
