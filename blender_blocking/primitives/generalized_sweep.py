"""Editable monotone local-axis sweeps with matched mesh/implicit boundary.

Cross sections remain parallel in one rigid local frame. A piecewise-linear
centerline and positive scales cannot fold that axial coordinate map. This is
not a full parallel-transport sweep around loops or a topology-changing loft.
The implicit field has the correct sign/zero set, but is not Euclidean distance.
"""
from __future__ import annotations
import numpy as np
from .primitive_protocol import MeshData,signed_power


class GeneralizedSweepPrimitive:
    def __init__(self,section_knots_normalized,*,center=(0.,0.,0.),rotation=None,
                 radii=(.5,.5,.5),section_exponent=1.):
        self.section_knots_normalized=np.asarray(section_knots_normalized,float).copy()
        self.center=np.asarray(center,float).copy()
        self.rotation=np.eye(3) if rotation is None else np.asarray(rotation,float).copy()
        self.radii=np.asarray(radii,float).copy()
        self.section_exponent=float(section_exponent)
        self.validate()

    def validate(self):
        knots=self.section_knots_normalized
        if (knots.ndim!=2 or knots.shape[1]!=5 or len(knots)<2 or len(knots)>64 or
            not np.isfinite(knots).all() or np.any(np.diff(knots[:,0])<=0.) or
            np.any(knots[:,3:]<=0.)):
            raise ValueError('sweep knots require 2..64 finite [z,cx,cy,rx,ry] rows, increasing z and positive section radii')
        if abs(knots[0,0]+.5)>1e-12 or abs(knots[-1,0]-.5)>1e-12:
            raise ValueError('normalized sweep axial endpoints must be -0.5 and 0.5')
        if (self.center.shape!=(3,) or not np.isfinite(self.center).all() or self.radii.shape!=(3,) or
            not np.isfinite(self.radii).all() or np.any(self.radii<=0.)):
            raise ValueError('sweep pose/scales must be finite and positive')
        if (self.rotation.shape!=(3,3) or not np.isfinite(self.rotation).all() or
            not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-8,rtol=0.) or
            not np.isclose(np.linalg.det(self.rotation),1.,atol=1e-8,rtol=0.)):
            raise ValueError('sweep rotation must be a proper orthonormal local frame')
        if not np.isfinite(self.section_exponent) or not .15<=self.section_exponent<=2.:
            raise ValueError('convex sweep section exponent must be in [0.15,2]')

    def sections(self,z):
        physical=self.section_knots_normalized*np.array([2*self.radii[2],2*self.radii[0],
            2*self.radii[1],2*self.radii[0],2*self.radii[1]])
        return np.column_stack([np.interp(z,physical[:,0],physical[:,column]) for column in range(1,5)])

    def sdf_batch(self,points):
        self.validate()
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('sweep field requires finite Nx3 points')
        if not len(points):return np.empty(0)
        local=(points-self.center)@self.rotation
        sections=self.sections(local[:,2]);delta=np.abs(local[:,:2]-sections[:,:2])/sections[:,2:]
        exponent=2./self.section_exponent
        # Stable Lp norm avoids overflow for high aspect ratios/cornered sections.
        maximum=delta.max(axis=1);scaled=np.divide(delta,maximum[:,None],out=np.zeros_like(delta),where=maximum[:,None]>0.)
        rho=maximum*np.power(np.sum(np.power(scaled,exponent),axis=1),1./exponent)
        lateral=(rho-1.)*sections[:,2:].min(axis=1)
        axial=np.abs(local[:,2])-self.radii[2]
        q=np.column_stack((lateral,axial))
        return np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(q.max(axis=1),0.)

    def to_mesh_data(self,resolution=32):
        self.validate();segments=max(8,int(resolution))
        # Include the exact axial knots and common cardinal cross-section points.
        segments=((segments+3)//4)*4
        theta=np.arange(segments)/segments*(2*np.pi)
        unit=np.column_stack((signed_power(np.cos(theta),self.section_exponent),
                              signed_power(np.sin(theta),self.section_exponent)))
        z=self.section_knots_normalized[:,0]*2*self.radii[2]
        sections=self.sections(z)
        xy=sections[:,None,:2]+sections[:,None,2:]*unit[None,:,:]
        local=np.column_stack((xy.reshape(-1,2),np.repeat(z,segments)))
        bottom=len(local);top=bottom+1
        local=np.vstack((local,[*sections[0,:2],z[0]],[*sections[-1,:2],z[-1]]))
        faces=[]
        for ring in range(len(z)-1):
            for index in range(segments):
                a=ring*segments+index;b=ring*segments+(index+1)%segments
                faces.append((a,b,b+segments,a+segments))
        last=(len(z)-1)*segments
        for index in range(segments):
            next_index=(index+1)%segments
            faces.extend(((bottom,next_index,index),(top,last+index,last+next_index)))
        return MeshData(local@self.rotation.T+self.center,tuple(faces))

    def to_mesh(self,resolution=32):return self.to_mesh_data(resolution)

    def mesh_field_disagreement(self,resolution=32):
        """Measured tessellation-field disagreement, not an exact distance bound."""
        mesh=self.to_mesh_data(resolution)
        faces=np.asarray([(face[0],face[index],face[index+1]) for face in mesh.faces for index in range(1,len(face)-1)])
        samples=mesh.vertices[faces].mean(axis=1)
        return {'mesh_resolution':int(resolution),'sampled_triangles':len(faces),
            'maximum_abs_vertex_field':float(np.max(np.abs(self.sdf_batch(mesh.vertices)))),
            'maximum_abs_face_centroid_field':float(np.max(np.abs(self.sdf_batch(samples)))),
            'scope':'sampled implicit-field disagreement; no global Hausdorff or Euclidean-distance certificate'}

    def sample_surface(self,n):
        if n<=0:return np.empty((0,3))
        mesh=self.to_mesh_data(48)
        faces=np.asarray([(f[0],f[i],f[i+1]) for f in mesh.faces for i in range(1,len(f)-1)])
        a,b,c=(mesh.vertices[faces[:,i]] for i in range(3))
        areas=.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
        indices=np.searchsorted(np.cumsum(areas),(np.arange(n)+.5)/n*areas.sum())
        root=np.sqrt(np.mod((np.arange(n)+.5)*.6180339887498949,1.))
        second=np.mod((np.arange(n)+.5)*.4142135623730951,1.)
        return (1.-root[:,None])*a[indices]+root[:,None]*((1.-second[:,None])*b[indices]+second[:,None]*c[indices])

    def to_dict(self):
        return {'type':'generalized_sweep','section_knots_normalized':self.section_knots_normalized.tolist(),
            'center':self.center.tolist(),'rotation':self.rotation.tolist(),'radii':self.radii.tolist(),
            'section_exponent':self.section_exponent,'field_semantics':'signed zero-set field; not Euclidean distance',
            'section_frame':'parallel sections in a proper rigid frame','topology':'single stable convex loop with caps'}

    @classmethod
    def from_program_parameters(cls,parameters,*,world=True):
        from reconstruction.program_transforms import rotation_matrix,position_vector,DIMENSION_KEYS
        return cls(parameters['section_knots_normalized'],
            radii=np.array([parameters.get(key,1.) for key in DIMENSION_KEYS],float)*.5,
            center=position_vector(parameters) if world else (0.,0.,0.),
            rotation=rotation_matrix(parameters) if world else None,
            section_exponent=parameters.get('section_exponent',1.))

    @classmethod
    def from_dict(cls,parameters):
        return cls(parameters['section_knots_normalized'],center=parameters.get('center',(0.,0.,0.)),
            rotation=parameters.get('rotation'),radii=parameters.get('radii',(.5,.5,.5)),
            section_exponent=parameters.get('section_exponent',1.))
