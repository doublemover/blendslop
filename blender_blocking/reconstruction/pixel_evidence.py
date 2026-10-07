"""Observed probability and reliability maps without collapsing to view means."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PixelEvidence:
    foreground: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    boundary_reliability: np.ndarray
    hard_mask: np.ndarray | None = None

    @property
    def weights(self):return self.valid*self.confidence*self.boundary_reliability


def observed_pixel_evidence(constraint):
    from .visibility import valid_evidence
    mask=np.asarray(getattr(constraint.mask,'mask',constraint.mask),bool)
    valid=valid_evidence(constraint).copy()
    uncertainty=getattr(constraint,'uncertainty',None)
    def field(name,default):
        raw=getattr(uncertainty,name,None)
        values=np.asarray(default if raw is None else raw,float)
        if values.ndim==0:values=np.full(mask.shape,float(values))
        if values.shape!=mask.shape:
            raise ValueError(f'{constraint.view}: spatial {name} must match original camera pixels')
        if not np.isfinite(values[valid]).all() or np.any(values[valid]<0.) or np.any(values[valid]>1.):
            raise ValueError(f'{constraint.view}: observed {name} must be finite in [0,1]')
        return np.where(valid,values,0.)
    foreground=field('foreground_prob',mask.astype(float))
    confidence=field('confidence',np.ones(mask.shape))
    boundary=field('boundary_uncertainty',np.zeros(mask.shape))
    reliability=np.where(valid,1.-boundary,0.)
    hard=mask&valid
    for values in (foreground,valid,confidence,reliability,hard):values.setflags(write=False)
    return PixelEvidence(foreground,valid,confidence,reliability,hard)


def resize_pixel_evidence(evidence,shape):
    """Area-resample numerators and reliability; never interpolate unknown zeros."""
    import cv2
    height,width=map(int,shape)
    if min(height,width)<=0:raise ValueError('pixel evidence output must be nonempty')
    area=lambda values:cv2.resize(np.asarray(values,float),(width,height),interpolation=cv2.INTER_AREA)
    observed=area(evidence.valid)
    weighted=area(evidence.weights)
    probability=area(evidence.foreground*evidence.weights)
    fallback=area(evidence.foreground*evidence.valid)
    hard=area((evidence.hard_mask if evidence.hard_mask is not None else evidence.foreground>=.5)*evidence.valid)
    reference=np.divide(probability,weighted,out=np.zeros_like(weighted),where=weighted>0.)
    reference=np.where(weighted>0.,reference,np.divide(fallback,observed,out=np.zeros_like(observed),where=observed>0.))
    return {'foreground':reference,'observed_fraction':observed,'reliability_weight':weighted,
        'hard_foreground':np.divide(hard,observed,out=np.zeros_like(observed),where=observed>0.),
        'scope':'original pixel probability times confidence and boundary reliability; unknown cells omitted'}
