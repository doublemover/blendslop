"""Bounded three-box proposals from observed stepped silhouettes only."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .frozen_family import _camera, coverage_contour_points, validate_family_budget

FIT_VIEWS = ("front", "side", "top", "oblique_35_28")
HELD_OUT_VIEW = "oblique_145_40"
_CORNERS = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


def projected_box_polygon(lower, upper, camera):
    """World camera-plane convex polygon; generic proper orthographic basis."""
    from scipy.spatial import ConvexHull
    matrix, _ = _camera(camera)
    points = (np.asarray(lower)+np.asarray(upper))*.5 + _CORNERS*(np.asarray(upper)-lower)*.5
    xy = (points-matrix[:3, 3]) @ matrix[:3, :2]
    xy = np.unique(xy, axis=0)
    return xy[ConvexHull(xy).vertices]


def polygon_signed_distance(points, polygon):
    """Vectorized exact segment distance and CCW convex containment."""
    start = np.asarray(polygon)
    edges = np.roll(start, -1, axis=0)-start
    delta = np.asarray(points)[:, None, :]-start
    fraction = np.clip(np.einsum('nej,ej->ne', delta, edges)/(edges*edges).sum(axis=1), 0., 1.)
    nearest = delta-fraction[..., None]*edges
    distance = np.sqrt((nearest*nearest).sum(axis=2).min(axis=1))
    cross = edges[None, :, 0]*delta[:, :, 1]-edges[None, :, 1]*delta[:, :, 0]
    inside = (cross >= -1e-12).all(axis=1)
    return np.where(inside, -distance, distance)


def _interval(values):
    values = np.asarray(values, float)
    ids = np.flatnonzero(values >= .5)
    if not len(ids) or ids[0] == 0 or ids[-1] == len(values)-1:
        raise ValueError("step interval must be nonempty and uncensored")
    a, b = int(ids[0]), int(ids[-1])
    if not (values[a:b+1] >= .5).all():
        raise ValueError("step interval must be contiguous")
    lower = a-.5+(.5-values[a-1])/(values[a]-values[a-1])
    upper = b+.5+(.5-values[b])/(values[b+1]-values[b])
    return float(lower), float(upper)


def _world_interval(coverage, row, camera, axis):
    a, b = _interval(coverage[int(row)])
    matrix, scale = _camera(camera)
    width = coverage.shape[1]
    return np.array([(a/width-.5)*scale, (b/width-.5)*scale])+matrix[axis, 3]


def _pixel_row(world_z, coverage, camera):
    matrix, scale = _camera(camera)
    return int(np.clip(round((.5-(world_z-matrix[2, 3])/scale)*coverage.shape[0]-.5), 0, coverage.shape[0]-1))


def observed_box_seed(coverage, cameras):
    """Measured base/two-arm decomposition, retaining the occluded Y interval."""
    for view, axes in (("front", (0, 2)), ("side", (1, 2)), ("top", (0, 1))):
        matrix, _ = _camera(cameras[view])
        if any(not np.allclose(matrix[:3, j], np.eye(3)[axis], atol=1e-5) for j, axis in enumerate(axes)):
            raise ValueError("stepped-box seed needs canonical front/side/top bases")
    front = np.asarray(coverage["front"], float)
    side = np.asarray(coverage["side"], float)
    top = np.asarray(coverage["top"], float)
    for image in (front, side, top):
        coverage_contour_points(image)  # Reject clipped, nonfinite or out-of-range source coverage.
    matrix, scale = _camera(cameras["front"])
    height, width = front.shape
    occupied = np.flatnonzero((front >= .9).any(axis=0))
    upper_edges = np.array([_interval(front[:, x])[0] for x in occupied])
    lower_edges = np.array([_interval(front[:, x])[1] for x in occupied])
    bottom = float(np.median(lower_edges))
    if np.ptp(lower_edges) > 2.:
        raise ValueError("stepped-box base has no observed common bottom")
    base_top_pixel = float(np.max(upper_edges))
    arm_columns = occupied[upper_edges < base_top_pixel-4.]
    groups = np.split(arm_columns, np.flatnonzero(np.diff(arm_columns) > 1)+1)
    groups = [g for g in groups if len(g) >= 6]
    if len(groups) != 2:
        raise ValueError("stepped-box proposal needs exactly two observed raised plateaus")
    front_points = coverage_contour_points(front)
    base_x = np.array([(front_points[:, 0].min()/width-.5)*scale,
                       (front_points[:, 0].max()/width-.5)*scale])+matrix[0, 3]
    z_bottom = (.5-bottom/height)*scale+matrix[2, 3]
    z_base = (.5-base_top_pixel/height)*scale+matrix[2, 3]
    if z_base <= z_bottom:
        raise ValueError("base has no positive observed height")
    top_points = coverage_contour_points(top)
    top_matrix, top_scale = _camera(cameras["top"])
    base_y = np.array([(.5-top_points[:, 1].max()/top.shape[0])*top_scale,
                       (.5-top_points[:, 1].min()/top.shape[0])*top_scale])+top_matrix[1, 3]
    arms = []
    for group in groups:
        interior = group[2:-2]
        edge = float(np.median([_interval(front[:, x])[0] for x in interior]))
        if np.ptp([_interval(front[:, x])[0] for x in interior]) > 2.:
            raise ValueError("arm top is not an observed horizontal plateau")
        z_top = (.5-edge/height)*scale+matrix[2, 3]
        row = _pixel_row((z_base+z_top)*.5, front, cameras["front"])
        # Isolate the observed plateau run; the other arm may share this height.
        values = front[row].copy()
        values[:max(0, int(group[0])-2)] = 0.
        values[min(width, int(group[-1])+3):] = 0.
        arms.append({'x':_world_interval(values[None, :], 0, cameras['front'], 0), 'top':z_top})
    arms.sort(key=lambda arm: arm['top'], reverse=True)
    tall, short = arms
    if tall['top']-short['top'] < 4.*scale/height:
        raise ValueError("side exposure needs two distinct observed arm heights")
    tall_row = _pixel_row((tall['top']+short['top'])*.5, side, cameras['side'])
    tall_y = _world_interval(side, tall_row, cameras['side'], 1)
    short_row = _pixel_row((short['top']+z_base)*.5, side, cameras['side'])
    combined_y = _world_interval(side, short_row, cameras['side'], 1)
    _, side_scale = _camera(cameras['side'])
    pixel = side_scale/side.shape[1]
    near_exposed = combined_y[0] < tall_y[0]-2.*pixel
    far_exposed = combined_y[1] > tall_y[1]+2.*pixel
    if not near_exposed or far_exposed:
        raise ValueError("initial multipart scope needs an exposed short-arm near edge and occluded far edge")
    short_y = np.array([combined_y[0], (tall_y[0]+tall_y[1])*.5])
    lower = np.array([tall_y[0]-2*pixel, tall_y[1]-2*pixel, short_y[0]-2*pixel,
                      max(short_y[0]+4*pixel, base_y[0])])
    upper = np.array([tall_y[0]+2*pixel, tall_y[1]+2*pixel, short_y[0]+2*pixel,
                      min(tall_y[1], base_y[1])])
    lower[:3] = np.maximum(lower[:3], np.array([base_y[0],base_y[0],base_y[0]]))
    upper[:3] = np.minimum(upper[:3], np.array([base_y[1],base_y[1],base_y[1]]))
    if (upper <= lower).any():
        raise ValueError("observed endpoint intervals have no positive allowance")
    initial = np.clip(np.r_[tall_y,short_y], lower+1e-10, upper-1e-10)
    overlap = .1*(z_base-z_bottom)
    return {'base_x':base_x,'base_y':base_y,'bottom':z_bottom,'base_top':z_base,
            'tall_x':tall['x'],'short_x':short['x'],'tall_top':tall['top'],'short_top':short['top'],
            'join_overlap':overlap,'initial':initial,'lower':lower,'upper':upper,
            'short_far_interval': [float(lower[3]),float(upper[3])],
            'pixel_world':pixel,'short_far_evidence':'side occlusion gives an interval; oblique_35_28 may identify endpoint'}


def boxes_from_endpoints(seed, endpoints):
    tall_lo,tall_hi,short_lo,short_hi = endpoints
    return [(np.array([seed['base_x'][0],seed['base_y'][0],seed['bottom']]),
             np.array([seed['base_x'][1],seed['base_y'][1],seed['base_top']])),
            (np.array([seed['tall_x'][0],tall_lo,seed['base_top']-seed['join_overlap']]),
             np.array([seed['tall_x'][1],tall_hi,seed['tall_top']])),
            (np.array([seed['short_x'][0],short_lo,seed['base_top']-seed['join_overlap']]),
             np.array([seed['short_x'][1],short_hi,seed['short_top']]))]


def fitted_multipart_program(masks, cameras, *, coverage_masks, max_evaluations=160, max_elapsed_s=3.):
    """Fit four observed Y endpoints; held-out pixels are never accessed."""
    from scipy.optimize import least_squares
    from primitives.shape_program import ShapeNode,ShapeProgram
    validate_family_budget(max_evaluations,max_elapsed_s)
    if not 16 <= max_evaluations <= 160 or max_elapsed_s > 3.:
        raise ValueError("multipart fit requires 16..160 evaluations and at most three seconds")
    if not all(view in masks and view in cameras and view in coverage_masks for view in FIT_VIEWS):
        raise ValueError("multipart fit needs explicit four-view coverage and cameras")
    started=time.perf_counter()
    seed=observed_box_seed(coverage_masks,cameras)
    observations=[]; censored_views=[]
    span=max(seed['base_x'][1]-seed['base_x'][0],seed['tall_top']-seed['bottom'])
    for view in FIT_VIEWS:
        image=np.asarray(coverage_masks[view],float)
        foreground=image>=.5
        clipped=bool(foreground[0].any() or foreground[-1].any() or foreground[:,0].any() or foreground[:,-1].any())
        if clipped: censored_views.append(view)
        points=coverage_contour_points(image,allow_clipped=(view=="oblique_35_28"))
        # Equal per-view weights, deterministic subsampling, unchanged full image coordinates.
        points=points[np.linspace(0,len(points)-1,min(384,len(points)),dtype=int)]
        matrix,scale=_camera(cameras[view]); h,w=image.shape
        uv=np.column_stack(((points[:,0]/w-.5)*scale,(.5-points[:,1]/h)*scale))
        yy,xx=np.mgrid[:20,:20]
        x=(xx.ravel()+.5)*w/20; y=(yy.ravel()+.5)*h/20
        known=image[np.minimum(y.astype(int),h-1),np.minimum(x.astype(int),w-1)]
        keep=(known<.05)|(known>.95)
        grid=np.column_stack(((x/w-.5)*scale,(.5-y/h)*scale))[keep]
        observations.append((view,uv,grid,known[keep]>.5))
    def vector(endpoints):
        boxes=boxes_from_endpoints(seed,endpoints); output=[]
        for view,contour,grid,inside in observations:
            points=np.vstack((contour,grid))
            values=np.min([polygon_signed_distance(points,projected_box_polygon(a,b,cameras[view])) for a,b in boxes],axis=0)
            output.append(values[:len(contour)]/(span*np.sqrt(len(contour))))
            violation=np.where(inside,np.maximum(values[len(contour):],0.),np.minimum(values[len(contour):],0.))
            output.append(violation*.25/(span*np.sqrt(len(grid))))
        return np.concatenate(output)
    calls=0; best=seed['initial'].copy(); score=float('inf'); termination='solver_completed'
    class AllowanceEnded(Exception): pass
    def residual(x):
        nonlocal calls,best,score,termination
        if calls >= max_evaluations-10 or (calls and time.perf_counter()-started >= max_elapsed_s):
            termination='evaluation_allowance' if calls>=max_evaluations-10 else 'elapsed_allowance'; raise AllowanceEnded
        calls+=1; result=vector(x); current=float(result@result)
        if current<score: score,best=current,x.copy()
        return result
    try:
        least_squares(residual,best,bounds=(seed['lower'],seed['upper']),diff_step=1e-4,
                      max_nfev=max_evaluations-10,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac')
    except AllowanceEnded: pass
    # Local observability is independent of quality acceptance. Never consult holdout.
    identifiability={'status':'unqualified','parameter_names':['tall_y_min','tall_y_max','short_y_min','short_y_max'],
                     'original_short_far_interval':seed['short_far_interval'],'scope':'local four-view projected residual rank; no global uniqueness claim'}
    if calls+9<=max_evaluations and time.perf_counter()-started<max_elapsed_s:
        columns=[]
        for index in range(4):
            if time.perf_counter()-started>=max_elapsed_s:
                break
            low,high=best.copy(),best.copy(); delta=seed['pixel_world']*.1
            low[index]=max(seed['lower'][index],best[index]-delta); high[index]=min(seed['upper'][index],best[index]+delta)
            columns.append((vector(high)-vector(low))/(high[index]-low[index])); calls+=2
        if len(columns)==4:
            singular=np.linalg.svd(np.column_stack(columns),compute_uv=False)
            rank=int(np.count_nonzero(singular>max(float(singular[0])*1e-8,1e-10)))
            active_far=bool(min(best[3]-seed['lower'][3],seed['upper'][3]-best[3])<=seed['pixel_world']*.25)
            identifiability.update(status='locally_identified' if rank==4 and not active_far else 'underconstrained',
                                   local_rank=rank,singular_values=singular.tolist(),fitted_short_far=float(best[3]),
                                   short_far_interval_bound_active=active_far,
                                   uncertainty_reason='hidden endpoint at an interval bound; retain original interval' if active_far else None)
    nodes=[]
    for name,(lower,upper) in zip(('base','tall_arm','short_arm'),boxes_from_endpoints(seed,best)):
        center=(lower+upper)*.5; dimensions=upper-lower
        nodes.append(ShapeNode('observed_'+name,'add','box',parameters={
            **dict(zip(('x','y','z'),center.tolist())),**dict(zip(('width_world','depth_world','height_world'),dimensions.tolist())),
            'rotation':np.eye(3).tolist()},name=name))
    metadata={'proposal':'observed_three_box_four_view_fit','fit_views':list(FIT_VIEWS),'held_out_view':HELD_OUT_VIEW,
              'support_evaluations':calls,'support_elapsed_s':time.perf_counter()-started,'support_squared_residual':score if np.isfinite(score) else None,
              'termination':termination,'identifiability':identifiability,'fit_status':identifiability['status'],
              'censored_views':censored_views,
              'censored_contour_scope':'observed internal half-coverage crossings only; no frame-edge closure or outside-frame support',
              'camera_sha256':hashlib.sha256(json.dumps({v:cameras[v] for v in FIT_VIEWS},sort_keys=True).encode()).hexdigest(),
              'join_overlap_world':seed['join_overlap'],'join_scope':'concealed overlap within measured base is a construction choice',
              'node_budget':3,'free_endpoint_count':4,'live_exact_union_modifiers':2,'source_geometry_read':False,
              'full_five_view_admission':'unrun; includes untouched oblique_145_40'}
    return ShapeProgram('1','frozen-family-asymmetric_multipart_solid',tuple(nodes),metadata=metadata)


def retained_multipart_program(wire):
    """Restore the exact saved bounded three-leaf recipe without refitting."""
    from primitives.shape_program import ShapeNode,ShapeProgram
    roots=wire.get('root_nodes',())
    if wire.get('constraints') or wire.get('residual_patches') or len(roots)!=3:
        raise ValueError('retained multipart recipe needs exactly three unconstrained boxes')
    nodes=[]
    for node in roots:
        if node.get('operation')!='add' or node.get('primitive_type')!='box' or node.get('children'):
            raise ValueError('retained multipart recipe needs three additive box leaves')
        nodes.append(ShapeNode(node['node_id'],'add','box',parameters=dict(node['parameters']),
                               name=node.get('name',''),editable=node.get('editable',True)))
    return ShapeProgram(wire['schema_version'],wire['program_id'],tuple(nodes),metadata=dict(wire.get('metadata',{})))
