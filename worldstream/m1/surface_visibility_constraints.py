"""Conditional visible-ray support from actual distinct acquisitions.

No fitted surface, box ownership, future frame, LiDAR, label or qualification is
used. A failed patch probe is unknown visibility, never proof of occlusion.
Multiple supported source rays are ambiguity, not a residual-ranking problem.
"""
from collections import Counter
from time import perf_counter
import numpy as np
from .surface_observability import (PatchObservation, alternative_rays,
    baseline_angle, corroborating_views, image_contract, measured_world, project)


def depth_order_conflicts(world, camera, eligible, margin):
    """Reject farther measured hypotheses sharing an observed source pixel.

    This conditional z-buffer tests incompatible explanations. It does not
    identify the true occluder, certify foreground ownership or fill a surface.
    """
    result=np.zeros(len(world),bool)
    if not np.any(eligible):return result
    uv,z=project(world,camera);h,w=camera.image.shape[:2]
    valid=(eligible & np.isfinite(uv).all(1) & np.isfinite(z) & (z>.5)
        & (uv[:,0]>=0) & (uv[:,0]<w-.5) & (uv[:,1]>=0) & (uv[:,1]<h-.5))
    idx=np.flatnonzero(valid)
    if not len(idx):return result
    xy=np.rint(uv[idx]).astype(int);cells=xy[:,1]*w+xy[:,0]
    depth=np.full(h*w,np.inf);np.minimum.at(depth,cells,z[idx])
    result[idx]=z[idx]>depth[cells]+margin
    return result


def candidate_masks(measurements):
    """Predeclared logical ablations of identical, label-free measurements.

    Each uses the same complete-neighbourhood base, exclusions and sources.
    These are observation masks, never a physical or neural reuse certificate.
    """
    m=measurements;direct=m['source_consistent'];extra=m['temporal_supported']|m['camera_supported']
    clear=~m['ambiguous'];visible=~m['order_conflict']
    return dict(surface_constraints=direct&extra&clear&visible,
        constraints_two_view_only=direct&clear,
        constraints_temporal_only=direct&m['temporal_supported']&clear&visible,
        constraints_camera_only=direct&m['camera_supported']&clear&visible,
        constraints_no_ambiguity=direct&extra&visible,
        constraints_no_depth_order=direct&extra&clear,
        constraints_observation_base=m['observation_base'].copy())


def apply_surface_constraints(previous,current,evidence,cameras,history,config):
    """Restrict proposed pixels using actual support and measured ambiguity.

    The finite aperture is the existing +/- search radius, sampled every half
    pixel; incompatible means >2 px, the existing correspondence tolerance.
    A lack of a measured alternative is NOT a proof of global uniqueness.
    """
    start=perf_counter();shape=current.image.shape[:2]
    base=evidence.stable_mask.copy()
    names=('source_consistent','temporal_supported','camera_supported','ambiguous','order_conflict')
    maps={k:np.zeros(shape,bool) for k in names};maps['observation_base']=base
    views,rejected=corroborating_views(current,cameras,history,config)
    counts=Counter();view_stats=[]
    legal=previous is not None and image_contract(current,previous,config,temporal=True) is None
    if legal and evidence.dense_source_map is not None:
        indices=np.flatnonzero(base);yy,xx=np.unravel_index(indices,shape)
        target=np.c_[xx,yy].astype(float);source=evidence.dense_source_map[yy,xx]
        world=np.full((len(indices),3),np.nan)
        per_view=[np.zeros(len(indices),bool) for _ in views]
        patch=PatchObservation(current,config);old_patch=PatchObservation(previous,config)
        others=[PatchObservation(view,config) for _,view in views]
        view_counts=[Counter() for _ in views]
        # Bounded chunks keep OpenCV's remap height below SHRT_MAX, including
        # all 17 alternatives. No subsampling of the candidate population.
        for lo in range(0,len(indices),256):
            sl=slice(lo,lo+256);ids=indices[sl];p=source[sl];q=target[sl]
            measured=measured_world(previous,current,p,q);xyz=measured['world'];world[sl]=xyz
            direct=patch.compare(old_patch,q,xyz)
            ok=measured['valid']&direct['supported']
            maps['source_consistent'].flat[ids]=ok
            for vi,((kind,view),other) in enumerate(zip(views,others)):
                obs=patch.compare(other,q,xyz)
                usable=measured['valid']&(baseline_angle(xyz,current,view)>=config.min_triangulation_angle_deg)&obs['observed']
                supported=usable&obs['supported']&ok
                key='temporal_supported' if kind=='additional_time' else 'camera_supported'
                maps[key].flat[ids]|=supported;per_view[vi][sl]=supported
                view_counts[vi].update(in_fov=int(obs['observed'].sum()),geometrically_usable=int(usable.sum()),supported=int(supported.sum()))
            # Measure alternatives for *all* directly supported points so that
            # two-view ablation uses the same ambiguity test. No GT selection.
            ii=np.flatnonzero(ok)
            if len(ii):
                ap,aq,aw,av=alternative_rays(previous,current,p[ii],q[ii],config)
                alt=patch.compare(old_patch,aq.reshape(-1,2),aw.reshape(-1,3))
                supported=alt['supported'].reshape(av.shape)&av
                incompatible=supported&(np.linalg.norm(ap-p[ii,None,:],axis=2)>2.)
                maps['ambiguous'].flat[ids[ii]]=incompatible.any(1)
                counts['supported_alternative_hypotheses']+=int(incompatible.sum())
        extra=(maps['temporal_supported']|maps['camera_supported']).ravel()[indices]
        conflicts=depth_order_conflicts(world,previous,extra,config.occlusion_depth_margin_m)
        for (kind,view),supported,vc in zip(views,per_view,view_counts):
            conflicts|=depth_order_conflicts(world,view,supported,config.occlusion_depth_margin_m)
            view_stats.append(dict(kind=kind,camera_id=view.camera_id,timestamp=view.timestamp,**vc))
        maps['order_conflict'].flat[indices]=conflicts
    masks=candidate_masks(maps)
    evidence.observation_supported_mask=base
    evidence.visibility_consistent_mask=maps['source_consistent']&(maps['temporal_supported']|maps['camera_supported'])&~maps['order_conflict']
    # An incompatible source explanation is ambiguous only if the proposed
    # original explanation also has actual source support (enforced above).
    evidence.ambiguity_rejected_mask=maps['ambiguous']|maps['order_conflict']
    evidence.stable_candidate_mask=masks['surface_constraints'].copy()
    evidence.stable_mask=evidence.stable_candidate_mask.copy()
    evidence.refresh_required_mask=~evidence.stable_mask
    evidence.constraint_measurements=maps
    evidence.source='actual_distinct_acquisition_visible_ray_support_conditional'
    funnel=evidence.diagnostics.setdefault('acceptance_funnel',{})
    live=base.copy()
    for name,condition in [('projective_source_observation',maps['source_consistent']),
            ('actual_additional_acquisition',maps['temporal_supported']|maps['camera_supported']),
            ('observed_alternative_rejection',~maps['ambiguous']),
            ('measured_source_depth_order',~maps['order_conflict'])]:
        before=int(live.sum());live &= condition
        funnel[name]=dict(total_pixels=base.size,standalone_population=int(base.sum()),
            alone_rejected=int((base&~condition).sum()),incremental_rejected=before-int(live.sum()),retained=int(live.sum()))
    evidence.diagnostics['surface_constraints']=dict(
        elapsed_ms=(perf_counter()-start)*1000,legal_source=legal,
        unavailable_views=rejected,actual_views=view_stats,
        mask_counts={k:int(v.sum()) for k,v in maps.items()},**counts,
        claim='conditional_surface_observation_only',physical_certification=False,
        alternative_scope='finite measured epipolar aperture; not global uniqueness',
        depth_order_scope='conditional measured hypotheses; true occluder and ownership unknown')
