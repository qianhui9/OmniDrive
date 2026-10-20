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
    result=dict(surface_constraints=direct&extra&clear&visible,
        constraints_two_view_only=direct&clear,
        constraints_temporal_only=direct&m['temporal_supported']&clear&visible,
        constraints_camera_only=direct&m['camera_supported']&clear&visible,
        constraints_no_ambiguity=direct&extra&visible,
        constraints_no_depth_order=direct&extra&clear,
        constraints_observation_base=m['observation_base'].copy())
    if 'adaptive_source_consistent' in m:
        a=m['adaptive_source_consistent'];v=m['adaptive_temporal_supported']|m['adaptive_camera_supported']
        c=~m['adaptive_ambiguous'];order=~m['adaptive_order_conflict'];joint=~m['joint_ambiguous']
        result.update(geometric_adaptive_patch=a&v&c&order,
            joint_multi_observation=direct&extra&~m['joint_legacy_ambiguous']&visible,
            correspondence_recovery=a&v&joint&order,
            recovery_no_extra_observation=a&c,
            recovery_h0_only=a&v&order,
            recovery_no_depth_order=a&v&joint,
            recovery_discriminating_third_only=a&v&joint&order&m['joint_discriminating'],
            recovery_temporal_only=a&m['adaptive_temporal_supported']&joint&order,
            recovery_camera_only=a&m['adaptive_camera_supported']&joint&order)
    return result


def joint_exclusions(target, original_world, alternative_world, alternatives, views,
                     patch, geometry, previous, source, config):
    """Eliminate only with a measured, locally validated third-view ray conflict.

    Patch failure, off-image projection and missing tracking never exclude a
    hypothesis. An actually supported alternative always survives that view.
    The finite support footprint does not establish global physical ownership.
    """
    from .geometry_adaptive_patch import measured_third_correspondence
    n,k=alternatives.shape
    excluded=np.zeros((n,k),bool);corroborated=np.zeros((n,k),bool)
    discriminating=np.zeros(n,bool);counts=Counter()
    for kind,view,other,original_supported in views:
        uv,_=project(original_world,view)
        track=measured_third_correspondence(patch.camera,view,target,uv,config)
        p3,z3=project(alternative_world,view)
        alt_obs=patch.compare(other,np.broadcast_to(target[:,None,:],(n,k,2)).reshape(-1,2),
            alternative_world.reshape(-1,3))
        observed=alt_obs['observed'].reshape(n,k)
        support=alt_obs['supported'].reshape(n,k)
        corroborated |= alternatives&support
        measured_source,_=project(track['world'],previous)
        source_cycle=np.linalg.norm(measured_source-source,axis=1)<=.5
        reliable=track['supported']&original_supported&source_cycle
        # Verify the measured local plane before using a track to exclude
        # competitors. Otherwise visibility/ownership remains unknown.
        if geometry is None: reliable[:]=False
        else: reliable &= geometry['applicable']
        separation=np.linalg.norm(p3-track['source_uv'][:,None,:],axis=2)
        # Only the actual tracked local footprint can furnish a contradiction;
        # merely being elsewhere in the image is not known visibility.
        in_footprint=separation<=config.support_patch_radius_px
        usable=(reliable[:,None]&observed&(z3>.5)&(z3<200)&in_footprint
            &(baseline_angle(alternative_world,patch.camera,view)>=config.min_triangulation_angle_deg))
        contradiction=alternatives&usable&(separation>2.)&~support
        excluded |= contradiction
        discriminating |= (alternatives&usable&(separation>2.)).any(1)
        counts['actual_third_view_tracks']+=int(track['supported'].sum())
        counts['geometrically_validated_third_view_tracks']+=int(reliable.sum())
        counts['per_view_contradictions']+=int(contradiction.sum())
        counts['simultaneously_supported_alternatives']+=int((alternatives&support).sum())
    # Conflicting actual views do not justify choosing a preferred explanation.
    excluded &= ~corroborated
    return excluded,discriminating,counts


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
    adaptive=config.patch_geometry_model=='local_inverse_depth' or config.joint_observation_disambiguation
    if adaptive:
        for k in ('adaptive_source_consistent','adaptive_temporal_supported','adaptive_camera_supported',
                  'adaptive_ambiguous','adaptive_order_conflict','joint_ambiguous','joint_legacy_ambiguous',
                  'joint_discriminating','joint_eliminated_any','patch_model_applicable','patch_geometry_recovered'):
            maps[k]=np.zeros(shape,bool)
    views,rejected=corroborating_views(current,cameras,history,config)
    counts=Counter();view_stats=[]
    legal=previous is not None and image_contract(current,previous,config,temporal=True) is None
    if legal and evidence.dense_source_map is not None:
        indices=np.flatnonzero(base);yy,xx=np.unravel_index(indices,shape)
        target=np.c_[xx,yy].astype(float);source=evidence.dense_source_map[yy,xx]
        world=np.full((len(indices),3),np.nan)
        per_view=[np.zeros(len(indices),bool) for _ in views]
        patch=PatchObservation(current,config);old_patch=PatchObservation(previous,config)
        if adaptive:
            from .geometry_adaptive_patch import GeometryAdaptivePatch
            adaptive_patch=GeometryAdaptivePatch(current,config)
            adaptive_per_view=[np.zeros(len(indices),bool) for _ in views]
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
            geometry=None;new_ok=ok
            if adaptive:
                geometry=adaptive_patch.fit(old_patch,q,xyz,evidence.dense_source_map,base)
                adapted=adaptive_patch.compare(old_patch,q,xyz,geometry)
                new_ok=measured['valid']&adapted['supported'] if config.patch_geometry_model=='local_inverse_depth' else ok
                maps['adaptive_source_consistent'].flat[ids]=new_ok
                maps['patch_model_applicable'].flat[ids]=geometry['applicable']
                maps['patch_geometry_recovered'].flat[ids]=new_ok&~ok
            joint_views=[]
            for vi,((kind,view),other) in enumerate(zip(views,others)):
                obs=patch.compare(other,q,xyz)
                usable=measured['valid']&(baseline_angle(xyz,current,view)>=config.min_triangulation_angle_deg)&obs['observed']
                supported=usable&obs['supported']&ok
                key='temporal_supported' if kind=='additional_time' else 'camera_supported'
                maps[key].flat[ids]|=supported;per_view[vi][sl]=supported
                view_counts[vi].update(in_fov=int(obs['observed'].sum()),geometrically_usable=int(usable.sum()),supported=int(supported.sum()))
                if adaptive:
                    adapted_obs=adaptive_patch.compare(other,q,xyz,geometry)
                    new_supported=usable&adapted_obs['supported']&new_ok
                    maps['adaptive_'+key].flat[ids]|=new_supported
                    adaptive_per_view[vi][sl]=new_supported
                    joint_views.append((kind,view,other,new_supported))
            # Measure alternatives for *all* directly supported points so that
            # two-view ablation uses the same ambiguity test. No GT selection.
            ii=np.flatnonzero(new_ok)
            if len(ii):
                ap,aq,aw,av=alternative_rays(previous,current,p[ii],q[ii],config)
                alt=patch.compare(old_patch,aq.reshape(-1,2),aw.reshape(-1,3))
                supported=alt['supported'].reshape(av.shape)&av
                incompatible=supported&(np.linalg.norm(ap-p[ii,None,:],axis=2)>2.)
                maps['ambiguous'].flat[ids[ii]]=incompatible.any(1)&ok[ii]
                counts['supported_alternative_hypotheses']+=int(incompatible.sum())
                if adaptive:
                    # Keep every legacy-supported explanation; adaptation can
                    # add competitors but never erase them using the same flow.
                    alt_geom=dict(geometry)
                    alt_geom['slopes']=np.repeat(geometry['slopes'][ii],av.shape[1],axis=0)
                    alt_geom['applicable']=np.repeat(geometry['applicable'][ii],av.shape[1])
                    alt_adapt=adaptive_patch.compare(old_patch,aq.reshape(-1,2),aw.reshape(-1,3),alt_geom)
                    new_alternatives=av&alt_adapt['supported'].reshape(av.shape)&(np.linalg.norm(ap-p[ii,None,:],axis=2)>2.)
                    maps['adaptive_ambiguous'].flat[ids[ii]]=new_alternatives.any(1)
                    excluded=np.zeros_like(new_alternatives);distinguishing=np.zeros(len(ii),bool)
                    if config.joint_observation_disambiguation and new_alternatives.any():
                        selected_geom=dict(slopes=geometry['slopes'][ii],applicable=geometry['applicable'][ii])
                        jviews=[(kind,view,other,sup[ii]) for kind,view,other,sup in joint_views]
                        excluded,distinguishing,jcounts=joint_exclusions(q[ii],xyz[ii],aw,new_alternatives,jviews,
                            patch,selected_geom,previous,p[ii],config)
                        counts.update(jcounts)
                    maps['joint_ambiguous'].flat[ids[ii]]=(new_alternatives&~excluded).any(1)
                    maps['joint_legacy_ambiguous'].flat[ids[ii]]=(incompatible&~excluded).any(1)&ok[ii]
                    maps['joint_discriminating'].flat[ids[ii]]=distinguishing
                    maps['joint_eliminated_any'].flat[ids[ii]]=excluded.any(1)
                    counts['excluded_hypotheses']+=int(excluded.sum())
                    counts['unresolved_hypotheses']+=int((new_alternatives&~excluded).sum())
        extra=(maps['temporal_supported']|maps['camera_supported']).ravel()[indices]
        conflicts=depth_order_conflicts(world,previous,extra,config.occlusion_depth_margin_m)
        for (kind,view),supported,vc in zip(views,per_view,view_counts):
            conflicts|=depth_order_conflicts(world,view,supported,config.occlusion_depth_margin_m)
            view_stats.append(dict(kind=kind,camera_id=view.camera_id,timestamp=view.timestamp,**vc))
        maps['order_conflict'].flat[indices]=conflicts
        if adaptive:
            new_extra=(maps['adaptive_temporal_supported']|maps['adaptive_camera_supported']).ravel()[indices]
            new_conflicts=depth_order_conflicts(world,previous,new_extra,config.occlusion_depth_margin_m)
            for (_,view),supported in zip(views,adaptive_per_view):
                new_conflicts|=depth_order_conflicts(world,view,supported,config.occlusion_depth_margin_m)
            maps['adaptive_order_conflict'].flat[indices]=new_conflicts
    masks=candidate_masks(maps)
    selected='correspondence_recovery' if adaptive and config.joint_observation_disambiguation else 'geometric_adaptive_patch' if adaptive and config.patch_geometry_model=='local_inverse_depth' else 'surface_constraints'
    evidence.observation_supported_mask=base
    evidence.visibility_consistent_mask=maps['source_consistent']&(maps['temporal_supported']|maps['camera_supported'])&~maps['order_conflict']
    # An incompatible source explanation is ambiguous only if the proposed
    # original explanation also has actual source support (enforced above).
    evidence.ambiguity_rejected_mask=(maps['joint_ambiguous']|maps['adaptive_order_conflict']) if selected=='correspondence_recovery' else (maps['adaptive_ambiguous']|maps['adaptive_order_conflict']) if selected=='geometric_adaptive_patch' else maps['ambiguous']|maps['order_conflict']
    evidence.stable_candidate_mask=masks[selected].copy()
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
    evidence.diagnostics['surface_constraints']['selected_method']=selected
    if adaptive:
        evidence.visibility_consistent_mask=maps['adaptive_source_consistent']&(maps['adaptive_temporal_supported']|maps['adaptive_camera_supported'])&~maps['adaptive_order_conflict']
        live=base.copy()
        for name,condition in [('adaptive_source_observation',maps['adaptive_source_consistent']),
                ('adaptive_additional_acquisition',maps['adaptive_temporal_supported']|maps['adaptive_camera_supported']),
                ('joint_alternative_rejection',~maps['joint_ambiguous']),
                ('adaptive_depth_order',~maps['adaptive_order_conflict'])]:
            before=int(live.sum());live &= condition
            funnel[name]=dict(total_pixels=base.size,standalone_population=int(base.sum()),
                alone_rejected=int((base&~condition).sum()),incremental_rejected=before-int(live.sum()),retained=int(live.sum()))
