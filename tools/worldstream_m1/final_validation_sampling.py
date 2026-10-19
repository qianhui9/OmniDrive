"""Candidate-conditioned OFFLINE physical evaluation, isolated from M1."""
import hashlib
import numpy as np
from worldstream.m1.geometry import region_slices

PROTOCOL=dict(version=1,seed=1907,max_samples_per_stratum=32,
    selection='Fixed-seed uniform without replacement within actual-link origin x offline ROI-context strata',
    strata=['background_context','vehicle_ROI_context','pedestrian_other_ROI_context','boundary_ROI_context','visibility_ROI_context'],
    denominator='All actual emitted links, including revoked computation candidates. ROI context does not assert foreground ownership.',
    truth='Exact rounded current LiDAR pixel, annotated instance-local rigid source projection, depth-visible at both sensor times. No proximity labels.',
    assumptions='Annotated object treated as rigid between sample LiDAR times; residual camera/LiDAR skew remains a limitation.',
    background='No independently certified background identity: Unscorable',
    no_multisweep=True,GT_online=False,weight='stratum_population / selected_count',
    paired='Same physical link is compared across ablation methods by exact source/current coordinate bytes; retention recorded separately')


def sample_links(link,candidate,reference,frame,cid,log_id,method):
    q=link['current_points'];p=link['source_points'];n=len(q)
    if n==0:return [],[],[]
    shape=candidate.shape;xy=np.rint(q).astype(int)
    context=np.zeros(shape,np.uint8)
    for event in reference['events']:
        region=region_slices(event['region'],shape)
        label=1 if event['semantic_class'] in ('car','truck','bus','trailer','construction_vehicle') else 2
        context[region]=label
        y,x=region
        context[y.start:min(y.start+2,y.stop),x]=3;context[max(y.start,y.stop-2):y.stop,x]=3
        context[y,x.start:min(x.start+2,x.stop)]=3;context[y,max(x.start,x.stop-2):x.stop]=3
        if any(r in event['reasons'] for r in ('visibility_drop','newly_visible')):context[region]=4
    labels=context[xy[:,1],xy[:,0]]
    origins=link.get('measurement_origins')
    if origins is None:origins=np.zeros(n,np.uint8)
    eligible=candidate[xy[:,1],xy[:,0]]
    # Projected LiDAR reference lookup: nearest depth has already been checked
    # by the existing pair oracle. Duplicate references at a pixel remain
    # ambiguous if their source projections differ by more than one pixel.
    lookup={};ambiguous=set()
    for j in np.flatnonzero(reference['scorable']&reference['annotated']):
        key=tuple(reference['xy'][j]);prior=lookup.get(key)
        if prior is not None and np.linalg.norm(reference['source_uv'][prior]-reference['source_uv'][j])>1:ambiguous.add(key)
        lookup[key]=j
    projections={tuple(v) for v in reference['xy'] if 0<=v[0]<shape[1] and 0<=v[1]<shape[0]}
    rows=[];denominators=[]
    for origin in np.unique(origins):
        for label in np.unique(labels):
            ids=np.flatnonzero((origins==origin)&(labels==label))
            if not len(ids):continue
            seed=int.from_bytes(hashlib.sha256(f'{PROTOCOL["seed"]}:{frame.frame_id}:{cid}:{origin}:{label}'.encode()).digest()[:8],'little')
            selected=np.sort(np.random.default_rng(seed).choice(ids,min(PROTOCOL['max_samples_per_stratum'],len(ids)),replace=False))
            denominators.append(dict(method=method,scene=frame.scene_id,frame=frame.frame_id,camera=cid,log=log_id,
                origin=int(origin),context=PROTOCOL['strata'][label],population=len(ids),selected=len(selected),
                candidate_eligible=int(eligible[ids].sum()),weight=len(ids)/len(selected)))
            for i in selected:
                key=tuple(xy[i]);j=lookup.get(key);scorable=j is not None and key not in ambiguous
                error=float(np.linalg.norm(p[i]+xy[i]-q[i]-reference['source_uv'][j])) if scorable else None
                reason=('ambiguous_projected_surfaces' if key in ambiguous else
                    'no_LiDAR_at_exact_candidate_pixel' if key not in projections else 'depth_identity_or_visibility_reference_unavailable')
                rows.append(dict(method=method,scene=frame.scene_id,log=log_id,frame=frame.frame_id,camera=cid,
                    origin=int(origin),context=PROTOCOL['strata'][label],current=q[i],source=p[i],
                    baseline_s=frame.cameras[cid].timestamp-link['source_timestamp'],weight=len(ids)/len(selected),
                    observation_link_valid=True,reuse_candidate_eligible=bool(eligible[i]),
                    scorable=bool(scorable),correct=bool(np.isfinite(error) and error<=2) if scorable else None,
                    reference_error_px=error,moving_reference=bool(reference['moving'][j]) if scorable else None,
                    unscorable_reason=None if scorable else reason,reference='instance_rigid_depth_checked_exact_pixel' if scorable else None))
    # Candidate-conditioned exact-reference census supplements the random
    # sample without pretending it has the sample's inclusion probabilities.
    census=[]
    flat=xy[:,1]*shape[1]+xy[:,0];order=np.argsort(flat);ordered=flat[order]
    for key,j in lookup.items():
        if key in ambiguous:continue
        x,y=key;loc=int(np.searchsorted(ordered,y*shape[1]+x))
        if loc==n or ordered[loc]!=y*shape[1]+x:continue
        i=order[loc];error=float(np.linalg.norm(p[i]+xy[i]-q[i]-reference['source_uv'][j]))
        census.append(dict(method=method,scene=frame.scene_id,log=log_id,frame=frame.frame_id,camera=cid,
            origin=int(origins[i]),current=q[i],source=p[i],moving_reference=bool(reference['moving'][j]),
            scorable=True,correct=bool(np.isfinite(error) and error<=2),reference_error_px=error,
            reuse_candidate_eligible=bool(eligible[i]),population='exact_annotated_LiDAR_reference_link_intersection_census',
            baseline_s=frame.cameras[cid].timestamp-link['source_timestamp']))
    return rows,denominators,census
