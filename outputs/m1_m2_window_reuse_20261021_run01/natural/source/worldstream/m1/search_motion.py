"""Unqualified, short-lived search hypotheses, isolated from world motion state.

Reads only prior actual commits. Tentative geometric IDs can propose searches;
they never update trusted velocity, certified identity history, or certificates.
"""
import numpy as np
from .geometry import state_geometry_version,project_points


def search_hypothesis(anchor,history,frame,config):
    d=dict(status='unknown',reason='short_actual_history',qualified=False,
           identity_claim=False,metric_geometry_claim=False)
    if anchor.anchor_type!='object' or anchor.semantic_class in {'barrier','traffic_cone'}:
        return dict(d,reason='static_or_nonobject')
    records=[]
    for record in history:
        if (record['scene_id']!=frame.scene_id or record['epoch']!=anchor.source_state_epoch
                or record['execution_signature']!=frame.execution_signature):continue
        a=record['anchors'].get(anchor.anchor_id)
        if a is None or a.source_frame_id!=record['frame_id'] or a.source_timestamp!=record['timestamp']:continue
        records.append((record,a))
    if len(records)<3:return d
    records=records[-3:];latest=records[-1][1]
    if latest.source_frame_id!=anchor.source_frame_id or state_geometry_version(latest)!=state_geometry_version(anchor):
        return dict(d,reason='source_version_mismatch')
    times=np.asarray([a.source_timestamp for _,a in records]);steps=np.diff(times);dt=frame.timestamp-times[-1]
    if (not np.isfinite(steps).all() or (steps<=0).any() or (steps>config.max_frame_gap_s).any()
            or not 0<dt<=min(config.max_state_age_s,.6)):
        return dict(d,reason='search_time_gap')
    # A tentative link needs an actual unambiguous geometric association record
    # for each step; reappearance after a missing frame has no velocity history.
    for index,(_,a) in enumerate(records[1:],1):
        if (a.association_history and a.association_history[-1].get('source_state_version')!=state_geometry_version(records[index-1][1])):
            return dict(d,reason='tentative_source_version_mismatch')
        if not a.association_history or a.association_history[-1]['status'] not in ('unqualified_geometric_track','qualified_identity'):
            return dict(d,reason='tentative_link_ambiguous_or_missing')
    centers=np.asarray([a.geometry_world.center for _,a in records]);velocities=np.diff(centers,axis=0)/steps[:,None]
    if not np.isfinite(velocities).all():return dict(d,reason='nonfinite_centers')
    velocity=velocities[-1];change=np.linalg.norm(velocities[-1]-velocities[-2])
    uncertainty=float(config.velocity_uncertainty_mps*dt+change*dt)
    if change>config.velocity_uncertainty_mps or uncertainty>=config.association_gate_m:
        return dict(d,reason='direction_or_acceleration_uncertain',uncertainty_m=uncertainty)
    return dict(d,status='search_only',reason='three_actual_tentative_centers',center_world=(centers[-1]+velocity*dt).tolist(),
        velocity_world=velocity.tolist(),uncertainty_m=uncertainty,source_timestamp=float(times[-1]),valid_until=float(times[-1]+.6),
        source_state_version=state_geometry_version(anchor),source_frame_id=anchor.source_frame_id,
        sources=[dict(frame_id=r['frame_id'],frame_digest=r['frame_digest'],state_version=state_geometry_version(a),
                      epoch=r['epoch'],execution_signature=r['execution_signature']) for r,a in records])


def search_edges(anchors,predictions,costs,frame,history,config):
    expanded=costs.copy();hypotheses={};counts=dict(attempts=0,generated=0,new_edges=0)
    for i,anchor in enumerate(anchors):
        h=search_hypothesis(anchor,history,frame,config);hypotheses[i]=h;counts['attempts']+=1
        if h['status']!='search_only':continue
        counts['generated']+=1
        for j,pred in enumerate(predictions):
            if (pred.anchor_type,pred.semantic_class)!=(anchor.anchor_type,anchor.semantic_class):continue
            a,b=anchor.geometry_world.size,pred.geometry_world.size
            if a is None or b is None or np.max(np.maximum(a/b,b/a))>config.box_size_ratio:continue
            distance=float(np.linalg.norm(pred.geometry_world.center-np.asarray(h['center_world'])))
            if distance+h['uncertainty_m']>config.association_gate_m:continue
            if not np.isfinite(expanded[i,j]):counts['new_edges']+=1
            expanded[i,j]=min(expanded[i,j],distance)
    return expanded,hypotheses,counts


def image_search_initial(anchor,old,new,points,hypothesis,config):
    """Box-centre optical depth is only a search hypothesis, not per-pixel truth."""
    if hypothesis is None or hypothesis.get('status')!='search_only':return None
    if (hypothesis['source_state_version']!=state_geometry_version(anchor)
            or hypothesis['source_frame_id']!=anchor.source_frame_id or old.timestamp>=new.timestamp):return None
    from .ego_motion import camera_pair_reason,project_source_depth
    if camera_pair_reason(old,new,config):return None
    center=anchor.geometry_world.center
    z=float((old.world_to_camera@np.r_[center,1])[2])
    if not .5<z<200:return None
    _,valid,world,_=project_source_depth(old,new,points,np.full(len(points),z))
    delta=np.asarray(hypothesis['velocity_world'])*(new.timestamp-old.timestamp)
    uv,target_z,front=project_points(world+delta,new.intrinsic,new.world_to_camera)
    uncertainty=np.abs(new.intrinsic[0,0])*hypothesis['uncertainty_m']/np.maximum(target_z,.5)
    uv[~valid|~front]=np.nan
    return uv,uncertainty
