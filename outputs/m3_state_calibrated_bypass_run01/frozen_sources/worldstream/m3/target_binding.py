"""Small CPU target-binding risk over actual committed M1 observations.

Historical objects can veto an unsafe nearest choice; never supply a current
answer. Risk scores and motion envelopes are not calibrated probabilities.
"""
import math
from time import perf_counter
import numpy as np


def target_binding_risk(profile,witness,witnesses,current,previous,temporal=False):
    from .answer_validity import fact_directions,fact_region
    tick=perf_counter()
    distance=lambda a:math.hypot(*a.center_ego[:2])
    nearest=profile.selector=='nearest'
    competitors=[]
    for a in witnesses:
        if a.anchor_id==witness.anchor_id:continue
        margin=distance(a)-distance(witness)
        changes_answer=(a.semantic_class!=witness.semantic_class if profile.intent=='category' else
                        fact_directions(a.center_ego)!=fact_directions(witness.center_ego))
        if nearest and margin<max(2.,.15*distance(witness)) and changes_answer:
            competitors.append(dict(anchor_id=a.anchor_id,distance_gap_m=margin,
                angular_gap_rad=abs(math.atan2(a.center_ego[1],a.center_ego[0])-
                                    math.atan2(witness.center_ego[1],witness.center_ego[0]))))
    boundary=nearest and ((profile.direct and abs(abs(witness.center_ego[1])-4.)<.75)
        or (profile.intent=='position' and any(abs(abs(v)-1.)<.75 for v in witness.center_ego[:2])))
    current_risk=min(1.,.45*bool(competitors)+.35*bool(boundary)+.2*(1.-witness.score)
                     +.1*(len(witness.cameras)==1))
    out=dict(current_risk=current_risk,temporal_risk=0.,current_competitors=competitors,
        boundary=bool(boundary),temporal_enabled=temporal,temporal_competitors=[],
        current_binding_wall_ms=(perf_counter()-tick)*1000,temporal_risk_wall_ms=0.,
        temporal_status='disabled',score_kind='UNCALIBRATED_BINDING_RISK',
        continuity=dict(current_observation_required=True,matched=False,identity_qualified=False,
                        admission_credit=0.,reason='No calibrated positive-evidence admission credit'))
    tick=perf_counter()
    if temporal and nearest:
        out['temporal_status']='no_legal_history'
        if (previous is not None and previous.scene_id==current.scene_id and previous.epoch==current.epoch
                and previous.execution_signature==current.execution_signature
                and 0<current.timestamp-previous.timestamp<=1.5):
            dt=current.timestamp-previous.timestamp
            from worldstream.m1.geometry import valid_transform
            old_pose=np.asarray(previous.ego_to_world)
            pose=np.asarray(current.ego_to_world)
            old_pose=old_pose.reshape(4,4) if old_pose.size==16 else np.zeros((4,4))
            pose=pose.reshape(4,4) if pose.size==16 else np.zeros((4,4))
            if valid_transform(old_pose) and valid_transform(pose):
                transform=np.linalg.inv(pose)@old_pose
                out['temporal_status']='legal_pose_age_motion_envelope'
                out['dt_s']=dt
                prior=next((a for a in previous.anchors if a.anchor_id==witness.anchor_id and a.observed
                    and a.semantic_class==witness.semantic_class),None)
                if prior is not None and witness.observed:
                    pred=transform[:3,:3]@np.asarray(prior.center_ego)+transform[:3,3]
                    if prior.velocity_ego:pred+=transform[:3,:3]@np.asarray(prior.velocity_ego)*dt
                    out['continuity'].update(matched=True,previous_source_frame=prior.source_frame,
                        residual_m=math.dist(pred[:2],witness.center_ego[:2]),
                        association_unambiguous=witness.association_unambiguous,
                        identity_qualified=bool(prior.identity_qualified and witness.identity_qualified))
                for a in previous.anchors:
                    if (a.anchor_type!='object' or not a.observed or a.evidence_origin!='actual_perception'
                            or a.score is None or a.score<.35):continue
                    # Reuse the question taxonomy; no matching by annotation ID.
                    from .world_delta import VEHICLES
                    classes=set(profile.classes)
                    if 'vehicle' in classes:classes=(classes-{'vehicle'})|VEHICLES
                    if classes and 'object' not in classes and a.semantic_class not in classes:continue
                    center=transform[:3,:3]@np.asarray(a.center_ego)+transform[:3,3]
                    reliable=bool(a.velocity_ego)
                    if reliable:center+=transform[:3,:3]@np.asarray(a.velocity_ego)*dt
                    uncertainty=.75+dt*(1. if reliable else 3. if a.semantic_class=='pedestrian' else 12.)
                    observed=[b for b in witnesses if b.semantic_class==a.semantic_class
                        and (b.anchor_id==a.anchor_id and b.association_unambiguous
                             or math.dist(b.center_ego[:2],center[:2])<1.5)]
                    if len(observed)==1:continue
                    # Test the envelope against question ROI, never call its center observed.
                    x,y=center[:2]
                    intersects=math.hypot(x,y)-uncertainty<=profile.near_m
                    if profile.direct:intersects &= abs(y)-uncertainty<=4.
                    for d in profile.directions:
                        intersects &= {'front':x+uncertainty>0,'rear':x-uncertainty<0,
                                       'left':y+uncertainty>0,'right':y-uncertainty<0}[d]
                    if not intersects or math.hypot(x,y)-uncertainty>distance(witness)+2.:continue
                    # Same-class category competition cannot change the required fact.
                    if profile.intent=='category' and a.semantic_class==witness.semantic_class:continue
                    if profile.intent=='position' and fact_directions(center)==fact_directions(witness.center_ego):continue
                    out['temporal_competitors'].append(dict(anchor_id=a.anchor_id,source_frame=a.source_frame,
                        source_timestamp=a.source_timestamp,semantic_class=a.semantic_class,
                        predicted_center_current_ego=center.tolist(),uncertainty_radius_m=uncertainty,
                        velocity_reliable=reliable,association_matches=len(observed),
                        reason='unobserved_competing_target_risk',current_fact=False))
                out['temporal_risk']=.7*bool(out['temporal_competitors'])
            else:
                out['temporal_status']='invalid_pose_uncertainty'
                out['temporal_risk']=.7
    out['temporal_risk_wall_ms']=(perf_counter()-tick)*1000 if temporal and nearest else 0.
    out['total_risk']=min(1.,out['current_risk']+out['temporal_risk'])
    return out
