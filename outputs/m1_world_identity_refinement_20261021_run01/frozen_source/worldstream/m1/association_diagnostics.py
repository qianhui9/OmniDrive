"""Read-only scalar gate diagnosis. No label or decision state is consumed."""
from collections import Counter
import json
import numpy as np
from .state_types import json_value
from .geometry import state_geometry_version


def edge_gates(old,new,frame,epoch,origin,config,prior=None,dormant=False):
    age=frame.timestamp-old.source_timestamp
    center,uncertainty=prior if prior is not None else (old.geometry_world.center,0.)
    residual=float(np.linalg.norm(new.geometry_world.center-center))
    size_ok=(old.geometry_world.size is not None and new.geometry_world.size is not None
             and np.max(np.maximum(old.geometry_world.size/new.geometry_world.size,
                                   new.geometry_world.size/old.geometry_world.size))<=config.box_size_ratio)
    gates=dict(category=(old.anchor_type,old.semantic_class)==(new.anchor_type,new.semantic_class),size=bool(size_ok),
        time=bool(np.isfinite(age) and 0<age<=(config.association_track_ttl_s if dormant else config.max_state_age_s)),
        scene=old.scene_id==frame.scene_id,epoch=old.source_state_epoch==epoch,
        signature=old.source_execution_signature==frame.execution_signature,source=old.evidence_origin==origin,
        geometry=bool(np.isfinite(residual) and residual+uncertainty<=config.association_gate_m),
        static_geometry=bool(np.linalg.norm(new.geometry_world.center-old.geometry_world.center)<=config.association_gate_m))
    return dict(age_s=age,static_center_distance_m=float(np.linalg.norm(new.geometry_world.center-old.geometry_world.center)),predicted_center=np.asarray(center).tolist(),prediction_residual_m=residual,
        uncertainty_budget_m=float(uncertainty),gates=gates)


def summarize(anchors,predictions,frame,store,origin,config,priors,dormant,graph,accepted,measurements,selected,byte_budget=None,session=None):
    """Diagnostic toggle controls recording only. Bounded in BOTH count and bytes."""
    byte_budget=config.association_diagnostic_bytes if byte_budget is None else byte_budget
    counts=Counter();classes={};samples=[];used=0;total=0;accepted=set(accepted)
    for i,old in enumerate(anchors):
        if old.anchor_type!='object':continue
        for j,new in enumerate(predictions):
            if new.anchor_type!='object':continue
            total+=1
            d=edge_gates(old,new,frame,store.generation,origin,config,priors.get(i),i in dormant)
            gates=d['gates'];value=measurements.get((i,j),dict(status='unknown',reason='not_measured',views=[]))
            supported=value['status']=='supported'
            legal=bool(np.isfinite(graph[i,j]));status='accepted' if (i,j) in accepted else 'unknown' if legal else 'rejected'
            reasons=[]
            if not gates['category']:reasons.append('class_mismatch')
            if not gates['size']:reasons.append('size_mismatch')
            if not gates['time']:reasons.append('time_mismatch')
            if not all(gates[k] for k in ('scene','epoch','signature','source')):reasons.append('source_mismatch')
            record=store.association_tracks.get(old.anchor_id)
            gates['state_version']=bool(record and record['samples'][-1]['version']==state_geometry_version(old))
            if record and not gates['state_version']:reasons.append('state_version_mismatch')
            if not record or len(record['samples'])<3:reasons.append('insufficient_history')
            if i not in priors:reasons.append('motion_prediction_unavailable')
            if not gates['geometry']:reasons.append('outside_geometry_gate')
            if value['status']=='contradicted':reasons.append('appearance_conflict')
            elif not supported:reasons.append('appearance_unknown')
            if any(v.get('occluded') for v in value.get('views',[])):reasons.append('occluded')
            if any(v.get('projection_valid') is False for v in value.get('views',[])):reasons.append('roi_projection_invalid')
            if legal and (np.isfinite(graph[i]).sum()>1 or np.isfinite(graph[:,j]).sum()>1):reasons.append('ambiguous_competitor')
            cls=classes.setdefault(old.semantic_class,Counter());cls[status]+=1;counts[status]+=1
            for reason in reasons:counts[reason]+=1;cls[reason]+=1
            # Store useful same-class relations first, not a quadratic log of
            # trivial class mismatches. All independent gates are still counted.
            if not config.association_diagnostics or not gates['category']:continue
            if len(samples)>=config.association_diagnostic_limit or used+1000>byte_budget:continue
            if session is not None and not value.get('views'):
                views=[]
                for cid in frame.cameras:
                    p=session.projection(j,cid)
                    views.append(dict(camera=cid,projection_valid=p.valid,region=list(p.region) if p.region is not None else None,
                        roi_size=[p.region[2]-p.region[0],p.region[3]-p.region[1]] if p.region is not None else None,
                        coverage=float(p.coverage),occluded=session.occluded(j,cid),visibility='unknown',
                        descriptor_available=cid in old.observation_support,aligned=False,tracks=0,
                        source_quadrants=0,current_quadrants=0,photometric_residual=None,descriptor_residual=None,
                        reason=value['reason'],status='unknown'))
                value=dict(value,views=views)
                if any(v.get('occluded') for v in views) and 'occluded' not in reasons:
                    reasons.append('occluded');counts['occluded']+=1;cls['occluded']+=1
            row=dict(anchor_id=old.anchor_id,prediction_index=selected[j][0],source_frame=old.source_frame_id,
                current_frame=frame.frame_id,semantic_class=old.semantic_class,source_state_version=state_geometry_version(old),
                history_state_version=record['samples'][-1]['version'] if record else None,**d,
                appearance=value,status=status,reasons=reasons,dormant=i in dormant)
            size=len(json.dumps(json_value(row),allow_nan=False).encode())+2
            if len(samples)<config.association_diagnostic_limit and used+size<=byte_budget:
                samples.append(row);used+=size
    return dict(edge_count=total,counts=dict(counts),by_class={k:dict(v) for k,v in classes.items()},
        samples=samples,sample_bytes=used,samples_truncated=total-len(samples))
