"""Identity-preserving task diagnostics. Never replaces official task metrics."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import write

# Declared before running followup experiments. The 3 m reporting gate uses
# the largest threshold in the repository's official OpenLane-V2 evaluator.
# Our symmetric continuous-line metric is a diagnostic, not its Frechet AP.
MAP_GATE_M = 3.0
DETECTION_GATE_M = 2.0  # nuScenes dist_th_tp, not an acceptance tolerance
ARC_SAMPLES = 101


def check_coordinates(native, candidate):
    if native != candidate or native != dict(frame='ego',units='m',dimensions=3):
        raise ValueError('Incompatible or unverified task coordinate contract')


def clean_line(points, valid_mask=None):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3:return None,'invalid_point_shape'
    if valid_mask is not None:
        mask=np.asarray(valid_mask)
        if mask.dtype!=bool or mask.shape!=(len(p),):return None,'invalid_point_mask'
        p=p[mask]
    if len(p)<2:return None,'insufficient_valid_points'
    if not np.isfinite(p).all():return None,'nonfinite_unmasked_points'
    keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>0]
    p=p[keep]
    if len(p)<2:return None,'zero_length_line'
    return p,None


def arc_samples(p,count=ARC_SAMPLES):
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    targets=np.linspace(0,length[-1],count)
    return np.stack([np.interp(targets,length,p[:,d]) for d in range(3)],axis=-1)


def point_segment_distances(points,line):
    start=line[:-1];delta=np.diff(line,axis=0)
    t=np.clip(((points[:,None]-start)*delta).sum(-1)/(delta*delta).sum(-1),0,1)
    return np.linalg.norm(points[:,None]-(start+t[...,None]*delta),axis=-1).min(-1)


def line_distances(a,b,mask_a=None,mask_b=None):
    a,ra=clean_line(a,mask_a);b,rb=clean_line(b,mask_b)
    if ra or rb:return dict(valid=False,reason=ra or rb)
    sa,sb=arc_samples(a),arc_samples(b)
    da,db=point_segment_distances(sa,b),point_segment_distances(sb,a)
    return dict(valid=True,forward_m=float(np.linalg.norm(sa-sb,axis=-1).mean()),
        reverse_m=float(np.linalg.norm(sa-sb[::-1],axis=-1).mean()),
        symmetric_m=float((da.mean()+db.mean())/2),
        directed_native_m=float(da.mean()),directed_candidate_m=float(db.mean()),
        symmetric_max_m=float(max(da.max(),db.max())))


def gated_assignment(cost,compatible,gate):
    """Maximum valid cardinality, then minimum cost; invalid pairs never consume matches."""
    cost=np.asarray(cost,dtype=float);compatible=np.asarray(compatible,dtype=bool)
    if cost.shape!=compatible.shape:raise ValueError('Cost/compatibility shapes differ')
    n,m=cost.shape
    if not n or not m:return [],list(range(n)),list(range(m))
    valid=compatible & np.isfinite(cost) & (cost<=gate) & (cost>=0)
    # One dummy per prediction on both sides. A lost valid match costs more
    # than every valid edge combined, ensuring maximum cardinality first.
    penalty=(n+m+1)*(gate+1)
    augmented=np.full((n+m,n+m),penalty*(n+m+1))
    augmented[:n,:m]=np.where(valid,cost,penalty*(n+m+1))
    augmented[np.arange(n),m+np.arange(n)]=penalty
    augmented[n+np.arange(m),np.arange(m)]=penalty
    augmented[n:,m:]=0
    rr,cc=linear_sum_assignment(augmented)
    pairs=[(int(i),int(j)) for i,j in zip(rr,cc) if i<n and j<m and valid[i,j]]
    return pairs,sorted(set(range(n))-{i for i,j in pairs}),sorted(set(range(m))-{j for i,j in pairs})


def line_summary(index,points,scores):
    from worldstream.m1.state_types import json_value
    p=np.asarray(points);confidence=np.asarray(scores)
    finite=np.isfinite(p).all(-1)
    return dict(index=int(index),class_id=int(confidence.argmax()) if confidence.ndim else 0,
        confidence=float(confidence.max()),class_confidences=confidence.tolist(),point_count=len(p),
        finite_points=int(finite.sum()),start=json_value(p[0]) if len(p) else None,end=json_value(p[-1]) if len(p) else None,
        coordinate_min=p[finite].min(0).tolist() if finite.any() else None,
        coordinate_max=p[finite].max(0).tolist() if finite.any() else None,points=json_value(p))


def diagnose_map(native,candidate,frame_id,gate=MAP_GATE_M):
    p,c=native;q,d=candidate
    p,q=[np.asarray(line) for line in p],[np.asarray(line) for line in q]
    c,d=np.asarray(c),np.asarray(d)
    if c.ndim==1:c=c[:,None]
    if d.ndim==1:d=d[:,None]
    ia=np.flatnonzero(c.max(-1)>=.3);ib=np.flatnonzero(d.max(-1)>=.3)
    legacy=[];old_cost=None
    shapes={line.shape for line in [*(p[i] for i in ia),*(q[j] for j in ib)]}
    legacy_reason='empty_prediction_set' if not len(ia) or not len(ib) else 'different_point_counts_or_shapes' if len(shapes)!=1 else 'nonfinite_points'
    shape_ok=len(shapes)==1 and len(next(iter(shapes)))==2 and next(iter(shapes))[0]>=2 and next(iter(shapes))[1]==3
    if len(ia) and len(ib) and shape_ok and all(np.isfinite(line).all() for line in [*(p[i] for i in ia),*(q[j] for j in ib)]):
        pp=np.stack([p[i] for i in ia]);qq=np.stack([q[j] for j in ib])
        forward=np.linalg.norm(pp[:,None]-qq[None],axis=-1).mean(-1)
        reverse=np.linalg.norm(pp[:,None]-qq[None,:,::-1],axis=-1).mean(-1)
        old_cost=np.minimum(forward,reverse)
        rr,cc=linear_sum_assignment(old_cost)
        legacy_pairs=list(zip(rr.tolist(),cc.tolist()));legacy_reason=None
    else:legacy_pairs=[]
    distances={};cost=np.full((len(ia),len(ib)),np.inf)
    classes=c[ia].argmax(-1)[:,None]==d[ib].argmax(-1)[None,:]
    for i in range(len(ia)):
        for j in range(len(ib)):
            value=line_distances(p[ia[i]],q[ib[j]]);distances[i,j]=value
            if value['valid']:cost[i,j]=value['symmetric_m']
    pairs,unmatched_a,unmatched_b=gated_assignment(cost,classes,gate)
    for i,j in legacy_pairs:
        value=distances[i,j]
        legacy.append(dict(frame_id=frame_id,native=line_summary(ia[i],p[ia[i]],c[ia[i]]),
            candidate=line_summary(ib[j],q[ib[j]],d[ib[j]]),legacy_forward_m=float(forward[i,j]),
            legacy_reverse_m=float(reverse[i,j]),legacy_cost_m=float(old_cost[i,j]),
            **value,diagnostic_valid=bool(classes[i,j] and value['valid'] and cost[i,j]<=gate),
            invalid_reason='class_incompatible' if not classes[i,j] else 'beyond_predeclared_geometry_gate' if cost[i,j]>gate else None,
            nearest_native_legacy_m=float(old_cost[i].min()),nearest_candidate_legacy_m=float(old_cost[:,j].min())))
    def unmatched(ids,side):
        result=[]
        for index in ids:
            compatible=classes[index] if side=='native' else classes[:,index]
            values=cost[index] if side=='native' else cost[:,index]
            reason=('no_class_compatible_candidate' if not compatible.any() else
                'no_finite_geometry' if not np.isfinite(values[compatible]).any() else
                'beyond_predeclared_geometry_gate' if np.min(values[compatible])>gate else 'one_to_one_competition')
            result.append(dict(index=int((ia if side=='native' else ib)[index]),reason=reason,
                nearest_compatible_distance_m=float(np.min(values[compatible])) if compatible.any() and np.isfinite(np.min(values[compatible])) else None))
        return result
    return dict(frame_id=frame_id,legacy=legacy,legacy_unavailable_reason=legacy_reason,
        native=[line_summary(i,p[i],c[i]) for i in ia],candidate=[line_summary(i,q[i],d[i]) for i in ib],
        symmetric_gated_matches=[dict(native_index=int(ia[i]),candidate_index=int(ib[j]),**distances[i,j]) for i,j in pairs],
        unmatched_native=unmatched(unmatched_a,'native'),unmatched_candidate=unmatched(unmatched_b,'candidate'))


def diagnose_detection(a,b,frame_id):
    def extract(result):
        result=result['pts_bbox'];ids=(result['scores_3d']>=.3).nonzero().flatten().numpy()
        return dict(ids=ids,center=result['boxes_3d'].gravity_center.numpy()[ids],
            labels=result['labels_3d'].numpy()[ids],scores=result['scores_3d'].numpy()[ids],
            boxes=result['boxes_3d'].tensor.numpy()[ids])
    a,b=extract(a),extract(b);cost=np.linalg.norm(a['center'][:,None]-b['center'][None,:],axis=-1)
    compatible=a['labels'][:,None]==b['labels'][None,:]
    legacy=[]
    for label in sorted(set(a['labels'].tolist())|set(b['labels'].tolist())):
        ia=np.flatnonzero(a['labels']==label);ib=np.flatnonzero(b['labels']==label)
        if not len(ia) or not len(ib):continue
        rr,cc=linear_sum_assignment(cost[np.ix_(ia,ib)])
        for i,j in zip(ia[rr],ib[cc]):
            legacy.append(dict(frame_id=frame_id,native_index=int(a['ids'][i]),candidate_index=int(b['ids'][j]),
                class_id=int(label),native_score=float(a['scores'][i]),candidate_score=float(b['scores'][j]),
                native_center=a['center'][i].tolist(),candidate_center=b['center'][j].tolist(),
                center_delta=(b['center'][j]-a['center'][i]).tolist(),distance_m=float(cost[i,j]),
                native_box=a['boxes'][i].tolist(),candidate_box=b['boxes'][j].tolist(),
                diagnostic_valid=bool(cost[i,j]<=DETECTION_GATE_M),
                invalid_reason=None if cost[i,j]<=DETECTION_GATE_M else 'beyond_predeclared_center_gate',
                native_nearest_m=float(cost[i,b['labels']==label].min()),candidate_nearest_m=float(cost[a['labels']==label,j].min())))
    pairs,ua,ub=gated_assignment(cost,compatible,DETECTION_GATE_M)
    return dict(frame_id=frame_id,legacy=legacy,gated_match_count=len(pairs),
        unmatched_native_indices=a['ids'][ua].tolist(),unmatched_candidate_indices=b['ids'][ub].tolist())


def trace_runtime(candidate_dir,summary):
    """Correlate archived state/selection with the actual displaced detection.

    Nearest historical geometric track is a candidate association, not GT identity.
    Global task heads prevent interpreting ROI overlap as a causal proof.
    """
    from worldstream.m1.omnidrive_adapter import quaternion_transform
    from worldstream.m1.geometry import transform_points,inverse_transform,project_points
    frame_id=summary['worst_detection']['frame_id'];candidate_dir=Path(candidate_dir)
    state_file=candidate_dir/'m1_state.jsonl'
    if not state_file.exists():return dict(status='UNAVAILABLE',reason='No archived M1 audit for this run')
    audits=[json.loads(s) for s in state_file.read_text().splitlines()]
    index=next(i for i,r in enumerate(audits) if r['precheck']['frame_id']==frame_id)
    current=audits[index]
    previous=audits[index-1] if index else None
    runtime=next(json.loads(s) for s in (candidate_dir/'per_frame_metrics.jsonl').read_text().splitlines() if json.loads(s)['frame_id']==frame_id)['m2']
    with open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb') as handle:infos=pickle.load(handle)['infos']
    info=next(r for r in infos if r['token']==frame_id)
    pose=quaternion_transform(info['ego2global_rotation'],info['ego2global_translation'])
    center=transform_points(np.asarray(summary['worst_detection']['native_center']),pose)
    anchors=(previous or {}).get('committed_state',{}).get('anchors',[])
    classes=json.loads((candidate_dir/'effective_config.json').read_text())['config']['class_names']
    label=classes[summary['worst_detection']['class_id']]
    candidates=[r for r in anchors if r['semantic_class']==label]
    candidates.sort(key=lambda r:np.linalg.norm(np.asarray(r['center_world'])-center))
    refresh=runtime['region_refresh'];cameras=runtime['cache_context']['camera_ids']
    def window_details(rt,cid,region):
        refresh=rt['region_refresh'];cameras=rt['cache_context']['camera_ids']
        footprints=rt['prefix_dependency_footprints'].get('0',rt['prefix_dependency_footprints'].get(0))
        result=[]
        for local,footprint in enumerate(footprints):
            x0,y0,x1,y1=footprint['input_region'];a,b,c,d=region
            if min(x1,c)<=max(x0,a) or min(y1,d)<=max(y0,b):continue
            flat=cameras.index(cid)*len(footprints)+local
            result.append(dict(window=flat,local_window=local,
                executed=flat in rt['stages'][0]['executed_window_ids'],
                mandatory=flat in refresh.get('mandatory_window_ids',[]),ordinary=flat in refresh.get('ordinary_window_ids',[]),
                age=refresh.get('ages',[None]*len(cameras)*len(footprints))[flat],
                drift=refresh.get('drift',[None]*len(cameras)*len(footprints))[flat],
                score=refresh.get('scores',[None]*len(cameras)*len(footprints))[flat]))
        return result
    linked=[]
    states=current['precheck']['computation_state']['region_states']
    for anchor in candidates[:3]:
        aid=anchor['anchor_id'];check=current['precheck']['diagnostics'].get('anchors',{}).get(aid,{})
        views=[]
        for cid,view in check.get('cameras',{}).items():
            region=view.get('predicted_region')
            if region is not None:
                views.append(dict(camera_id=cid,bbox_xyxy=region,geometry_valid=view['geometry_valid'],
                    observation_reason=view.get('reason'),visibility=view.get('evidence',{}).get('visibility_status'),
                    exported_region=any(r['camera_id']==cid and r['track_id']==aid for r in states),windows=window_details(runtime,cid,region)))
        linked.append(dict(track_id=aid,historical_world_center=anchor['center_world'],
            distance_to_native_prediction_m=float(np.linalg.norm(np.asarray(anchor['center_world'])-center)),
            association_qualification='nearest geometric hypothesis only',views=views))
    # Map and Detection extremes need not occur on the same frame. Never
    # project one frame's ego coordinates with another frame's pose/contract.
    map_frame=summary['worst_map']['frame_id']
    map_current=next(a for a in audits if a['precheck']['frame_id']==map_frame)
    map_info=next(r for r in infos if r['token']==map_frame)
    map_runtime=next(json.loads(s)['m2'] for s in (candidate_dir/'per_frame_metrics.jsonl').read_text().splitlines() if json.loads(s)['frame_id']==map_frame)
    map_pose=quaternion_transform(map_info['ego2global_rotation'],map_info['ego2global_translation'])
    metadata=map_current['precheck']['diagnostics']['execution_context']
    projected=[]
    line=np.asarray(summary['worst_map']['native']['points']);world_line=transform_points(line,map_pose)
    for camera_index,cid in enumerate(map_runtime['cache_context']['camera_ids']):
        cam=map_info['cams'][cid]
        c2w=quaternion_transform(cam['ego2global_rotation'],cam['ego2global_translation'])@quaternion_transform(cam['sensor2ego_rotation'],cam['sensor2ego_translation'])
        k=np.asarray(metadata['intrinsics'][camera_index])[:3,:3]
        uv,depth,valid=project_points(world_line,k,inverse_transform(c2w))
        height,width=metadata['shapes'][cid][:2]
        valid &= (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        if valid.any():
            points=uv[valid];region=[points[:,0].min(),points[:,1].min(),points[:,0].max()+1,points[:,1].max()+1]
            projected.append(dict(camera_id=cid,frame_id=map_frame,projected_bbox=region,windows=window_details(map_runtime,cid,region)))
    return dict(status='MEASURED',frame_id=frame_id,detector_native_world_center=center.tolist(),
        detector_class=label,nearest_historical_tracks=linked,map_frame_id=map_frame,map_native_extreme_projections=projected,
        region_state_count=len(states),event_regions=refresh.get('event_regions'),
        caveat='Archived hypotheses and sparse projected-line ROI overlap are correlation only. Current-frame forced-full counterfactual is evaluated separately.')


def diagnose_runs(native_dir,candidate_dir,output):
    native_dir,candidate_dir,output=Path(native_dir),Path(candidate_dir),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    tokens=json.loads((native_dir/'sequence.json').read_text())['tokens']
    if tokens!=json.loads((candidate_dir/'sequence.json').read_text())['tokens']:raise ValueError('Sample identity/order mismatch')
    runs=[]
    for directory in (native_dir,candidate_dir):
        with (directory/'predictions.pkl').open('rb') as handle:runs.append(pickle.load(handle))
    if any(len(run)!=len(tokens) for run in runs):raise ValueError('Incomplete prediction/token coverage')
    configurations=[json.loads((directory/'effective_config.json').read_text())['config'] for directory in (native_dir,candidate_dir)]
    if configurations[0]['model']['map_head']['pc_range']!=configurations[1]['model']['map_head']['pc_range']:
        raise ValueError('Map coordinate ranges differ')
    if configurations[0]['data']['test']['pipeline']!=configurations[1]['data']['test']['pipeline']:
        raise ValueError('Task preprocessing differs')
    check_coordinates(dict(frame='ego',units='m',dimensions=3),dict(frame='ego',units='m',dimensions=3))
    maps=[];detections=[]
    for token,a,b in zip(tokens,*runs):
        maps.append(diagnose_map(a['lane_results'][0],b['lane_results'][0],token))
        detections.append(diagnose_detection(a,b,token))
    all_map=[r for f in maps for r in f['legacy']];all_det=[r for f in detections for r in f['legacy']]
    worst=max(all_map,key=lambda r:r['legacy_cost_m']);det_worst=max(all_det,key=lambda r:r['distance_m'])
    position=tokens.index(worst['frame_id'])
    original_a,original_b=runs[0][position]['lane_results'][0],runs[1][position]['lane_results'][0]
    nearest={}
    for name,target,others,scores,index in (
        ('native_extreme',np.asarray(worst['native']['points']),original_b[0],original_b[1],worst['native']['index']),
        ('candidate_extreme',np.asarray(worst['candidate']['points']),original_a[0],original_a[1],worst['candidate']['index'])):
        distances=[line_distances(target,line) for line in others]
        valid=[(i,d['symmetric_m']) for i,d in enumerate(distances) if d['valid']]
        valid.sort(key=lambda item:(item[1],item[0]))
        nearest[name]=dict(note='All confidence levels for attribution only; original score filter is unchanged. Same query is not an identity certificate.',
            same_query_index=index,same_query_score=np.asarray(scores[index]).tolist() if index<len(scores) else None,
            same_query_geometry=distances[index] if index<len(distances) else None,
            nearest=[dict(index=i,score=np.asarray(scores[i]).tolist(),symmetric_m=d) for i,d in valid[:8]])
    new_dist=[r['symmetric_m'] for f in maps for r in f['symmetric_gated_matches']]
    summary=dict(map_gate_m=MAP_GATE_M,detection_gate_m=DETECTION_GATE_M,arc_samples=ARC_SAMPLES,
        gate_basis='OpenLane-V2 largest Frechet threshold (3m); diagnostic metric differs from official AP',
        legacy_map_mean_m=float(np.mean([r['legacy_cost_m'] for r in all_map])),
        legacy_map_max_m=worst['legacy_cost_m'],gated_symmetric_mean_m=float(np.mean(new_dist)) if new_dist else None,
        gated_map_matches=len(new_dist),map_unmatched_native=sum(len(r['unmatched_native']) for r in maps),
        map_unmatched_candidate=sum(len(r['unmatched_candidate']) for r in maps),
        detection_max_m=det_worst['distance_m'],worst_map=worst,worst_detection=det_worst,
        all_confidence_attribution=nearest,
        coordinate_evidence=dict(map_pc_range=configurations[0]['model']['map_head']['pc_range'],
            source='PETRHeadM._get_bboxes_single returns decoded ego coordinates directly; equal pc_range and preprocessing; no comparison-time transform',
            map_head_classes=configurations[0]['model']['map_head']['num_classes']),
        scope='Prediction diagnostics, not ground-truth quality; original legacy metric and all rejected pairs retained')
    write(output/'maps.json',maps);write(output/'detections.json',detections);write(output/'summary.json',summary)
    write(output/'state_window_trace.json',trace_runtime(candidate_dir,summary))
    if (candidate_dir/'per_frame_metrics.jsonl').exists():
        records=[json.loads(s) for s in (candidate_dir/'per_frame_metrics.jsonl').read_text().splitlines()]
        write(output/'anomaly_runtime.json',[r for r in records if r['frame_id'] in (worst['frame_id'],det_worst['frame_id'])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=next(r for r in maps if r['frame_id']==worst['frame_id'])
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for row in target['native']:
        p=np.asarray(row['points']);axes[0].plot(p[:,0],p[:,1],alpha=.45);axes[0].text(*p[0,:2],str(row['index']),fontsize=7)
    for row in target['candidate']:
        p=np.asarray(row['points']);axes[1].plot(p[:,0],p[:,1],alpha=.45);axes[1].text(*p[0,:2],str(row['index']),fontsize=7)
    for ax,title,row in zip(axes,('Native','WSGRR'),(worst['native'],worst['candidate'])):
        p=np.asarray(row['points']);ax.plot(p[:,0],p[:,1],color='red',linewidth=3,label=f"legacy pair index {row['index']}")
        ax.set(xlim=(-55,55),ylim=(-55,55),xlabel='ego x (m)',ylabel='ego y (m)',title=title);ax.set_aspect('equal');ax.grid();ax.legend()
    fig.suptitle(worst['frame_id']+' | old forced match: %.4f m'%worst['legacy_cost_m']);fig.tight_layout()
    fig.savefig(output/'map_extreme.svg');plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=diagnose_runs(args.native,args.candidate,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('worst_map','worst_detection')},indent=2))
