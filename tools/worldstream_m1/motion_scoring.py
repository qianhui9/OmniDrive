"""Offline sparse LiDAR/annotation correspondence scoring, after online commits.

Sparse surface samples and annotated rigid-box motion are an imperfect reference,
not dense instance truth or producer qualification. No label enters online M1.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .dataset import load_oracle,frame_from_info
from .coverage_oracle import world_lidar,prepare_pair
from .scene_protocol import ratio,distribution
from .effective_acceptance import sha
from worldstream.m1 import M1Config


def score_points(reference, measurements):
    p=np.asarray(measurements['source_points'],float);q=np.asarray(measurements['current_points'],float)
    valid=np.asarray(measurements['correspondence_valid'],bool)&np.asarray(measurements['geometry_valid'],bool)&np.asarray(measurements['patch_supported'],bool)
    accepted=np.asarray(measurements.get('observation_supported_after_exclusion',valid),bool)
    ids=np.flatnonzero(reference['scorable'])
    rows=[]
    if len(ids):
        tree=cKDTree(reference['target_uv'][ids]);distance,index=tree.query(q)
    else:distance=np.full(len(q),np.inf);index=np.zeros(len(q),int)
    for j in range(len(q)):
        status='unscorable';depth=None;semantic='unknown';motion='unknown';visibility='sparse_reference_unavailable';error=None
        if len(ids) and distance[j]<=1.:
            i=ids[index[j]];delta=reference['target_uv'][i]-q[j]
            # Local displacement approximation within one pixel, with the same
            # existing 2px offline correspondence scoring threshold.
            error=float(np.linalg.norm(p[j]+delta-reference['source_uv'][i]))
            status='correct' if error<=2. else 'wrong'
            depth=float(reference['depth_m'][i]);semantic=str(reference['semantic_class'][i])
            motion='object_moving' if reference['moving'][i] else 'conditional_static'
            visibility='both_sparse_depth_supported'
            # Multiple LiDAR surfaces under the same pixel must not manufacture
            # a uniquely scorable match from a convenient nearest neighbour.
            nearby=ids[tree.query_ball_point(q[j],1.)]
            alternatives=reference['source_uv'][nearby]
            if len(alternatives)>1 and np.linalg.norm(alternatives-reference['source_uv'][i],axis=1).max()>2.:
                status='unscorable';visibility='sparse_reference_ambiguous'
        rows.append(dict(index=j,measured=bool(valid[j]),observation_supported=bool(accepted[j]),score=status,
            correspondence_error_px=error,depth_m=depth,depth_group='unknown' if depth is None else 'near_lt10m' if depth<10 else 'middle_10to30m' if depth<30 else 'far_ge30m',
            semantic_class=semantic,object_motion=motion,visibility=visibility))
    return rows


def counts(rows):
    c=Counter(total=len(rows),correct=0,wrong=0,unscorable=0)
    for r in rows:c[r['score']]+=1
    return dict(counts=dict(c),scorable_fraction=ratio(c['correct']+c['wrong'],c['total']),
        correspondence_accuracy=ratio(c['correct'],c['correct']+c['wrong']))


def summarize(rows):
    selected=[r for r in rows if r['measured']];observed=[r for r in rows if r['observation_supported']]
    return dict(all_tracks=counts(rows),measured_same_RGB=counts(selected),observation_supported=counts(observed),
        strata={key:{v:counts([r for r in selected if r[key]==v]) for v in sorted({r[key] for r in selected})}
                for key in ('depth_group','semantic_class','object_motion','visibility','ego_motion_group')})


def panel(previous,current,cid,m,scored,path,rejected=False):
    left=previous.cameras[cid].image.copy();right=current.cameras[cid].image.copy()
    h,w=left.shape[:2];image=np.concatenate([left,right],axis=1)
    p=np.asarray(m['source_points']);q=np.asarray(m['current_points']);pred=np.asarray(m.get('predicted_points',[]),float)
    raw=np.asarray(m.get('raw_points',m['current_points']),float)
    drawable=[x for x in scored if bool(m['correspondence_valid'][x['index']]) and (not x['measured'] if rejected else x['measured'])]
    for r in drawable[:32]:
        j=r['index'];color={'correct':(0,230,0),'wrong':(255,0,0),'unscorable':(255,180,0)}[r['score']]
        a=tuple(np.rint(p[j]).astype(int));b=tuple(np.rint(q[j]+[w,0]).astype(int))
        cv2.line(image,a,b,color,1);cv2.circle(image,b,3,color,1)
        if bool(m.get('raw_valid',m['correspondence_valid'])[j]) and np.isfinite(raw[j]).all():cv2.drawMarker(image,tuple(np.rint(raw[j]+[w,0]).astype(int)),(0,255,255),cv2.MARKER_CROSS,5,1)
        if len(pred) and bool(m.get('prediction_valid',[False]*len(p))[j]) and np.isfinite(pred[j]).all():cv2.drawMarker(image,tuple(np.rint(pred[j]+[w,0]).astype(int)),(255,0,255),cv2.MARKER_DIAMOND,6,1)
    cv2.putText(image,('quality-rejected FB-valid tracks' if rejected else 'measured tracks')+': green correct/red wrong/orange unscorable; cyan raw; magenta prediction',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))


def score_run(root,output,runtime_config=None,m1_config=None):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    raw=(root/'protocol.json').exists()
    protocol=json.loads((root/('protocol.json' if raw else 'provenance.json')).read_text())
    config=M1Config.load(protocol['config'] if raw else m1_config)
    if not raw:
        if runtime_config is None or m1_config is None:raise ValueError('Native image reconstruction needs explicit frozen runtime and M1 configurations')
        from mmcv import Config
        from .state_interface_replay import preprocessed_frame
        cfg=Config.fromfile(runtime_config)
    def get_frame(info):
        return frame_from_info(info,'data/nuscenes') if raw else preprocessed_frame(info,cfg,config)[0]
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) if raw else sorted((root/'scenes').glob('*/m1_state.jsonl'))
    rows_by_scene={p.parent.name:[json.loads(x) for x in p.read_text().splitlines()] for p in traces}
    with open(protocol.get('infos_path',protocol.get('dataset')),'rb') as f:infos=pickle.load(f)['infos']
    tokens={r['precheck']['frame_id'] for rows in rows_by_scene.values() for r in rows}
    selected={i['token']:i for i in infos if i['token'] in tokens}
    (out/'protocol.json').write_text(json.dumps(dict(online_GT=False,labels_read_after_commits=True,
        coordinates='resized_raw_image' if raw else 'native preprocessed_image reconstructed with existing test pipeline',
        runtime_config=runtime_config,m1_config=m1_config,
        target_match_radius_px=1.,correspondence_tolerance_px=2.,qualification_population=False,
        source_traces={str(p):sha(p) for p in traces},scorer_sha256=sha(__file__),
        limitations=['Unannotated LiDAR points assumed static','Rigid annotated box motion approximation',
                    'Sparse depth only; camera/LiDAR timing and occlusions can be unscorable',
                    'No object ownership or formal identity accuracy follows from image point counts']),indent=2)+'\n')
    oracle=load_oracle(list(selected.values()),'data/nuscenes')
    all_scores=[];summaries=[];residual_pool={}
    for scene,rows in rows_by_scene.items():
        scores=[];previous=None;previous_points=None;previous_info=None;timing={};funnel=Counter();drawn=False
        folder=out/scene;folder.mkdir()
        raw_tracks=folder/'scored_tracks.jsonl'
        with raw_tracks.open('w') as f:
            for row in rows:
                token=row['precheck']['frame_id'];info=selected[token]
                current=get_frame(info);points=world_lidar(info)
                if previous is not None:
                    ref=prepare_pair(previous,current,previous_points,points,oracle[previous.frame_id],oracle[token],config)
                    for cid,e in row['precheck']['diagnostics'].get('temporal_evidence',{}).items():
                        m=e.get('point_measurements')
                        if m is None:continue
                        scored=score_points(ref[cid],m)
                        ego=e.get('ego_motion',{})
                        angle=ego.get('rotation_deg');translation=ego.get('camera_translation_m')
                        # Same thresholds for all variants, based on actual camera poses.
                        relative=current.cameras[cid].world_to_camera@np.linalg.inv(previous.cameras[cid].world_to_camera)
                        angle=float(np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1))))
                        translation=float(np.linalg.norm(relative[:3,3]))
                        group='turn_ge3deg' if angle>=3 else 'translation_ge1m' if translation>=1 else 'small_motion'
                        for r in scored:
                            r.update(scene_id=scene,source_frame=previous.frame_id,current_frame=token,camera=cid,ego_rotation_deg=angle,
                                ego_translation_m=translation,ego_motion_group=group)
                            f.write(json.dumps(r)+'\n')
                        scores+=scored
                        for name in ('raw_valid','recovered','conflicts','depth_points','predicted_points'):
                            fallback=e.get('bidirectional_points',0) if name=='raw_valid' else 0
                            funnel[name]+=int(ego.get(name,fallback))
                        derived=dict(m)
                        pp=np.asarray(m['source_points'],float);qq=np.asarray(m['current_points'],float)
                        derived['measured_displacement_px']=np.linalg.norm(qq-pp,axis=1)
                        if 'predicted_points' in m:
                            predicted=np.asarray(m['predicted_points'],float)
                            derived['raw_prediction_residual_px']=np.linalg.norm(np.asarray(m['raw_points'],float)-predicted,axis=1)
                            derived['prediction_uncertainty_px']=np.asarray(m['prediction_spread_px'],float)
                        for name in ('fb_error_px','raw_fb_error_px','prediction_residual_px','patch_residual',
                                     'measured_displacement_px','raw_prediction_residual_px','prediction_uncertainty_px'):

                            vals=np.asarray(derived.get(name,[]),float)
                            if len(vals):
                                mask=np.asarray(m['correspondence_valid'],bool)
                                kept=vals[mask&np.isfinite(vals)].tolist()
                                timing.setdefault(name,[]).extend(kept)
                                residual_pool.setdefault(name,[]).extend(kept)
                        if not drawn and any(r['measured'] for r in scored):
                            panel(previous,current,cid,m,scored,folder/'actual_correspondence_case.png')
                            panel(previous,current,cid,m,scored,folder/'quality_rejection_case.png',rejected=True);drawn=True
                previous=current;previous_points=points;previous_info=info
        prediction_counts=dict(funnel)
        if config.ego_motion_mode=='off':
            prediction_counts.update({k:None for k in ('recovered','conflicts','depth_points','predicted_points')})
        elif config.ego_motion_mode=='rotation':prediction_counts['depth_points']=None
        summary=dict(scene_id=scene,**summarize(scores),prediction_funnel=prediction_counts,
            residuals={k:distribution(v) for k,v in timing.items()})
        (folder/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary);all_scores+=scores
        print(scene,'offline correspondence scoring',summary['observation_supported'],flush=True)
    result=dict(scenes=summaries,micro=summarize(all_scores),
        residual_micro={k:distribution(v) for k,v in residual_pool.items()},
        residual_macro_scene_median={k:distribution([s['residuals'].get(k,{}).get('median') for s in summaries]) for k in residual_pool},macro={pop:{metric:distribution([s[pop][metric] for s in summaries])
        for metric in ('scorable_fraction','correspondence_accuracy')} for pop in ('measured_same_RGB','observation_supported')},
        label_table_sha256={str(p):sha(p) for p in [Path('data/nuscenes/v1.0-trainval')/n for n in ('sample_annotation.json','instance.json','category.json')]},
        qualification_generated=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
