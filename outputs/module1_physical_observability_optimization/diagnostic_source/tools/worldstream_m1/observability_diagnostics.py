"""Pre-method identifiability replay; reference truth is strictly offline."""
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.surface_observability import observe_points
from worldstream.m1.identity_temporal import _lk
from worldstream.m1.temporal_visibility import gray_image
from worldstream.m1.state_types import json_value
from .dataset import frame_from_info
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .evidence_experiments import load_inputs,append_json
from .io_utils import write_json

ROOT=Path('outputs/module1_physical_observability_optimization')
PRIOR=Path('outputs/module1_quality_coverage_refinement')


def point_record(m,i):
    view=[]
    for v in m['extra_views']:
        view.append(dict(kind=v['kind'],camera=v['camera_id'],timestamp=v['timestamp'],
            observed=bool(v['observed'][i]),usable=bool(v['usable'][i]),supported=bool(v['supported'][i]),
            predicted_uv=v['predicted_uv'][i].tolist(),angle_deg=float(v['angle'][i]),
            photo=float(v['photo'][i]),gradient=float(v['gradient'][i]),texture=float(v['texture'][i]),
            compatible_alternative_count=int(v['alternative_supported'][i].sum()),
            incompatible_alternative_count=int((v['alternative_supported'][i]&m['alternative_incompatible'][i]).sum())))
    hypotheses=[]
    for j in np.flatnonzero(m['alternative_supported'][i]):
        hypotheses.append(dict(source_uv=m['alternative_source_uv'][i,j].tolist(),world=m['alternative_world'][i,j].tolist(),
            incompatible=bool(m['alternative_incompatible'][i,j]),photo=float(m['alternative_photo'][i,j]),
            supporting_extra_views=[v['camera_id'] for v in m['extra_views'] if v['alternative_supported'][i,j]]))
    return dict(world=m['world'][i],parallax_deg=float(m['angle'][i]),triangulation_error_px=float(m['error'][i]),
        depths=m['depths'][i],triangulation_valid=bool(m['valid'][i]),
        projective_two_view_patch={k:(v[i].tolist() if hasattr(v[i],'tolist') else v[i]) for k,v in m['two_view_patch'].items()},
        alternatives=hypotheses,extra_views=view,unavailable_views=m['unavailable_views'])


def panel(previous,current,row,path):
    images=[]
    for c,center in ((previous,row['measured_source_uv']),(current,row['pixel'])):
        im=c.image.copy()
        if im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        for hyp in row['observability']['alternatives']:
            if c is previous:
                cv2.circle(im,tuple(np.rint(hyp['source_uv']).astype(int)),2,(220,150,20) if hyp['incompatible'] else (20,220,60),1)
        xy=np.rint(center).astype(int);cv2.circle(im,tuple(xy),3,(240,30,30),1)
        rad=24;h,w=im.shape[:2];x,y=xy
        crop=im[max(0,y-rad):min(h,y+rad),max(0,x-rad):min(w,x+rad)]
        images.append(cv2.resize(crop,(288,288),interpolation=cv2.INTER_NEAREST))
    canvas=np.zeros((340,576,3),np.uint8);canvas[:288]=np.concatenate(images,axis=1)
    cv2.putText(canvas,f"Reference false={row['false']}  incompatible observed alternatives={sum(h['incompatible'] for h in row['observability']['alternatives'])}",(4,308),0,.38,(255,255,255),1)
    cv2.putText(canvas,'Red: measured; amber: alternative. Supported patches do not establish truth.',(4,328),0,.36,(255,255,255),1)
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))


def scene(spec):
    infos,outputs,oracle=_INPUTS;folder=ROOT/'diagnostics'/spec['scene_id'];folder.mkdir(parents=True,exist_ok=False)
    cfg=replace(M1Config.load('configs/worldstream_m1_independent.yaml'),dense_region_policy='surface_consensus',
        dense_guard_radius_px=2,surface_boundary_guard=False,surface_diagnostics=True)
    m=WorldStateManager(cfg);previous=None;oldpoints=None;parity=Counter()
    archived={}
    for row in map(json.loads,(PRIOR/'traces/development'/spec['scene_id']/'pixels.jsonl').read_text().splitlines()):
        if row['method']=='surface_consensus':archived[row['frame'],row['camera']]=row
    oldrecords=[r for r in _OLD_POINTS if r['scene']==spec['scene_id']]
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token])
        refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        result=m.precheck(frame)
        if refs:
            for cid,cam in frame.cameras.items():
                e=m.last_evidence.get(cid);ref=dict(refs[cid]);ref['scorable']=ref['scorable']&ref['annotated'];invalid=m.last_masks[cid]
                score=score_camera(ref,invalid,e,cfg);expected=archived[token,cid]
                parity['camera_rows']+=1
                parity['differences']+=any(json_value(score[k])!=expected[k] for k in ('accepted_points','false_points','critical_points','missed_critical_points','events'))
                # Diagnose every ROI failure, including tiny events. No GT ROI
                # enters the manager or any online observation function.
                for event in score['events']:
                    if event['invalidation_coverage']<.9:
                        append_json(folder/'roi_failures.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,event=event,
                            observation_stable_pixels=int(e.stable_mask.sum()),forced=result.full_refresh_hint,
                            mask_tile_contract='Exact refresh mask is scored. Regions is a conservative tile envelope, not the exact refresh mask.'))
                idx=np.flatnonzero(ref['scorable']);xy=ref['xy'];accepted=idx[~invalid[xy[idx,1],xy[idx,0]]]
                tags={int(i):['surface_consensus'] for i in accepted}
                oldlookup={}
                for old in oldrecords:
                    if old['frame']!=token or old['camera']!=cid:continue
                    match=idx[(xy[idx]==old['pixel']).all(1)]
                    if not len(match):raise RuntimeError('Frozen old point lost independent reference')
                    chosen=int(match[np.argmin(np.linalg.norm(ref['source_uv'][match]-old['reference_source_uv'],axis=1))])
                    tags.setdefault(chosen,[]).append('surface_local_plane');oldlookup[chosen]=old['case_id']
                selected=np.asarray(sorted(tags),int)
                if not len(selected) or e.dense_source_map is None:continue
                q=xy[selected].astype(float);p=e.dense_source_map[xy[selected,1],xy[selected,0]]
                observations=observe_points(previous.cameras[cid],cam,p,q,cfg,frame.cameras,m._surface_history.get(cid))
                tracked,fb,valid=_lk(gray_image(cam),gray_image(previous.cameras[cid]),q.astype(np.float32))
                distance=cv2.distanceTransform(e.stable_mask.astype(np.uint8),cv2.DIST_L2,3)
                for i,index in enumerate(selected):
                    x,y=q[i].astype(int);error=float(np.linalg.norm(p[i]-ref['source_uv'][index]));false=not np.isfinite(error) or error>2 or bool(ref['moving'][index])
                    units=[u for u in e.surface_units or [] if u['bounds'][0]<=x<u['bounds'][2] and u['bounds'][1]<=y<u['bounds'][3] and u['models']]
                    fields=e.dense_measurements or {}
                    row=dict(scene=frame.scene_id,log_token=spec.get('log_token'),frame=token,source_frame=previous.frame_id,camera=cid,
                        reference_index=int(index),pixel=[int(x),int(y)],populations=tags[int(index)],old_case_id=oldlookup.get(int(index)),
                        measured_source_uv=p[i],reference_source_uv=ref['source_uv'][index],reference_error_px=error,false=false,
                        root_cause='Unknown_Root_Cause',reference_motion=bool(ref['moving'][index]),
                        online={k:float(v[y,x]) for k,v in fields.items() if k in ('fb','photo','gradient','texture','jacobian')},
                        independent_retracking=dict(valid=bool(valid[i]),fb=float(fb[i]),source_uv=tracked[i],
                            DIS_difference_px=float(np.linalg.norm(tracked[i]-p[i])),reference_error_px=float(np.linalg.norm(tracked[i]-ref['source_uv'][index]))),
                        consensus_units=units,observability=point_record(observations,i),
                        source_pose=previous.cameras[cid].world_to_camera,current_pose=cam.world_to_camera,
                        source_intrinsic=previous.cameras[cid].intrinsic,current_intrinsic=cam.intrinsic,
                        evidence_interpretation='Extra RGB acquisition is a different geometric measurement, not statistically independent ownership truth')
                    if false and 'surface_consensus' in row['populations']:
                        path=ROOT/'visualizations/correspondence_cases'/f'{token}_{cid}_{int(index)}.png'
                        panel(previous.cameras[cid],cam,row,path);row['visualization']=str(path)
                    append_json(folder/'points.jsonl',row)
        m.commit(frame,outputs[token],result);previous=frame;oldpoints=points
    assert not parity['differences'],parity
    write_json(folder/'complete.json',dict(scene=spec['scene_id'],parity=dict(parity)))
    print('observability',spec['scene_id'],dict(parity),flush=True)


def summarize():
    records=[];rois=[]
    for p in (ROOT/'diagnostics').glob('*/complete.json'):
        records.extend(map(json.loads,(p.parent/'points.jsonl').read_text().splitlines()) if (p.parent/'points.jsonl').exists() else [])
        rois.extend(map(json.loads,(p.parent/'roi_failures.jsonl').read_text().splitlines()) if (p.parent/'roi_failures.jsonl').exists() else [])
    main=[r for r in records if 'surface_consensus' in r['populations']]
    assert len(main)==690 and sum(r['false'] for r in main)==196,(len(main),sum(r['false'] for r in main))
    counts={}
    for label,rows in [('correct',[r for r in main if not r['false']]),('false',[r for r in main if r['false']])]:
        counts[label]=dict(points=len(rows),two_view_ambiguous=sum(any(h['incompatible'] for h in r['observability']['alternatives']) for r in rows),
            extra_observation_supported=sum(any(v['supported'] for v in r['observability']['extra_views']) for r in rows),
            temporal_supported=sum(any(v['kind']=='additional_time' and v['supported'] for v in r['observability']['extra_views']) for r in rows),
            cross_camera_supported=sum(any(v['kind']=='overlap_camera' and v['supported'] for v in r['observability']['extra_views']) for r in rows),
            same_RGB_LK_agreement_05=sum(r['independent_retracking']['valid'] and r['independent_retracking']['DIS_difference_px']<=.5 for r in rows))
    write_json(ROOT/'surface_ambiguity_analysis.json',dict(population='Frozen 690 current and 194 historical sparse accepts, merged by actual reference index',
        measured_comparison=counts,records=records,physical_root_causes_confirmed=0,
        limits='Finite epipolar patch alternatives; no global uniqueness proof. Photometric failure alone is not an occlusion diagnosis.'))
    write_json(ROOT/'surface_wrong_correspondence_cases.json',dict(false_count=196,correct_count=494,
        records=[r for r in main if r['false']],true_controls=[r for r in main if not r['false']],physical_cause='Unknown_Root_Cause'))
    write_json(ROOT/'critical_roi_diagnosis.json',dict(events=rois,minimum=min(r['event']['invalidation_coverage'] for r in rois),
        note='Measured Stable overlaps annotated event ROI. Root physical cause not inferred from event labels; exact mask and conservative tile envelope are separate existing outputs.'))
    paths=[ROOT/n for n in ('surface_ambiguity_analysis.json','surface_wrong_correspondence_cases.json','critical_roi_diagnosis.json')]
    write_json(ROOT/'observability_diagnostic_freeze.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print('measured identifiability',counts,'ROI violations',len(rois),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    if a.summarize_only:summarize();return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    logs={s['token']:s['log_token'] for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    specs=[dict(s,log_token=logs[s['scene_id']]) for s in specs]
    global _INPUTS,_OLD_POINTS
    _INPUTS=(infos,outputs,oracle);_OLD_POINTS=json.loads((PRIOR/'surface_true_false_comparison.json').read_text())['records']
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,specs))
    summarize()


if __name__=='__main__':main()
