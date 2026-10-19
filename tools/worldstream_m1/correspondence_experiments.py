"""nuScenes + original OmniDrive + M1 replay; references used only offline.

All logical ablations share identical label-free measurements and committed
sources. Full original perception runs every frame; no mask controls Commit.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import copy
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import unexplained_regions
from worldstream.m1.surface_visibility_constraints import candidate_masks
from .dataset import frame_from_info
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .evidence_oracle import identity_population
from .evidence_experiments import load_inputs,append_json
from .quality_coverage_metrics import region_metrics,identity_metrics
from .io_utils import write_json

ROOT=Path('outputs/module1_correspondence_recovery_optimization')
BASELINE=Path('outputs/module1_physical_observability_optimization')
PRIOR=Path('outputs/module1_quality_coverage_refinement')


def source_manifest():
    paths=[*Path('worldstream/m1').glob('*.py'),Path('configs/worldstream_m1_correspondence_recovery.yaml'),Path(__file__)]
    paths += [Path('worldstream/m1/geometry_adaptive_patch.py'), Path('tools/worldstream_m1/correspondence_diagnostics.py')]
    paths += [Path('tools/worldstream_m1')/p for p in ('coverage_oracle.py','evidence_oracle.py','dataset.py',
        'evidence_experiments.py','quality_coverage_metrics.py','io_utils.py')]
    return {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def variants(e):
    if e.constraint_measurements is not None:return candidate_masks(e.constraint_measurements)
    return {name:np.zeros_like(e.stable_mask) for name in ('surface_constraints','constraints_two_view_only',
        'constraints_temporal_only','constraints_camera_only','constraints_no_ambiguity',
        'constraints_no_depth_order','constraints_observation_base', 'geometric_adaptive_patch',
        'joint_multi_observation','correspondence_recovery','recovery_no_extra_observation','recovery_h0_only',
        'recovery_no_depth_order','recovery_discriminating_third_only','recovery_temporal_only','recovery_camera_only')}


def point_details(e, old, cam, frame, history, cfg, selected):
    """Actual per-hypothesis provenance for sparse audit, after online Precheck."""
    from worldstream.m1.surface_observability import observe_points, PatchObservation, corroborating_views, baseline_angle
    from worldstream.m1.geometry_adaptive_patch import GeometryAdaptivePatch
    from worldstream.m1.surface_visibility_constraints import joint_exclusions
    from .observability_diagnostics import point_record
    if not len(selected) or e.dense_source_map is None:return []
    q=np.asarray(selected,float)
    p=e.dense_source_map[q[:,1].astype(int),q[:,0].astype(int)]
    observed=observe_points(old,cam,p,q,cfg,frame.cameras,history)
    patch=GeometryAdaptivePatch(cam,cfg);old_patch=PatchObservation(old,cfg)
    geometry=patch.fit(old_patch,q,observed['world'],e.dense_source_map,e.constraint_measurements['observation_base'])
    views,_=corroborating_views(cam,frame.cameras,history,cfg)
    actual=[]
    for kind,view in views:
        other=PatchObservation(view,cfg);obs=patch.compare(other,q,observed['world'],geometry)
        supported=obs['supported']&observed['valid']&(baseline_angle(observed['world'],cam,view)>=cfg.min_triangulation_angle_deg)
        actual.append((kind,view,other,supported))
    excluded,discriminating,counts=joint_exclusions(q,observed['world'],observed['alternative_world'],
        observed['alternative_incompatible'],actual,patch,geometry,old,p,cfg)
    return [dict(pixel=q[i],source_uv=p[i],measurements=point_record(observed,i),
        local_geometry={k:v[i] for k,v in geometry.items() if k!='offsets'},
        measured_joint_excluded_legacy_hypotheses=excluded[i],discriminating_third=bool(discriminating[i]),
        joint_scope='Only actually tracked local third-view footprint; missing support is not negative evidence') for i in range(len(q))]


def scene(spec):
    infos,outputs,oracle=_INPUTS;folder=ROOT/'traces/development'/spec['scene_id']
    folder.mkdir(parents=True,exist_ok=False)
    cfg=M1Config.load('configs/worldstream_m1_correspondence_recovery.yaml')
    m=WorldStateManager(cfg);previous=None;oldpoints=None;truth={};parity=Counter()
    old_points={(r['frame'],r['camera'],r['reference_index']) for r in _OLD_POINTS if r['scene']==spec['scene_id'] and 'surface_consensus' in r['populations']}
    archived={}
    for r in map(json.loads,(PRIOR/'traces/development'/spec['scene_id']/'identity.jsonl').read_text().splitlines()):
        if r['method']=='temporal_object':archived[r['frame'],r['anchor_id'],r['prediction']]=r
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token]);current=outputs[token]
        refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        result=m.precheck(frame)
        for cid,cam in frame.cameras.items():
            e=m.last_evidence[cid];masks=variants(e)
            if refs:
                ref=dict(refs[cid]);ref['scorable'] &= ref['annotated'];xy=ref['xy'];idx=np.flatnonzero(ref['scorable'])
            for name,mask in masks.items():
                observed=copy.copy(e);observed.stable_mask=mask.copy();observed.stable_candidate_mask=None
                meta,invalid=unexplained_regions(cam,[],[],cfg,result.full_refresh_hint,observed)
                if name=='correspondence_recovery':assert np.array_equal(invalid,m.last_masks[cid])
                row=dict(method=name,scene=frame.scene_id,log_token=spec['log_token'],frame=token,camera=cid,
                    pixels=invalid.size,stable_pixels=int(decode_mask(meta['observation_stable_mask']).sum()),
                    effective_pixels=int((~invalid).sum()),changed_pixels=int(e.changed_mask.sum()),
                    unknown_pixels=int(decode_mask(meta['unknown_mask']).sum()),forced_refresh=result.full_refresh_hint)
                if name=='correspondence_recovery':row['camera_diagnostics']=e.diagnostics
                if refs:row.update(score_camera(ref,invalid,e,cfg))
                append_json(folder/'pixels.jsonl',row)
            if refs:
                for ri in idx:
                    x,y=xy[ri];in_old=(token,cid,int(ri)) in old_points
                    if not masks['constraints_observation_base'][y,x] and not in_old:continue
                    uv=e.dense_source_map[y,x] if e.dense_source_map is not None else np.array([np.nan,np.nan])
                    err=float(np.linalg.norm(uv-ref['source_uv'][ri]));bad=not np.isfinite(err) or err>2 or bool(ref['moving'][ri])
                    append_json(folder/'point_decisions.jsonl',dict(scene=frame.scene_id,log_token=spec['log_token'],frame=token,
                        camera=cid,reference_index=int(ri),pixel=[int(x),int(y)],false=bad,
                        error_px=err if np.isfinite(err) else None,old_consensus_accepted=in_old,forced_refresh=result.full_refresh_hint,
                        source_uv=uv,
                        accepted={name:bool(mask[y,x] and not result.full_refresh_hint) for name,mask in masks.items()},
                        measured={k:bool(v[y,x]) for k,v in (e.constraint_measurements or {}).items()}))
                selected=[xy[ri] for ri in idx if masks['constraints_observation_base'][xy[ri,1],xy[ri,0]] and not result.full_refresh_hint]
                for detail in point_details(e,previous.cameras[cid],cam,frame,m._surface_history.get(cid),cfg,selected):
                    append_json(folder/'hypotheses.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,
                        source_frame=previous.frame_id,log_token=spec['log_token'],**detail))
                # Four previously failing ROI cases: save actual before/after
                # masks and RGB. ROI labels are read only after online Precheck.
                failing=[r for r in _ROI if r['frame']==token and r['camera']==cid]
                for j,old in enumerate(failing):
                    event=next(v for v in score_camera(ref,m.last_masks[cid],e,cfg)['events'] if v['instance']==old['event']['instance'])
                    append_json(folder/'roi_pairs.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,
                        old_event=old['event'],new_event=event,online_diagnostics=e.diagnostics['surface_constraints'],
                        exact_refresh_mask_scored=True,GT_used_online=False))
                    canvas=cam.image.copy();canvas[~m.last_masks[cid]]=(.4*canvas[~m.last_masks[cid]]+.6*np.array([0,220,0])).astype(np.uint8)
                    x0,y0,x1,y1=np.rint(event['region']).astype(int);cv2.rectangle(canvas,(x0,y0),(x1,y1),(255,30,30),2)
                    path=ROOT/'visualizations/roi_pairs'/f'{token}_{cid}_{j}.png';path.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(path),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
        oldmap=m.store.anchors.copy();known,_=identity_population(current.anchors,oracle[token]);m.commit(frame,current,result)
        assert not m.get_current_certifications().certified_anchor_ids
        diag=m.last_commit_diagnostics
        for p in diag.get('identity_continuity_proposals',[]):
            aid=p['anchor_id'];raw=p['prediction_index'];a=oldmap[aid];before=truth.get(aid);now=known.get(raw)
            row=dict(method='temporal_object',scene=frame.scene_id,log_token=spec['log_token'],frame=token,
                source_frame=a.source_frame_id,source_timestamp=a.source_timestamp,current_timestamp=frame.timestamp,
                anchor_id=aid,prediction=raw,source_instance=before['instance_id'] if before else None,
                current_instance=now['instance_id'] if now else None,correct=(before['instance_id']==now['instance_id']) if before and now else None,
                candidate=p['status']=='candidate',certified=False,rejection_reasons=p['rejection_reasons'],
                measurements=p.get('measurements'),qualification_reasons=p.get('qualification_reasons'),
                recovered_short_gap=bool(p.get('recovered_short_gap')),resolved_competition=bool(p.get('resolved_competition')),
                acceptance_path=p.get('acceptance_path','original_producer'),online_class=a.semantic_class,
                online_source_age_s=frame.timestamp-a.source_timestamp,
                online_distance_m=float(np.linalg.norm(a.geometry_world.center-frame.ego_to_world[:3,3])),
                online_displacement_m=float(np.linalg.norm(current.anchors[raw].geometry_world.center-a.geometry_world.center)))
            expected=archived.get((token,aid,raw));parity['identity_edges']+=1
            parity['identity_differences']+=expected is None or any(row[k]!=expected[k] for k in ('candidate','correct','acceptance_path','recovered_short_gap'))
            append_json(folder/'identity.jsonl',row)
        selected=diag['selected_prediction_indices']
        for raw,aid in zip(selected,diag['committed_anchor_ids']):
            truth.pop(aid,None)
            if raw in known:truth[aid]=known[raw]
        truth={a:t for a,t in truth.items() if a in m.store.anchors}
        append_json(folder/'state.jsonl',dict(frame=token,qualified=0,duplicate_history_ids=len(diag['committed_anchor_ids'])-len(set(diag['committed_anchor_ids']))))
        previous=frame;oldpoints=points
    assert not parity['identity_differences'],parity
    if source_manifest()!=json.loads((ROOT/'development_source_freeze.json').read_text())['files']:raise RuntimeError('Source changed during replay')
    write_json(folder/'complete.json',dict(scene=spec['scene_id'],frames=len(spec['samples']),parity=dict(parity)))
    print('development',spec['scene_id'],dict(parity),flush=True)


def summarize():
    rows=[];identities=[]
    for p in sorted((ROOT/'traces/development').glob('*/complete.json')):
        rows.extend(map(json.loads,(p.parent/'pixels.jsonl').read_text().splitlines()))
        identities.extend(map(json.loads,(p.parent/'identity.jsonl').read_text().splitlines()))
    result=dict(population='exploratory_reused_population',status='Run' if rows else 'Not Run',
        region={n:region_metrics([r for r in rows if r['method']==n]) for n in sorted({r['method'] for r in rows})},
        identity=dict(temporal_object=identity_metrics(identities,5798)))
    write_json(ROOT/'development_summary.json',result)
    print(json.dumps({k:{n:{key:v[key] for key in ('effective_stable_area_ratio','sparse_reference_far','minimum_roi_event_refresh_ratio','accepted_scorable_points','false_stable_points')} for n,v in result[k].items()} for k in ['region']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--scene-index',type=int)
    p.add_argument('--freeze-source',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    if a.summarize_only:summarize();return
    if a.freeze_source:
        path=ROOT/'development_source_freeze.json'
        if path.exists():raise RuntimeError('Preserve existing source freeze')
        write_json(path,dict(population='exploratory_reused_population',files=source_manifest(),
            mechanism='spatially held-out inverse-depth plane and joint observed third-ray contradiction',quality_tuning=False))
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    logs={r['token']:r['log_token'] for r in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    specs=[dict(s,log_token=logs[s['scene_id']]) for s in specs]
    if a.scene_index is not None:specs=[specs[a.scene_index]]
    else:specs=[s for s in specs if not (ROOT/'traces/development'/s['scene_id']/'complete.json').exists()]
    global _INPUTS,_OLD_POINTS,_ROI
    _INPUTS=(infos,outputs,oracle)
    _OLD_POINTS=json.loads((BASELINE/'surface_ambiguity_analysis.json').read_text())['records']
    _ROI=json.loads((BASELINE/'critical_roi_diagnosis.json').read_text())['events']
    if a.workers>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,specs))
    else:
        for s in specs:scene(s)
    summarize()


if __name__=='__main__':main()
