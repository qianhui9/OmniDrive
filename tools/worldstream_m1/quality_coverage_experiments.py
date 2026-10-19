"""Frozen-protocol M1-only replay, with reference labels strictly offline."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import cv2
import numpy as np

from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.association import associate
from worldstream.m1.mask_codec import decode_mask
from .dataset import frame_from_info, load_oracle
from .coverage_oracle import world_lidar, prepare_pair, score_camera
from .evidence_oracle import identity_population
from .evidence_experiments import append_json, load_inputs
from .identity_surface_audit import point_diagnostics, panels
from .quality_coverage_audit import ROOT, PRIOR, sha
from .quality_coverage_metrics import identity_metrics, region_metrics
from .io_utils import write_json


def methods():
    # The parameters are physical pixel-domain tests fixed before evaluation,
    # not a quality-tuned score or search grid.
    return dict(
        conservative=dict(identity_continuity=True,foreground_correspondence=True),
        dense_original=dict(dense_guard_radius_px=2,dense_motion_guard=False),
        local_r12=dict(dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=12),
        surface_visibility=dict(dense_region_policy='surface_visibility',dense_guard_radius_px=2),
        surface_observable_ablation=dict(dense_region_policy='surface_observable_ablation',dense_guard_radius_px=2),
        surface_local_plane=dict(dense_region_policy='surface_local_plane',dense_guard_radius_px=2),
        surface_no_history=dict(dense_region_policy='surface_local_plane',surface_use_history=False,dense_guard_radius_px=2),
        surface_no_boundary_guard=dict(dense_region_policy='surface_local_plane',surface_boundary_guard=False,dense_guard_radius_px=2),
        temporal_identity=dict(identity_continuity=True,identity_method='temporal'),
        identity_no_history=dict(identity_continuity=True,identity_method='temporal',identity_use_history=False),
        identity_no_competition=dict(identity_continuity=True,identity_method='temporal',identity_resolve_competition=False))


def source_manifest():
    paths=[*Path('worldstream/m1').glob('*.py'),*Path('tools/worldstream_m1').glob('*.py'),
           Path('configs/worldstream_m1_quality_coverage.yaml')]
    return {str(p):sha(p) for p in sorted(paths)}


def get_inputs(stage):
    if stage=='development':return load_inputs('calibration')
    selection=json.loads((ROOT/'data_selection.json').read_text());specs=selection['splits'][stage]
    wanted={t for s in specs for t in s['samples']}
    if not wanted:return specs,{},{},{}
    infos=[i for i in pickle.load(open(selection['info_file'],'rb'))['infos'] if i['token'] in wanted]
    outputs=pickle.load(open(ROOT/'model_observations.pkl','rb'))
    return specs,{i['token']:i for i in infos},outputs,load_oracle(infos,'data/nuscenes')


def run_scene(stage,spec,infos,outputs,oracle):
    folder=ROOT/'traces'/stage/spec['scene_id']
    if folder.exists():raise RuntimeError('Refuse to overwrite scene '+str(folder))
    folder.mkdir(parents=True)
    cfg=M1Config.load('configs/worldstream_m1_independent.yaml')
    managers={k:WorldStateManager(replace(cfg,**v)) for k,v in methods().items()}
    truths={k:{} for k in managers};previous=None;oldpoints=None;parity=Counter();visuals=0
    logs={s['token']:s['log_token'] for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    log=logs[spec['scene_id']]
    archived={}
    p=PRIOR/'traces/development'/spec['scene_id']/'pixels.jsonl'
    if stage=='development' and p.exists():
        archived={(r['method'],r['frame'],r['camera']):r for r in map(json.loads,p.read_text().splitlines())}
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token]);current=outputs[token]
        refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        known,statuses=identity_population(current.anchors,oracle[token])
        for name,m in managers.items():
            result=m.precheck(frame);old=list(m.store.anchors.values());truth=truths[name].copy()
            for cid,cam in frame.cameras.items():
                e=m.last_evidence.get(cid);meta=result.invalidation_regions[cid];invalid=m.last_masks[cid]
                row=dict(method=name,scene=frame.scene_id,log_token=log,frame=token,camera=cid,pixels=invalid.size,
                    stable_pixels=int(decode_mask(meta['observation_stable_mask']).sum()),effective_pixels=int((~invalid).sum()),
                    changed_pixels=int(decode_mask(meta['observed_changed_mask']).sum()),unknown_pixels=int(decode_mask(meta['unknown_mask']).sum()),
                    forced_refresh=result.full_refresh_hint,camera_diagnostics=e.diagnostics if e else {})
                assert row['stable_pixels']+row['changed_pixels']+row['unknown_pixels']==invalid.size
                if refs:
                    ref=dict(refs[cid]);ref['scorable']=ref['scorable']&ref['annotated']
                    row.update(score_camera(ref,invalid,e,cfg))
                    if e is not None and e.dense_source_map is not None and name in ('surface_local_plane','surface_no_history','surface_no_boundary_guard'):
                        cases=point_diagnostics(previous.cameras[cid],cam,e,invalid,ref,name,frame.scene_id,token)
                        for case in cases:append_json(folder/'false_stable.jsonl',dict(case,log_token=log))
                        if cases and visuals<3:
                            panels(previous.cameras[cid],cam,e,meta,cases,ROOT/'visualizations'/stage/frame.scene_id/(name+'_'+token+'_'+cid+'.jpg'));visuals+=1
                if (name,token,cid) in archived:
                    expected=archived[name,token,cid]
                    keys=['pixels','stable_pixels','effective_pixels','changed_pixels','unknown_pixels','accepted_points','false_points','critical_points','missed_critical_points','events']
                    from worldstream.m1.state_types import json_value
                    parity['compared']+=1;parity['different']+=any(json_value(row.get(k))!=expected.get(k) for k in keys)
                append_json(folder/'pixels.jsonl',row)
                if name=='conservative':
                    full=dict(row,method='full_refresh',stable_pixels=0,changed_pixels=0,unknown_pixels=invalid.size,effective_pixels=0)
                    if refs:full.update(score_camera(ref,np.ones_like(invalid),e,cfg))
                    append_json(folder/'pixels.jsonl',full)
            pre=result.to_dict();m.commit(frame,current,result);post=m.get_current_certifications()
            assert result.physically_certified_anchor_ids==pre['physically_certified_anchor_ids']
            assert not post.certified_anchor_ids, 'No real producer qualification is supplied for research replay'
            diag=m.last_commit_diagnostics;selected=diag['selected_prediction_indices'];preds=[current.anchors[j] for j in selected]
            pairs,costs=associate(old,preds,frame.timestamp,cfg)
            if name in ('conservative','temporal_identity','identity_no_history','identity_no_competition'):
                proposals=diag['identity_continuity_proposals'];lookup={(r['anchor_id'],r.get('prediction_index')):r for r in proposals}
                oldmap={a.anchor_id:a for a in old}
                records=[]
                if name=='conservative':
                    byaid={r['anchor_id']:r for r in proposals};fore={r['anchor_id']:r for r in diag['foreground_correspondence_proposals']}
                    continued=set(map(tuple,diag['association_pairs']))
                    for i,j in pairs:
                        a=old[i]
                        if a.anchor_type!='object':continue
                        for method,table in [('selective_identity',byaid),('strict_foreground',fore),('hungarian_geometry',{})]:
                            proposal=table.get(a.anchor_id) if (a.anchor_id,selected[j]) in continued else None
                            records.append((method,a.anchor_id,selected[j],proposal,method=='hungarian_geometry'))
                else:
                    # Every original legal edge was measured, including geometrically
                    # rejected/short-gap sources. The denominator stays the baseline.
                    for proposal in proposals:
                        records.append((name,proposal['anchor_id'],proposal['prediction_index'],proposal,False))
                for method,aid,raw,proposal,geometry in records:
                    a=oldmap[aid];before=truth.get(aid);now=known.get(raw)
                    item=dict(method=method,scene=frame.scene_id,log_token=log,frame=token,source_frame=a.source_frame_id,
                        source_timestamp=a.source_timestamp,current_timestamp=frame.timestamp,anchor_id=aid,prediction=raw,
                        source_instance=before['instance_id'] if before else None,current_instance=now['instance_id'] if now else None,
                        correct=(before['instance_id']==now['instance_id']) if before and now else None,
                        candidate=geometry or bool(proposal and proposal['status']=='candidate'),certified=False,
                        rejection_reasons=proposal['rejection_reasons'] if proposal else ([] if geometry else ['geometric_assignment_ambiguity_or_history_gate']),
                        measurements=proposal.get('measurements') if proposal else None,
                        qualification_reasons=proposal.get('qualification_reasons') if proposal else None,
                        recovered_short_gap=bool(proposal and proposal.get('recovered_short_gap')),
                        resolved_competition=bool(proposal and proposal.get('resolved_competition')))
                    item['acceptance_path']=proposal.get('acceptance_path','original_producer') if proposal else None
                    append_json(folder/'identity.jsonl',item)
                if name=='conservative':
                    for raw,status in statuses.items():
                        append_json(folder/'population.jsonl',dict(scene=frame.scene_id,log_token=log,frame=token,prediction=raw,**status,
                            matched_geometric=any(selected[j]==raw for _,j in pairs)))
                append_json(folder/'state.jsonl',dict(method=name,frame=token,precheck=pre,commit=diag,post_commit=post.to_dict()))
            for raw,aid in zip(selected,diag['committed_anchor_ids']):
                truths[name].pop(aid,None)
                if raw in known:truths[name][aid]=known[raw]
            truths[name]={a:t for a,t in truths[name].items() if a in m.store.anchors}
        previous=frame;oldpoints=points
    write_json(folder/'baseline_parity.json',dict(parity))
    write_json(folder/'complete.json',dict(stage=stage,scene=spec['scene_id'],frames=len(spec['samples']),sources=source_manifest()))
    print(stage,spec['scene_id'],'complete',dict(parity),flush=True)


def summarize(stage):
    rows=[];identities=[]
    for path in sorted((ROOT/'traces'/stage).glob('*/complete.json')):
        rows.extend(map(json.loads,(path.parent/'pixels.jsonl').read_text().splitlines()))
        identities.extend(map(json.loads,(path.parent/'identity.jsonl').read_text().splitlines()))
    denominator=sum(r['method']=='selective_identity' for r in identities)
    result=dict(stage=stage,status='Run' if rows else 'Not Run',
        region={name:region_metrics([r for r in rows if r['method']==name]) for name in sorted({r['method'] for r in rows})},
        identity={name:identity_metrics([r for r in identities if r['method']==name],denominator) for name in sorted({r['method'] for r in identities})})
    write_json(ROOT/(stage+'_summary.json'),result)
    if stage=='development':
        write_json(ROOT/'identity_proposal_results.json',result['identity'])
        write_json(ROOT/'identity_risk_coverage.json',dict(stage=stage,methods=result['identity']))
        write_json(ROOT/'region_risk_coverage.json',dict(stage=stage,methods=result['region']))
        write_json(ROOT/'surface_method_ablation.json',{k:v for k,v in result['region'].items() if k.startswith('surface_')})
        baseline={(r['scene'],r['source_frame'],r['frame'],r['prediction']) for r in identities if r['method']=='selective_identity' and r['candidate']}
        new=[r for r in identities if r['method']=='temporal_identity' and r['candidate'] and (r['scene'],r['source_frame'],r['frame'],r['prediction']) not in baseline]
        write_json(ROOT/'identity_recovery_results.json',dict(new_vs_selective=len(new),correct=sum(r['correct'] is True for r in new),wrong=sum(r['correct'] is False for r in new),unscorable=sum(r['correct'] is None for r in new),records=new))
        write_json(ROOT/'identity_false_acceptance_cases.json',[r for r in identities if r['candidate'] and r['correct'] is False])
        for key in ('scene','camera','frame'):
            grouped=defaultdict(list)
            for row in rows:grouped[row['method'],row[key]].append(row)
            write_json(ROOT/('region_quality_by_'+key+'.json'),{method:{group:region_metrics(v) for (n,group),v in grouped.items() if n==method} for method in result['region']})
    return result


def _scene_worker(spec):
    stage,infos,outputs,oracle=_WORKER_INPUTS
    run_scene(stage,spec,infos,outputs,oracle)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','calibration','heldout','final_test'],required=True)
    p.add_argument('--scene-index',type=int);p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    cv2.setNumThreads(1)
    if a.stage in ('heldout','final_test'):
        frozen=json.loads((ROOT/'frozen_validation_protocol.json').read_text())
        if frozen['source_sha256']!=source_manifest():raise RuntimeError('Frozen source differs; no heldout/final evaluation allowed')
    if a.summarize_only:summarize(a.stage);return
    specs,infos,outputs,oracle=get_inputs(a.stage)
    if a.scene_index is not None:specs=[specs[a.scene_index]]
    if a.workers>1 and specs:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        global _WORKER_INPUTS
        _WORKER_INPUTS=(a.stage,infos,outputs,oracle)
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            list(pool.map(_scene_worker,specs))
    else:
        for spec in specs:run_scene(a.stage,spec,infos,outputs,oracle)
    if a.scene_index is None:summarize(a.stage)


if __name__=='__main__':main()
