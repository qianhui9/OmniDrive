"""Frozen selective identity and three-view surface experiments; Oracle offline."""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from scipy.stats import beta
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.association import associate
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.identity_continuity import IDENTITY_CLAIM
from .dataset import frame_from_info,load_oracle
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .evidence_oracle import identity_population
from .evidence_experiments import append_json,summary,load_inputs as previous_inputs
from .identity_surface_audit import ROOT,PRIOR,point_diagnostics,panels
from .io_utils import write_json

METHODS=dict(conservative={},dense_original=dict(dense_guard_radius_px=2,dense_motion_guard=False),
    local_r12=dict(dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=12),
    surface_observable_ablation=dict(dense_region_policy='surface_observable_ablation',dense_guard_radius_px=2),
    surface_visibility=dict(dense_region_policy='surface_visibility',dense_guard_radius_px=2))


def inputs(stage):
    if stage=='development':return previous_inputs('calibration')
    selection=json.loads((ROOT/'data_selection.json').read_text());specs=selection['splits'][stage]
    needed={t for s in specs for t in s['samples']}
    infos=pickle.load(open(selection['info_file'],'rb'))['infos'];infos=[i for i in infos if i['token'] in needed]
    with (ROOT/'model_observations.pkl').open('rb') as f:outputs=pickle.load(f)
    return specs,{i['token']:i for i in infos},outputs,load_oracle(infos,'data/nuscenes')


def identity_summary(rows):
    result={}
    for method in ('strict_foreground','selective_identity'):
        allrows=[r for r in rows if r['method']==method];selected=[r for r in allrows if r['candidate']]
        scorable=[r for r in selected if r['correct'] is not None];bad=[r for r in scorable if not r['correct']]
        scored_scenes={r['scene'] for r in scorable};badscenes={r['scene'] for r in bad}
        units={r['scene'] for r in selected};unknown=[r for r in selected if r['correct'] is None]
        upper=float(beta.ppf(.95,len(badscenes)+1,len(scored_scenes)-len(badscenes))) if scored_scenes and len(badscenes)<len(scored_scenes) else 1. if scored_scenes else None
        result[method]=dict(all_geometric_candidates=len(allrows),candidate_count=len(selected),
            candidate_coverage=len(selected)/len(allrows) if allrows else None,accepted_count=sum(r['certified'] for r in allrows),
            accepted_scorable_count=0,independently_correct_candidate_links=len(scorable)-len(bad),
            candidate_scorable_count=len(scorable),candidate_false_identity=len(bad),unscorable_candidates=len(unknown),
            conditional_candidate_FAR=len(bad)/len(scorable) if scorable else None,
            conditional_scene_event_upper95=upper,independent_candidate_scene_count=len(units),
            independent_scorable_candidate_scenes=len(scored_scenes),candidate_error_scenes=len(badscenes),
            qualified_FAR=None,qualified_upper95=None,independent_certified_scenes=0,
            correct_identity_recovery=sum(r['correct'] is True and r['candidate'] for r in allrows),
            correct_identity_rejected=sum(r['correct'] is True and not r['candidate'] for r in allrows),
            wrong_identity_rejected=sum(r['correct'] is False and not r['candidate'] for r in allrows),
            extra_producer_cpu_ms=sum(r['producer_cpu_ms'] for r in allrows),
            rejection_counts=dict(Counter(reason for r in allrows for reason in r['rejection_reasons'])),
            note='Candidates with independent identities are feasibility observations only; unscorable candidates never become certified successes')
    return result


def run_scene(stage,spec,infos,outputs,oracle):
    scene=spec['scene_id'];out=ROOT/'traces'/stage/scene
    if out.exists():raise RuntimeError('Refuse overwrite of scene results: '+str(out))
    out.mkdir(parents=True);cfg=M1Config.load('configs/worldstream_m1_independent.yaml')
    managers={name:WorldStateManager(replace(cfg,identity_continuity=name=='conservative',foreground_correspondence=name=='conservative',**settings)) for name,settings in METHODS.items()}
    previous=None;oldpoints=None;truth={};last_instance_anchor={};rows=[];identity=[];population=[];timings=[];falsepoints=[];memory=[];parity=Counter();visuals=0
    old_masks={}
    if stage=='development':
        old_masks={r['precheck']['frame_id']:r['precheck'] for r in map(json.loads,(PRIOR/'traces/calibration'/scene/'state.jsonl').read_text().splitlines())}
    for index,token in enumerate(spec['samples']):
        info=infos[token];frame=frame_from_info(info,'data/nuscenes');points=world_lidar(info)
        # Reference and model outputs are deliberately absent from runtime Precheck args.
        refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        current=outputs[token];known,statuses=identity_population(current.anchors,oracle[token])
        for name,m in managers.items():
            tick=perf_counter();r=m.precheck(frame);pre_ms=(perf_counter()-tick)*1000
            for cid,cam in frame.cameras.items():
                meta=r.invalidation_regions[cid];invalid=m.last_masks[cid];e=m.last_evidence.get(cid)
                row=dict(method=name,stage=stage,scene=scene,frame=token,camera=cid,pixels=invalid.size,
                    stable_pixels=int(decode_mask(meta['observation_stable_mask']).sum()),effective_pixels=int((~invalid).sum()),
                    changed_pixels=int(decode_mask(meta['observed_changed_mask']).sum()),unknown_pixels=int(decode_mask(meta['unknown_mask']).sum()),
                    forced=r.full_refresh_hint,camera_diagnostics=e.diagnostics if e else {},
                    interval_s=frame.timestamp-previous.timestamp if previous else None,
                    ego_displacement_m=float(np.linalg.norm(frame.ego_to_world[:3,3]-previous.ego_to_world[:3,3])) if previous else None)
                assert row['stable_pixels']+row['changed_pixels']+row['unknown_pixels']==invalid.size
                assert row['effective_pixels']==invalid.size-int(invalid.sum())
                if refs:
                    ref=dict(refs[cid]);ref['scorable']=ref['scorable']&ref['annotated']
                    row.update(score_camera(ref,invalid,e,cfg))
                    if e is not None and e.dense_source_map is not None:
                        cases=point_diagnostics(previous.cameras[cid],cam,e,invalid,ref,name,scene,token)
                        falsepoints.extend(cases)
                        for case in cases:append_json(out/'false_stable.jsonl',case)
                        if cases and visuals<8:
                            path=ROOT/'visualizations'/stage/scene/(name+'_'+str(index)+'_'+cid+'.jpg')
                            panels(previous.cameras[cid],cam,e,meta,cases,path);visuals+=1
                if old_masks and name=='conservative':
                    expected=decode_mask(old_masks[token]['invalidation_regions'][cid]['refresh_required_mask'])
                    parity['compared_camera_masks']+=1;parity['different_camera_masks']+=int(not np.array_equal(expected,invalid))
                rows.append(row);append_json(out/'pixels.jsonl',row)
                if name=='conservative':
                    full=dict(row,method='full_refresh',effective_pixels=0)
                    if refs:full.update(score_camera(ref,np.ones_like(invalid),e,cfg))
                    rows.append(full);append_json(out/'pixels.jsonl',full)
            old_anchors=list(m.store.anchors.values());source_truth=truth.copy();before=list(r.physically_certified_anchor_ids)
            tick=perf_counter();m.commit(frame,current,r);commit_ms=(perf_counter()-tick)*1000
            assert r.physically_certified_anchor_ids==before
            post=m.get_current_certifications();assert not post.certified_anchor_ids # no real qualification artifact
            diag=m.last_commit_diagnostics
            timing=dict(method=name,scene=scene,frame=token,precheck_ms=pre_ms,commit_ms=commit_ms,total_ms=pre_ms+commit_ms,
                identity_ms=sum(v['producer_cpu_ms'] for v in diag['identity_continuity_proposals']),
                foreground_ms=sum(v['producer_cpu_ms'] for v in diag['foreground_correspondence_proposals']),
                certification_ms=diag['physical_certification_ms'],association_ms=diag['association_ms'],
                surface_ms=sum(e.diagnostics.get('surface_ms',0) for e in m.last_evidence.values()),
                surface_history_bytes=sum(a.image.nbytes+mapping.nbytes for a,_,mapping in m._surface_history.values()))
            timings.append(timing);append_json(out/'timing.jsonl',timing)
            if name=='conservative':
                selected=diag['selected_prediction_indices'];preds=[current.anchors[j] for j in selected]
                pairs,costs=associate(old_anchors,preds,frame.timestamp,cfg);continued=set(map(tuple,diag['association_pairs']))
                proposals={v['anchor_id']:v for v in diag['identity_continuity_proposals']}
                fore={v['anchor_id']:v for v in diag['foreground_correspondence_proposals']}
                for i,j in pairs:
                    a=old_anchors[i]
                    if a.anchor_type!='object':continue
                    raw=selected[j];old=source_truth.get(a.anchor_id);new=known.get(raw)
                    correct=old['instance_id']==new['instance_id'] if old and new else None
                    gap=index-spec['samples'].index(a.source_frame_id) if a.source_frame_id in spec['samples'] else None
                    for method,lookup in [('strict_foreground',fore),('selective_identity',proposals)]:
                        proposal=lookup.get(a.anchor_id) if (a.anchor_id,raw) in continued else None
                        item=dict(method=method,scene=scene,frame=token,anchor_id=a.anchor_id,prediction=raw,
                            source_frame=a.source_frame_id,source_gap_frames=gap,current_status=statuses[raw]['status'],
                            source_instance=old['instance_id'] if old else None,current_instance=new['instance_id'] if new else None,
                            correct=correct,continued=(a.anchor_id,raw) in continued,
                            candidate=bool(proposal and proposal['status']=='candidate'),certified=False,
                            producer_cpu_ms=proposal['producer_cpu_ms'] if proposal else 0.,
                            rejection_reasons=proposal['rejection_reasons'] if proposal else ['geometric_assignment_ambiguity_or_history_gate'],
                            measurements=proposal.get('measurements') if proposal else None)
                        identity.append(item);append_json(out/'identity.jsonl',item)
                current_instance_anchor={}
                for raw,aid in zip(selected,diag['committed_anchor_ids']):
                    if current.anchors[raw].anchor_type!='object':continue
                    truth.pop(aid,None)
                    if raw in known:
                        truth[aid]=known[raw];current_instance_anchor[known[raw]['instance_id']]=aid
                    population.append(dict(scene=scene,frame=token,prediction=raw,anchor_id=aid,**statuses[raw],
                        fragmented=bool(raw in known and known[raw]['instance_id'] in last_instance_anchor
                            and last_instance_anchor[known[raw]['instance_id']]!=aid)))
                truth={a:t for a,t in truth.items() if a in m.store.anchors};last_instance_anchor=current_instance_anchor
                append_json(out/'state.jsonl',dict(precheck=r.to_dict(),post_commit=post.to_dict(),commit=diag))
            memory.append(dict(method=name,frame=token,anchors=len(m.store.anchors),features=len(m.store.features),
                tensor_bytes=m.store.tensor_bytes,history_bytes=timing['surface_history_bytes']))
        previous=frame;oldpoints=points
    write_json(out/'population.json',population);write_json(out/'memory.json',memory)
    write_json(out/'summary.json',dict(region={n:summary([r for r in rows if r['method']==n]) for n in [*METHODS,'full_refresh']},
        identity=identity_summary(identity),parity=dict(parity),false_stable_taxonomy=dict(Counter(r['root_cause'] for r in falsepoints))))
    write_json(out/'complete.json',dict(scene=scene,stage=stage,frames=len(spec['samples'])))
    print(stage,scene,'B candidates',sum(r['candidate'] and r['method']=='selective_identity' for r in identity),flush=True)


def stage_summary(stage):
    folders=list((ROOT/'traces'/stage).glob('*/complete.json'));pixels=[];identity=[]
    for p in folders:
        pixels.extend(map(json.loads,(p.parent/'pixels.jsonl').read_text().splitlines()))
        identity.extend(map(json.loads,(p.parent/'identity.jsonl').read_text().splitlines()))
    result=dict(region={n:summary([r for r in pixels if r['method']==n]) for n in [*METHODS,'full_refresh']},identity=identity_summary(identity))
    write_json(ROOT/(stage+'_summary.json'),result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['development','calibration','heldout','final_test']);a=parser.parse_args()
    cv2.setNumThreads(1)
    if a.stage in ('heldout','final_test') and not (ROOT/'frozen_validation_protocol.json').exists():raise RuntimeError('Freeze selected methods first')
    specs,infos,outputs,oracle=inputs(a.stage)
    for spec in specs:run_scene(a.stage,spec,infos,outputs,oracle)
    result=stage_summary(a.stage)
    if a.stage=='calibration':
        b=result['identity']['selective_identity'];v=result['region']['surface_visibility'];base=result['region']['conservative']
        identity_ok=b['independently_correct_candidate_links']>0 and b['conditional_candidate_FAR'] is not None and b['conditional_candidate_FAR']<=.05
        region_ok=(v['critical_change_recall'] is not None and v['critical_change_recall']>=.998 and v['critical_point_recall']>=.998
            and v['sparse_reference_far'] is not None and v['sparse_reference_far']<=.05 and v['far_upper95_iid_descriptive']<=.05
            and v['accepted_scene_support']>=3 and v['effective_stable_area']>0 and v['effective_stable_area']>=base['effective_stable_area'])
        selection=dict(identity_feasibility_candidate='selective_identity' if identity_ok else None,
            region_candidate='surface_visibility' if region_ok else None,qualified_physical_producer=None,
            no_final_reselection=True,identity_feasibility_not_deployment=True)
        write_json(ROOT/'algorithm_selection.json',selection)
        paths=[*Path('worldstream/m1').glob('*.py'),Path(__file__),Path('tools/worldstream_m1/coverage_oracle.py'),Path('tools/worldstream_m1/evidence_oracle.py')]
        write_json(ROOT/'frozen_validation_protocol.json',dict(selection=selection,methods=METHODS,
            experiment_protocol_sha256=hashlib.sha256((ROOT/'experiment_protocol.json').read_bytes()).hexdigest(),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            frozen_before_heldout_and_final_quality=True))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
