"""Paired 96-frame M1 ablations, offline link sampling and state equivalence."""
import argparse
import copy
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import resource
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import validate_refresh_partition
from .source_state_refinement_replay import preprocessed_frame
from .evidence_experiments import load_inputs,append_json,summary
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .final_validation_sampling import sample_links,PROTOCOL
from .io_utils import write_json

ROOT=Path('outputs/module1_final_validation_refinement')


def state_digest(manager):
    # Time instrumentation differs, all decision-bearing commit diagnostics
    # and persisted identity/provenance are compared separately from masks.
    from worldstream.m1.state_types import json_value
    anchors={aid:dict(source_frame=a.source_frame_id,source_timestamp=a.source_timestamp,
        epoch=a.source_state_epoch,version=a.state_version,observations=a.history_observations,
        geometry=asdict(a.geometry_world),history=a.association_history) for aid,a in manager.store.anchors.items()}
    commit={k:v for k,v in manager.last_commit_diagnostics.items()
        if k in ('association_pairs','rejected_ambiguous_associations','committed_anchor_ids')}
    return hashlib.sha256(json.dumps(json_value(dict(anchors=anchors,commit=commit)),sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',type=int,default=8)
    parser.add_argument('--frames',type=int,default=12);parser.add_argument('--output',default=str(ROOT/'replay'))
    args=parser.parse_args();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    specs,infos,outputs,oracle=load_inputs('development');specs=specs[:args.scenes]
    current=M1Config.load('configs/worldstream_m1_source_state_refined.yaml')
    final=M1Config.load('configs/worldstream_m1_final_validation.yaml')
    methods=dict(old_source=M1Config.load('configs/worldstream_m1_source_linked_state.yaml'),
        current_optimized=current,motion_only=replace(final,measurement_reuse=False),
        measurement_only=replace(final,motion_risk_revocation=False),final=final)
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [*Path('worldstream/m1').glob('*.py'),Path(__file__),Path('configs/worldstream_m1_final_validation.yaml')]}
    write_json(dest/'protocol.json',dict(population='exploratory_reused_population',scenes=specs,frames_per_scene=args.frames,
        methods={k:v.to_dict() for k,v in methods.items()},source_sha256=pins,sampling=PROTOCOL,
        online_GT=False,service='persistent_state_every_arm',order='method order rotates per frame'))
    rows=[];timings=[];identities=[];links=[];paired=[]
    old_failures=json.loads(Path('outputs/module1_source_state_refinement/change_first_refresh.json').read_text())['methods']['optimized']['failures']
    old_keys={(s['frame'],s['camera'],s['instance']) for s in old_failures if not s['recalled']}
    for spec in specs:
        managers={name:WorldStateManager(settings) for name,settings in methods.items()}
        previous=previous_points=None
        for ordinal,token in enumerate(spec['samples'][:args.frames]):
            frame,_=preprocessed_frame(infos[token],cfg,current)
            current_points=world_lidar(infos[token])
            refs=prepare_pair(previous,frame,previous_points,current_points,oracle[previous.frame_id],oracle[token],current) if previous else None
            # Log identity is an offline source field. Read it from nuScenes
            # scene metadata when available; no frame is called an independent log.
            log_id=infos[token].get('log_token',spec['scene_id'])
            names=list(methods);names=names[ordinal%len(names):]+names[:ordinal%len(names)]
            for name in names:
                manager=managers[name];start=perf_counter();result=manager.precheck(frame);pre_ms=(perf_counter()-start)*1000
                bundle=result.computation_state
                for cid,cam in frame.cameras.items():
                    contract=bundle.refresh_contract[cid];candidate=decode_mask(contract['reuse_candidate_mask'])
                    dirty=decode_mask(contract['dirty_hint_mask']);unknown=decode_mask(contract['unknown_mask'])
                    assert validate_refresh_partition(dirty,candidate,unknown)
                    link=bundle.observation_links[cid];e=manager.last_evidence.get(cid)
                    row=dict(method=name,scene=frame.scene_id,frame=token,camera=cid,pixels=candidate.size,
                        stable_pixels=len(link['source_points']),effective_pixels=int(candidate.sum()),changed_pixels=int(dirty.sum()),
                        unknown_pixels=int(unknown.sum()),forced_refresh=result.full_refresh_hint,
                        motion=link.get('motion_risk',{}))
                    if refs:
                        ref=dict(refs[cid]);ref['scorable']=ref['scorable']&ref['annotated']
                        score_e=copy.copy(e);mapping=e.dense_source_map.copy() if e.dense_source_map is not None else np.full((*candidate.shape,2),np.nan,np.float32)
                        p,q=link['source_points'],link['current_points'];xy=np.rint(q).astype(int)
                        if len(xy):mapping[xy[:,1],xy[:,0]]=p+xy-q
                        score_e.dense_source_map=mapping
                        row.update(score_camera(ref,~candidate,score_e,manager.config))
                        row['strict_refresh_score']=score_camera(ref,manager.last_masks[cid],e,manager.config)
                        if name in ('current_optimized','motion_only','final') and len(q):
                            samples,denominators,census=sample_links(link,candidate,refs[cid],frame,cid,log_id,name)
                            for item in samples:append_json(dest/'physical_samples.jsonl',item)
                            for item in denominators:append_json(dest/'sampling_denominators.jsonl',item)
                            for item in census:append_json(dest/'physical_reference_census.jsonl',item)
                        for event in row.get('events',[]):
                            if (token,cid,event['instance']) in old_keys:
                                paired.append(dict(method=name,scene=frame.scene_id,frame=token,camera=cid,**event))
                        if name=='final' and any((token,cid,v['instance']) in old_keys for v in row.get('events',[])):
                            d=ROOT/'critical_event_overlays';d.mkdir(exist_ok=True)
                            np.savez_compressed(d/f'final_{token}_{cid}.npz',image=cam.image,candidate=candidate,dirty=dirty,unknown=unknown,
                                motion_dirty=e.motion_risk_dirty_mask,motion_unknown=e.motion_risk_unknown_mask)
                    append_json(dest/'pixels.jsonl',row);rows.append(row)
                    links.append(dict(method=name,frame=token,camera=cid,count=len(link['source_points']),
                        eligible=int(candidate.sum()),recovery=link.get('link_recovery'),motion=link.get('motion_risk')))
                start=perf_counter();manager.commit(frame,copy.deepcopy(outputs[token]),result);commit_ms=(perf_counter()-start)*1000
                identities.append(dict(method=name,frame=token,digest=state_digest(manager),anchor_ids=list(manager.store.anchors)))
                timings.append(dict(method=name,frame=token,precheck_ms=pre_ms,commit_ms=commit_ms,total_ms=pre_ms+commit_ms,
                    precheck=result.diagnostics['timings_ms'],commit={k:v for k,v in manager.last_commit_diagnostics.items()
                    if k.endswith('_ms') or k=='association_profile'}))
            previous,previous_points=frame,current_points;print(frame.scene_id,token,'completed',flush=True)
    results={}
    for name in methods:
        selected=[s for s in rows if s['method']==name];results[name]=summary(selected)
        results[name]['minimum_ROI_refresh']=min((e['invalidation_coverage'] for s in selected for e in s.get('events',[])),default=None)
    write_json(dest/'method_comparison.json',results);write_json(dest/'timings.json',timings)
    write_json(dest/'identity.json',identities);write_json(dest/'links.json',links);write_json(dest/'paired_events.json',paired)
    write_json(dest/'completion.json',dict(status='completed',frames=len({s['frame'] for s in rows}),
        camera_observations=len({(s['frame'],s['camera']) for s in rows}),source_sha256=pins,
        rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
