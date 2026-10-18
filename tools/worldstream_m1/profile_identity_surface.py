"""Isolated cost and output comparison of claim-specific experiment paths."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .identity_surface_audit import ROOT
from .dataset import frame_from_info,prediction_outputs
from .io_utils import read_prediction_subset,write_json
from .independent_refinement import baseline_manager,distribution


def main():
    cv2.setNumThreads(1)
    for stage in ('development','calibration','heldout','final_test'):
        if not (ROOT/(stage+'_summary.json')).exists():raise RuntimeError('Finish quality jobs before isolated timing')
    protocol=json.loads(Path('outputs/module1_independent_refinement/experiment_protocol.json').read_text())
    spec=protocol['scenes']['development_long'][0];tokens=spec['samples'][:8]
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];infos={i['token']:i for i in infos if i['token'] in tokens}
    frames=[frame_from_info(infos[t],'data/nuscenes') for t in tokens]
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',set(tokens))
    cfg=M1Config.load('configs/worldstream_m1_independent.yaml')
    settings=dict(previous_baseline={},identity_only=dict(identity_continuity=True),strict_only=dict(foreground_correspondence=True),
        both_producers=dict(identity_continuity=True,foreground_correspondence=True),
        surface_identity=dict(identity_continuity=True,dense_region_policy='surface_visibility',dense_guard_radius_px=2))
    rows=[];mask_comparisons=0;identity_comparisons=0
    for repeat in range(4):
        order=list(settings)
        if repeat%2:order.reverse()
        captures={}
        for method in order:
            config=replace(cfg,**settings[method])
            m=baseline_manager(ROOT,config) if method=='previous_baseline' else WorldStateManager(config)
            masks=[];identities=[]
            for i,frame in enumerate(frames):
                start=perf_counter();r=m.precheck(frame);p=perf_counter()
                out=prediction_outputs(predictions[frame.frame_id],config);a=perf_counter();m.commit(frame,out,r);end=perf_counter()
                d=m.last_commit_diagnostics
                masks.append({cid:v.copy() for cid,v in m.last_masks.items()})
                identities.append([{k:v[k] for k in ('anchor_id','status','rejection_reasons','measurements')} for v in d.get('identity_continuity_proposals',[])])
                if repeat and i:
                    rows.append(dict(method=method,repeat=repeat,frame=frame.frame_id,total_ms=(end-start)*1000,
                        precheck_ms=(p-start)*1000,adapter_ms=(a-p)*1000,commit_ms=(end-a)*1000,
                        foreground_ms=sum(v['producer_cpu_ms'] for v in d.get('foreground_correspondence_proposals',[])),
                        identity_ms=sum(v['producer_cpu_ms'] for v in d.get('identity_continuity_proposals',[])),
                        surface_ms=sum(e.diagnostics.get('surface_ms',0) for e in m.last_evidence.values())))
            captures[method]=(masks,identities)
        for method in ('identity_only','strict_only','both_producers'):
            for left,right in zip(captures['previous_baseline'][0],captures[method][0]):
                for cid in left:assert np.array_equal(left[cid],right[cid]);mask_comparisons+=1
        for method in ('both_producers','surface_identity'):
            assert captures['identity_only'][1]==captures[method][1]
            identity_comparisons+=len(frames)
    write_json(ROOT/'isolated_runtime_profile.json',dict(methods={name:{key:distribution([r[key] for r in rows if r['method']==name])
        for key in ('total_ms','precheck_ms','adapter_ms','commit_ms','foreground_ms','identity_ms','surface_ms')} for name in settings},
        raw=rows,warmup_repetitions=1,timed_repetitions=3,unique_frames=8,mask_comparisons=mask_comparisons,
        exact_mask_equality=True,identity_frame_comparisons=identity_comparisons,identity_decisions_and_measurements_equal=True,
        excludes='IO, Oracle, logging; fixed preloaded development workload',
        comparison='Claim-specific ablation, not disabling necessary conditions or claiming qualified speedup',
        P1_runtime_algorithm_optimization=False))
    # Full exposed development stream: memory/lifecycle check, no quality tuning.
    full_tokens=spec['samples']
    all_infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    all_infos={i['token']:i for i in all_infos if i['token'] in full_tokens}
    all_predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',set(full_tokens))
    m=WorldStateManager(replace(cfg,**settings['surface_identity']));memory=[]
    for token in full_tokens:
        frame=frame_from_info(all_infos[token],'data/nuscenes');r=m.precheck(frame)
        m.commit(frame,prediction_outputs(all_predictions[token],cfg),r)
        assert len(m._surface_history)<=len(frame.cameras)
        memory.append(dict(frame=token,anchors=len(m.store.anchors),features=len(m.store.features),tensor_bytes=m.store.tensor_bytes,
            retained_history_bytes=sum(a.image.nbytes+mapping.nbytes for a,_,mapping in m._surface_history.values()),
            oldest_retained_camera_timestamp=min((a.timestamp for a,_,_ in m._surface_history.values()),default=None),
            current_timestamp=frame.timestamp))
    write_json(ROOT/'long_sequence_memory.json',dict(frames=len(memory),scene=spec['scene_id'],raw=memory,
        bound='One older camera and one source map per camera; newest camera separately retained; no unbounded history',
        max_history_bytes=max(r['retained_history_bytes'] for r in memory),hours_long_endurance=False))
    from .profile_evidence import write_gpu_profile
    write_gpu_profile(ROOT)
    print('isolated runtime and output comparisons complete')


if __name__=='__main__':main()
