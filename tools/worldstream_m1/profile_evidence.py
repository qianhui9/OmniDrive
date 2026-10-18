"""Isolated, preloaded CPU comparison; measurement only, no P1 optimization."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .dataset import frame_from_info,prediction_outputs
from .io_utils import read_prediction_subset,write_json
from .independent_refinement import baseline_manager,distribution


def main():
    root=Path('outputs/module1_evidence_optimization');cv2.setNumThreads(1)
    # Avoid conflating contested multi-job timings with an isolated comparison.
    for stage in ('development','calibration','heldout','final_test'):
        if not (root/(stage+'_summary.json')).exists():raise RuntimeError('Complete experiments before isolated timing')
    source=json.loads(Path('outputs/module1_independent_refinement/experiment_protocol.json').read_text())
    spec=source['scenes']['development_long'][0]
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    infos={i['token']:i for i in infos if i['token'] in spec['samples'][:8]}
    frames=[frame_from_info(infos[token],'data/nuscenes') for token in spec['samples'][:8]]
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',set(infos))
    config=M1Config.load('configs/worldstream_m1_independent.yaml')
    measurements=[];fingerprints={};mask_comparisons=0
    for repeat in range(5):
        order=['baseline','new_without_producer','new_with_producer']
        if repeat%2:order.reverse()
        for name in order:
            m=baseline_manager(root,config) if name=='baseline' else WorldStateManager(replace(config,foreground_correspondence=name=='new_with_producer'))
            masks=[]
            for i,frame in enumerate(frames):
                start=perf_counter();r=m.precheck(frame);pre=perf_counter()
                out=prediction_outputs(predictions[frame.frame_id],config);adapt=perf_counter()
                m.commit(frame,out,r);end=perf_counter()
                masks.append({cid:mask.copy() for cid,mask in m.last_masks.items()})
                if repeat and i:
                    measurements.append(dict(method=name,repeat=repeat,frame=frame.frame_id,
                        total_adapter_precheck_commit_ms=(end-start)*1000,precheck_ms=(pre-start)*1000,
                        output_adapter_ms=(adapt-pre)*1000,commit_ms=(end-adapt)*1000,
                        producer_ms=sum(p['producer_cpu_ms'] for p in m.last_commit_diagnostics.get('foreground_correspondence_proposals',[]))))
            if name=='baseline':fingerprints[repeat]=masks
            else:
                # Alternating order can encounter new before baseline; compare
                # after all methods in this repetition have completed.
                fingerprints[repeat,name]=masks
        for name in ('new_without_producer','new_with_producer'):
            for old,new in zip(fingerprints[repeat],fingerprints[repeat,name]):
                for cid in old:
                    assert np.array_equal(old[cid],new[cid]);mask_comparisons+=1
    result={name:{key:distribution([r[key] for r in measurements if r['method']==name]) for key in
        ('total_adapter_precheck_commit_ms','precheck_ms','output_adapter_ms','commit_ms','producer_ms')}
        for name in ('baseline','new_without_producer','new_with_producer')}
    write_json(root/'isolated_cpu_profile.json',dict(methods=result,raw=measurements,
        warmup_repetitions=1,timed_repetitions=4,unique_frames=8,mask_comparisons=mask_comparisons,all_masks_equal=True,
        scope='Actual preloaded RGB + historical full-model detections; includes prediction adapter, excludes IO/oracle/logs',
        selection='Already exposed development stream; timing workload only',optimization_performed=False))
    write_gpu_profile(root)
    print(json.dumps(result,indent=2))


def write_gpu_profile(root):
    folder='gpu_parity_isolated' if (root/'gpu_parity_isolated/results.json').exists() else 'gpu_parity'
    gpu=[json.loads(line) for line in (root/folder/'m1_state.jsonl').read_text().splitlines()]
    profile={key:distribution([r['commit'][key] for r in gpu]) for key in
        ('m1_total_including_adapters_ms','commit_total_ms','physical_certification_ms','output_adapter_including_transfers_ms','output_gpu_to_cpu_transfer_ms')}
    for key in ('precheck_total','input_adapter_including_transfers','input_gpu_to_cpu_transfer'):
        profile[key+'_ms']=distribution([r['precheck']['diagnostics']['timings_ms'][key] for r in gpu])
    profile['foreground_producer_ms']=distribution([sum(p['producer_cpu_ms'] for p in r['commit']['foreground_correspondence_proposals']) for r in gpu])
    write_json(root/'gpu_adapter_profile.json',dict(measurements=profile,unique_frames=len(gpu),
        run_folder=folder,concurrent_M1_experiments=folder!='gpu_parity_isolated',
        scope='Actual original checkpoint GPU perception input protocol 640x640; no LLM/M2/M3; no E2E speedup claim'))


if __name__=='__main__':main()
