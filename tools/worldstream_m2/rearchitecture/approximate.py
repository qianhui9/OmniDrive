"""Independent offline necessary-feasibility probe, with current-patch oracle costs."""
import copy
from dataclasses import replace
import json
from time import perf_counter
import numpy as np
import torch
from worldstream.m1 import WorldStateManager,M1Config,FullOutputs
from worldstream.m1.omnidrive_adapter import frame_from_model
from worldstream.m2.approximate_probe import align_patch_features
from worldstream.m2.profiler import measure
from .common import ROOT,write,backbone


@torch.no_grad()
def main():
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    cfg.data.test.test_mode=True;dataset=build_dataset(cfg.data.test)
    specs=json.load(open(ROOT/'protocol.json'))['real_scenes']
    model=backbone();rows=[];timings=[]
    for spec in specs:
        manager=WorldStateManager(replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),logging=False))
        previous=None
        for index in spec['indices']:
            batch=scatter(collate([dataset[index]],samples_per_gpu=1),[torch.cuda.current_device()])[0]
            data={k:v[0] if k in ('img','input_ids') else v[0][0].unsqueeze(0) for k,v in batch.items() if k!='img_metas'}
            metas=batch['img_metas'][0]
            frame=frame_from_model(metas,data,manager.config)
            t=perf_counter();result=manager.precheck(frame);validation=(perf_counter()-t)*1000
            images=data['img'][0]
            current=model.patch_embed(images)
            full_cost=measure(lambda:model.patch_embed(images),warmup=3,repeats=10)
            probes=[]
            for ci,cid in enumerate(frame.cameras):
                evidence=manager.last_evidence.get(cid)
                if previous is None or evidence is None:
                    probes.append(dict(camera=cid,accepted_patches=0,executed=False,reason='no_history_or_evidence'));continue
                torch.cuda.synchronize();alignment_start=perf_counter()
                candidate,report=align_patch_features(previous[ci],current[ci],evidence.dense_source_map,
                    ~manager.last_masks[cid])
                torch.cuda.synchronize();report['alignment_wall_ms']=(perf_counter()-alignment_start)*1000
                report['camera']=cid;probes.append(report)
            rows.append(dict(scene=spec['scene'],index=index,patch_oracle_timing=full_cost,
                m1_precheck_wall_ms=validation,probes=probes,
                note='Geometry-only optimistic probe: no detection/map exclusion anchors, so cannot establish deployed task validity'))
            # Current actual observations supply the next source. No current oracle
            # feature is claimed as a historical speedup; patch forwards are probe costs.
            manager.commit(frame,FullOutputs([],source='offline_geometry_only_no_perception_anchors'),result);previous=current.clone()
    feasible=any(r['patch_oracle_timing']['wall_ms']>r['m1_precheck_wall_ms'] and
        any(p.get('accepted_patches',0)>0 for p in r['probes']) for r in rows)
    accepted=sum(p.get('accepted_patches',0) for r in rows for p in r['probes'])
    reason=(f'No complete observation-supported patch in these {len(rows)} frames. No aligned feature or downstream approximation was scored.'
        if not accepted else ('Necessary cost condition met in some probes, but downstream quality/state proof is absent; no online approval.' if feasible else
            'Even eliminating the entire patch embedding cannot pay current observation validation on frames with reusable candidates; partial aligned patches save less. No online approval.'))
    write(ROOT/'approximate_probe_results.json',dict(status='offline_necessary_feasibility_only',rows=rows,
        go=False,optimistic_patch_saving_exceeds_validation=feasible,
        accepted_patch_count=accepted,feature_error_status='not_scored_no_reusable_patch' if not accepted else 'per_candidate_errors_recorded',
        reason=reason,
        downstream_test='not_run_quality_unverified' if feasible else 'not_run_after_cost_no_go',early_window_warp='not_implemented_RoPE_and_cross_token_dependency_unproven',
        current_position_embedding='Alignment is before abs position; never reuse historical position embedding',
        source='actual current-to-history M1 dense map; geometry-only optimistic bound, independent from full M1 pipeline'))
    print('approximate necessary probe finished; no online admission',flush=True)

if __name__=='__main__':main()
