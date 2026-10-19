"""Independent-manager candidate-base and online-guard ablations on nuScenes."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager
from .correspondence_experiments import ROOT, source_manifest
from .evidence_experiments import load_inputs, append_json
from .dataset import frame_from_info
from .coverage_oracle import world_lidar, prepare_pair, score_camera
from .io_utils import write_json
from .quality_coverage_metrics import region_metrics

OVERRIDES = dict(recovery_conservative_neighbourhood=dict(dense_guard_radius_px=4),
                 recovery_no_motion_sweep=dict(dense_motion_guard=False))


def scene(spec):
    infos,outputs,oracle=_INPUTS
    root=ROOT/'validation_traces'/spec['scene_id'];root.mkdir(parents=True,exist_ok=False)
    cfg=M1Config.load('configs/worldstream_m1_correspondence_recovery.yaml')
    managers={name:WorldStateManager(replace(cfg,**change)) for name,change in OVERRIDES.items()}
    previous=None;oldpoints=None
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token])
        refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg) if previous else None
        for name,m in managers.items():
            result=m.precheck(frame)
            for cid,cam in frame.cameras.items():
                e=m.last_evidence[cid];invalid=m.last_masks[cid]
                row=dict(method=name,scene=frame.scene_id,log_token=spec['log_token'],frame=token,camera=cid,
                    pixels=invalid.size,stable_pixels=int(e.stable_mask.sum()),effective_pixels=int((~invalid).sum()),
                    changed_pixels=int(e.changed_mask.sum()),unknown_pixels=int((invalid&~e.changed_mask).sum()),
                    forced_refresh=result.full_refresh_hint)
                if refs:
                    ref=dict(refs[cid]);ref['scorable'] &= ref['annotated'];row.update(score_camera(ref,invalid,e,cfg))
                append_json(root/'pixels.jsonl',row)
            m.commit(frame,outputs[token],result)
            assert not m.get_current_certifications().certified_anchor_ids
        previous=frame;oldpoints=points
    assert source_manifest()==json.loads((ROOT/'development_source_freeze.json').read_text())['files']
    write_json(root/'complete.json',dict(scene=spec['scene_id'],frames=len(spec['samples'])))
    print('guard/base ablations',spec['scene_id'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    logs={s['token']:s['log_token'] for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    specs=[dict(s,log_token=logs[s['scene_id']]) for s in specs]
    pin=ROOT/'validation_ablation_freeze.json'
    if not pin.exists():write_json(pin,dict(overrides=OVERRIDES,files=source_manifest(),population='exploratory_reused_population'))
    else:assert json.loads(pin.read_text())['files']==source_manifest()
    specs=[s for s in specs if not (ROOT/'validation_traces'/s['scene_id']/'complete.json').exists()]
    global _INPUTS
    _INPUTS=infos,outputs,oracle
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,specs))
    rows=[]
    for complete in (ROOT/'validation_traces').glob('*/complete.json'):
        rows.extend(map(json.loads,(complete.parent/'pixels.jsonl').read_text().splitlines()))
    write_json(ROOT/'validation_ablation_summary.json',{name:region_metrics([r for r in rows if r['method']==name]) for name in OVERRIDES})


if __name__=='__main__':main()
