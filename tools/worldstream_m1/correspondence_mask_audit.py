"""Replay exact pixel intersections and conservative tile containment."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.geometry import region_slices
from worldstream.m1.mask_codec import encode_mask
from .correspondence_experiments import ROOT,BASELINE,variants,source_manifest
from .correspondence_diagnostics import sha
from .evidence_experiments import load_inputs,append_json
from .dataset import frame_from_info
from .io_utils import write_json


def scene(spec):
    infos,outputs,_=_INPUTS;cfg=M1Config.load('configs/worldstream_m1_correspondence_recovery.yaml')
    manager=WorldStateManager(cfg);counts=defaultdict(Counter);roi=[];cameras=0
    folder=ROOT/'mask_audit_traces'/spec['scene_id'];folder.mkdir(parents=True,exist_ok=False)
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');result=manager.precheck(frame)
        for cid,cam in frame.cameras.items():
            masks=variants(manager.last_evidence[cid])
            effective={name:mask&~np.full_like(mask,result.full_refresh_hint) for name,mask in masks.items()}
            full=effective['correspondence_recovery'];refresh=manager.last_masks[cid]
            assert np.array_equal(full,~refresh)
            tile=np.zeros_like(refresh)
            for box in result.invalidation_regions[cid]['regions']:tile[region_slices(box,cam.image.shape)]=True
            assert (tile|~refresh).all()
            cameras+=1
            for name,mask in effective.items():
                counts[name].update(total_pixels=mask.size,common_pixels=int((full&mask).sum()),
                    new_only_pixels=int((full&~mask).sum()),baseline_only_pixels=int((mask&~full).sum()))
            # Offline historical ROI metadata is deliberately consulted only
            # after Precheck and never reaches the manager.
            for old in _ROI:
                if old['frame']!=token or old['camera']!=cid:continue
                sl=region_slices(old['event']['region'],cam.image.shape)
                roi.append(dict(scene=frame.scene_id,frame=token,camera=cid,event=old['event'],
                    exact_refresh_ratio=float(refresh[sl].mean()),tile_refresh_ratio=float(tile[sl].mean()),
                    exact_refresh_mask=encode_mask(refresh),GT_online=False))
        manager.commit(frame,outputs[token],result)
        assert not manager.get_current_certifications().certified_anchor_ids
    assert source_manifest()==json.loads((ROOT/'development_source_freeze.json').read_text())['files']
    write_json(folder/'complete.json',dict(pixel_pairs={k:dict(v) for k,v in counts.items()},roi=roi,cameras=cameras))
    print('pixel/mask audit',spec['scene_id'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    global _INPUTS,_ROI
    _INPUTS=infos,outputs,oracle
    _ROI=json.loads((BASELINE/'critical_roi_diagnosis.json').read_text())['events']
    pin=ROOT/'mask_audit_source_freeze.json'
    if not pin.exists():write_json(pin,dict(runtime_files=source_manifest(),audit_tool_sha256=sha(__file__),
        purpose='Exact mask intersections and mapping verification only, no method or threshold changes'))
    pending=[s for s in specs if not (ROOT/'mask_audit_traces'/s['scene_id']/'complete.json').exists()]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,pending))
    counts=defaultdict(Counter);roi=[];cameras=0
    for p in (ROOT/'mask_audit_traces').glob('*/complete.json'):
        r=json.loads(p.read_text());roi.extend(r['roi']);cameras+=r['cameras']
        for name,v in r['pixel_pairs'].items():counts[name].update(v)
    assert cameras==1440
    write_json(ROOT/'pixel_pair_and_mask_audit.json',dict(comparisons={k:dict(v) for k,v in counts.items()},
        roi=roi,cameras=cameras,tile_contains_exact_refresh_everywhere=True,
        scope='All actual pixels; pairwise acceptance is known, physical correctness of dense area remains unscorable'))


if __name__=='__main__':main()
