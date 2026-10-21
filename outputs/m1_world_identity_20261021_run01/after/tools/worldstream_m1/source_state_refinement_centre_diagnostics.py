"""Offline nearby-reference diagnostics; never turns a nearby sample into truth."""
from collections import defaultdict
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from mmcv import Config
from scipy.spatial import cKDTree
from worldstream.m1 import M1Config
from .state_interface_replay import preprocessed_frame
from .evidence_experiments import load_inputs
from .coverage_oracle import world_lidar,prepare_pair
from .io_utils import write_json
from .state_interface_gpu_report import percentile


def main():
    root=Path('outputs/module1_source_state_refinement')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    centres=json.loads((root/'replay/centre_quality.json').read_text())
    groups=defaultdict(list)
    for i,s in enumerate(centres):groups[s['frame'],s['camera']].append((i,s))
    specs,infos,_,oracle=load_inputs('development')
    specs=json.loads((root/'replay/protocol.json').read_text())['scenes']
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    settings=M1Config.load('configs/worldstream_m1_source_state_refined.yaml')
    rows=[]
    for spec in specs:
        previous=previous_points=None
        for token in spec['samples'][:12]:
            frame,_=preprocessed_frame(infos[token],cfg,settings);points=world_lidar(infos[token])
            if previous is not None:
                refs=prepare_pair(previous,frame,previous_points,points,oracle[previous.frame_id],oracle[token],settings)
                for cid,ref in refs.items():
                    population=groups.get((token,cid),[])
                    if not population:continue
                    ids=np.flatnonzero(ref['scorable']&ref['annotated'])
                    if not len(ids):
                        rows.extend(dict(index=i,method=s['method'],nearest_reference_px=None,source_residual_px=None) for i,s in population)
                        continue
                    q=np.array([s['current'] for _,s in population]);p=np.array([s['source'] for _,s in population])
                    distance,index=cKDTree(ref['xy'][ids]).query(q)
                    j=ids[index]
                    predicted=p+ref['xy'][j]-q
                    residual=np.linalg.norm(predicted-ref['source_uv'][j],axis=1)
                    rows.extend(dict(index=i,method=s['method'],nearest_reference_px=float(d),source_residual_px=float(e))
                        for (i,s),d,e in zip(population,distance,residual))
            previous,previous_points=frame,points
    summary={}
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method]
        summary[method]=dict(sampled_centres=len(selected),within_pixels={str(radius):sum(
            r['nearest_reference_px'] is not None and r['nearest_reference_px']<=radius for r in selected) for radius in (.75,2,4)},
            nearby_4px_source_residual=percentile([r['source_residual_px'] for r in selected
                if r['nearest_reference_px'] is not None and r['nearest_reference_px']<=4]))
    write_json(root/'centre_nearby_diagnostics.json',dict(summary=summary,records=rows,
        interpretation='Nearby projected annotated depth-visible LiDAR only. Different surfaces may occur within a few pixels; this does not label candidate centres Correct/False or change the primary exact-pixel scoring population.',
        online_GT=False,physical_quality_of_unmatched_centres='Unscorable'))
    print(json.dumps(summary))


if __name__=='__main__':main()
