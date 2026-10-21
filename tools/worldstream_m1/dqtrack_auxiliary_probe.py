"""Capped, GT-free diagnostic geometry features; never used as acceptance gates."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import numpy as np
from pyquaternion import Quaternion
from worldstream.m1 import M1Config,Frame
from worldstream.m1.geometry import box_geometry
from worldstream.m1.dqtrack_bridge import align
from worldstream.m1.dqtrack_geometry import geometry_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--per-scene',type=int,default=25)
    a=p.parse_args()
    if not 1<=a.per_scene<=100:raise ValueError('Bounded diagnostic only')
    root=Path(a.root);protocol=json.loads((root/'protocol.json').read_text())
    with (root/'omnidrive/model_observations.pkl').open('rb') as f:candidates=pickle.load(f)
    rows=[]
    for spec in protocol['development']:
        file=root/'dqtrack'/spec['scene_id']/'normalized_tracking.json';data=json.loads(file.read_text())
        settings=replace(M1Config.load('configs/worldstream_m1_balanced.yaml'),dqtrack_results_path=str(file))
        used=0;previous={}
        for token in spec['samples']:
            mapping=data['m1_manifest']['frames'][token]
            frame=Frame(spec['scene_id'],token,mapping['timestamp'],None,{})  # metadata-only offline alignment
            predictions=candidates[token].anchors;hints,_=align(frame,predictions,settings)
            tracks={r['tracking_id']:r for r in data['results'][mapping['sample_token']]}
            for index,h in sorted(hints.items()):
                r=tracks[h['tracking_id']];matrix=np.eye(4)
                matrix[:3,:3]=Quaternion(r['rotation']).rotation_matrix;matrix[:3,3]=r['translation']
                geometry=box_geometry(np.zeros(3),np.asarray(r['size'])[[1,0,2]],to_world=matrix)
                features=geometry_features(geometry,predictions[index].geometry_world)
                prior=previous.get(h['tracking_id']);angle=None
                if prior is not None:
                    dot=abs(np.dot(np.array(prior['rotation']),np.array(r['rotation'])))
                    angle=float(2*np.arccos(np.clip(dot,-1,1)))
                rows.append(dict(scene_id=spec['scene_id'],frame_id=token,tracking_id=h['tracking_id'],
                    quaternion_step_angle_rad=angle,verified_velocity_residual=None,**features))
                used+=1
                if used>=a.per_scene:break
            previous=tracks
            if used>=a.per_scene:break
    with (root/'auxiliary_geometry_probe.json').open('x') as f:json.dump(dict(
        scope='first chronological alignments per scene, not a calibrated acceptance model',
        oriented_iou='true convex world-box intersection; unknown on numerical failure',
        velocity='not used: no independently verified velocity',rows=rows),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
