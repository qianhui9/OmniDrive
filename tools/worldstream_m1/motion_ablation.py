"""Freeze metadata-only M1.2 targets, then independently replay four variants."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from .dataset import log_context_from_root
from .scene_protocol import select_scenes
from .effective_acceptance import replay,sha,source_manifest
from worldstream.m1.omnidrive_adapter import quaternion_transform

VARIANTS=('baseline','a','b','c')


def freeze(output,frames=12,scenes=6):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    predictions=Path('outputs/module1_evidence_optimization/model_observations.pkl')
    spec=json.loads(Path('outputs/module1_evidence_optimization/data_selection.json').read_text())
    with predictions.open('rb') as f:outputs=pickle.load(f)
    with open(spec['info_file'],'rb') as f:infos=pickle.load(f)['infos']
    context=log_context_from_root('data/nuscenes')
    sequences,selection=select_scenes(infos,set(outputs),0,frames,context)
    features=[]
    for scene,rows in sequences.items():
        poses=[quaternion_transform(r['ego2global_rotation'],r['ego2global_translation']) for r in rows]
        translation=sum(np.linalg.norm(b[:3,3]-a[:3,3]) for a,b in zip(poses,poses[1:]))
        angles=[float(np.degrees(np.arccos(np.clip((np.trace(b[:3,:3]@a[:3,:3].T)-1)/2,-1,1)))) for a,b in zip(poses,poses[1:])]
        features.append(dict(scene_id=scene,log_token=context.log_for_scene(scene),frames=len(rows),
            ego_translation_m=float(translation),ego_rotation_deg=sum(angles),max_step_rotation_deg=max(angles,default=0),
            model_object_detections=sum(a.anchor_type=='object' for r in rows for a in outputs[r['token']].anchors)))
    chosen=[]
    # Fixed metadata extrema, independent of compensation results and labels.
    for key,reverse in [('ego_translation_m',False),('ego_translation_m',True),('ego_rotation_deg',True),
                         ('max_step_rotation_deg',True),('model_object_detections',True),('model_object_detections',False)]:
        for row in sorted(features,key=lambda x:((-x[key] if reverse else x[key]),x['scene_id'])):
            if row['scene_id'] not in chosen:
                chosen.append(row['scene_id']);break
        if len(chosen)>=scenes:break
    chosen=chosen[:scenes]
    for s in selection['scenes']:
        s.update(scheduled=s['scene_id'] in chosen,status='Scheduled' if s['scene_id'] in chosen else 'Not Run')
    selection.update(scheduled_scene_count=len(chosen),sampling_rule='Frozen metadata extrema; per-scene chronological keyframe prefixes',
        unexecuted_scene_ids=[s['scene_id'] for s in selection['scenes'] if s['scene_id'] not in chosen])
    configs={v:dict(path=f'configs/worldstream_m12_{v}.yaml',sha256=sha(f'configs/worldstream_m12_{v}.yaml')) for v in VARIANTS}
    value=dict(version='M1.2',frozen_before_ablation=True,online_GT=False,
        sampling_rule='Distinct deterministic metadata extrema: min/max translation, max total/step rotation, max/min actual model object counts; no outcome or label selection',
        selected_scene_ids=chosen,target_frames_per_scene=frames,metadata_features=features,
        source_info=spec['info_file'],source_info_sha256=sha(spec['info_file']),detection_sha256=sha(predictions),
        configs=configs,source_sha256=source_manifest(),selection=selection,
        unexecuted_scene_ids=[s['scene_id'] for s in selection['scenes'] if s['scene_id'] not in chosen],
        GPU_target_scene_ids=['01452fbfbf4543af8acdfd3e8a1ee806','01e4fcbe6e49483293ce45727152b36e'],
        GPU_selection_reason='Previously exposed M1.1 positive/zero-coverage validation controls; not an independent qualification population',
        qualification_population=False,m2_enabled=False,m3_enabled=False)
    (root/'frozen_protocol.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def run(root,repeats=2):
    root=Path(root);spec=json.loads((root/'frozen_protocol.json').read_text())
    for variant in VARIANTS:
        config=spec['configs'][variant]
        if sha(config['path'])!=config['sha256']:raise ValueError('Frozen ablation configuration changed')
        replay(root/variant,spec['target_frames_per_scene'],0,config['path'],repeats,spec['selected_scene_ids'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run'])
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--scenes',type=int,default=6);p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.output,a.frames,a.scenes)
    else:run(a.output,a.repeats)


if __name__=='__main__':main()
