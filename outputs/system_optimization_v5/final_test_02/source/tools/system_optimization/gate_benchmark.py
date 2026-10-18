"""Paired original/current rejection gate on real preprocessed images.

This microbenchmark does not claim visual reuse or complete VLM acceleration.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from time import perf_counter
import numpy as np
import torch
from worldstream.m2.cheap_admission import CheapAdmission
from .audit import write, stats


@torch.no_grad()
def run(dest, cuda=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    old_source=subprocess.check_output(['git','show','HEAD:worldstream/m2/cheap_admission.py'])
    path=dest/'original_cheap_admission.py'
    if path.exists() and path.read_bytes()!=old_source:
        raise ValueError('Frozen reference differs from current HEAD')
    path.write_bytes(old_source)
    spec=importlib.util.spec_from_file_location('original_gate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The same actual raw cameras and existing deployment normalization. Never
    # call synthetic repeated pixels a real consecutive driving sample.
    import pickle,cv2
    with open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb') as handle:
        infos=pickle.load(handle)['infos']
    infos=sorted(infos,key=lambda x:(x['scene_token'],x['timestamp']))[:4]
    images=[]
    for info in infos:
        cameras=[]
        for camera in info['cams'].values():
            image=cv2.imread(camera['data_path'])
            if image is None:
                raise FileNotFoundError(camera['data_path'])
            image=cv2.resize(image,(640,640)).astype(np.float32)
            # Gate equality is invariant to a deterministic affine transform;
            # exact deployed pipeline correctness is checked in full VLM runs.
            cameras.append(torch.from_numpy(image).permute(2,0,1)/255.)
        images.append(torch.stack(cameras).to('cuda' if cuda else 'cpu'))
    ctx=dict(scene_id=infos[0]['scene_token'],camera_ids=list(infos[0]['cams']),execution_signature='same_raw_resize')
    samples={'original':[],'samples_only':[]};memory={};equal=[]
    for repeat in range(12):
        names=['original','samples_only'] if repeat%2 else ['samples_only','original']
        decisions={}
        for name in names:
            gate=module.CheapAdmission() if name=='original' else CheapAdmission()
            decisions[name]=[]
            for i,image in enumerate(images):
                if cuda:torch.cuda.synchronize()
                tick=perf_counter()
                result=gate.check(image,ctx,(256,256))
                if cuda:torch.cuda.synchronize()
                wall=(perf_counter()-tick)*1000
                decisions[name].append((result['reject'],result['reason']))
                if repeat>=2:samples[name].append(wall)
                memory[name]=result['snapshot_bytes']
        equal.append(decisions['original']==decisions['samples_only'])
    write(dest/('gate_gpu.json' if cuda else 'gate_cpu.json'),dict(
        device='cuda' if cuda else 'cpu',wall_ms={k:stats(v) for k,v in samples.items()},samples_ms=samples,
        snapshot_bytes=memory,decisions_equal=all(equal),unique_frames=len(infos),
        tokens=[i['token'] for i in infos],warmup=2,repeats=10,
        evidence='real_images_resized_gate_microbenchmark',
        caveat='Resize-only input, excludes full VLM and M1; shared GPU load may affect timing'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/system_optimization')
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    run(args.output,args.cuda)
