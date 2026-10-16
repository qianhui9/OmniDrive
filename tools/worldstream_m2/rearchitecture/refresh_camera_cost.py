"""Final numeric-policy-specific zero-update costs, after fresh full-model recheck."""
import copy
import json
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key
from worldstream.m2.profiler import measure
from .common import ROOT,write


@torch.no_grad()
def refresh(model):
    path=ROOT/'camera_costs.json';table=json.load(open(path))
    images=torch.randn(6,3,640,640,device='cuda')
    key=camera_cost_key(model,images);new=json.loads(key)
    if table.get('final_numeric_policy_explicit') and any(r['key']==key and r['changed_cameras']==0 and
            r.get('quality_certified') is True for r in table['records']):return
    compatible=[]
    for r in table['records']:
        old=json.loads(r['key'])
        if all(new.get(k)==v for k,v in old.items()):compatible.append(r)
    if not any(r['changed_cameras']==0 and r['quality_certified'] for r in compatible):
        raise AssertionError('Fresh model differs from prior independently task-verified conditions')
    context=dict(scene_id='final_cost_policy',frame_id='0',camera_ids=list(range(6)))
    runtime=IncrementalVisionRuntime(model,dict(mode='exact',reuse_boundary='camera'))
    runtime.forward(images,execution_context=context);cache=runtime.camera_executor.cache
    saved=copy.copy(vars(cache))
    def execute():
        cache.__dict__.update(saved)
        return runtime.forward(images,execution_context=dict(context,frame_id='1'))
    reuse=measure(execute,warmup=3,repeats=10)
    full=measure(lambda:model(images),warmup=3,repeats=10)
    write(ROOT/'pre_numeric_policy_camera_costs.json',table)
    table['records']=[r for r in table['records'] if r['key']!=key]
    table['records'].append(dict(key=key,changed_cameras=0,quality_certified=True,
        reuse_wall_ms=reuse['wall_ms'],full_wall_ms=full['wall_ms'],reuse_gpu_ms=reuse['gpu_ms'],full_gpu_ms=full['gpu_ms'],
        reuse=reuse,full=full,qualification='Existing exact full-model repeat evidence plus fresh 16-frame baseline native-memory recheck; remeasured with explicit actual Flash scale/inner norm policy before final balanced Auto tests'))
    table['final_numeric_policy_explicit']=True
    write(path,table)

