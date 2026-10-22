import copy
import numpy as np
import pytest
import torch
from worldstream.m1 import M1Config, WorldStateManager, CameraFrame, Frame, FullOutputs
from worldstream.m2 import IncrementalVisionRuntime


def frame_for(images, context):
    cams={cid:CameraFrame(cid,images[i].permute(1,2,0).numpy().copy(),
        np.array([[30.,0,20.],[0,30.,20.],[0,0,1.]]),np.eye(4),context['timestamp'],
        coordinate_space='preprocessed_image') for i,cid in enumerate(context['camera_ids'])}
    return Frame(context['scene_id'],context['frame_id'],context['timestamp'],np.eye(4),cams,
        execution_signature=context['execution_signature'])


@torch.no_grad()
def test_live_precheck_execution_commit_reuses_exact_and_refreshes_change(eva,context):
    m=WorldStateManager(M1Config(source_linked_state=True,temporal_visibility=True,local_support=True))
    rt=IncrementalVisionRuntime(eva,dict(cache_capture='always'))
    rt.costs.beneficial=lambda key,count:(True,None)
    x=torch.randint(0,256,(2,3,40,40),dtype=torch.uint8)
    diagnostics=[]
    for i in range(3):
        ctx=dict(context,frame_id=str(i),timestamp=100.+.1*i)
        if i==2:x=x.clone();x[0,:,0,0]=0
        f=frame_for(x,ctx);r=m.precheck(f);ctx['state_epoch']=r.diagnostics['state_epoch']
        out,d=rt.forward_from_m1(x.float()/255,m,ctx)
        assert torch.allclose(out[0],eva(x.float()/255)[0],atol=3e-6,rtol=3e-6)
        assert d['m1_snapshot_read'] and not d['world_identity_authorized_pixel_reuse']
        receipt=copy.deepcopy(d)
        m.commit(f,FullOutputs([],execution_receipt=receipt),r)
        diagnostics.append(d)
    assert diagnostics[1]['actual_skipped_windows']>0
    assert diagnostics[2]['recomputed_windows']>0
    assert diagnostics[2]['actual_skipped_windows']<diagnostics[1]['actual_skipped_windows']


@torch.no_grad()
@pytest.mark.parametrize('fault',['missing','stale','epoch','aborted','mutated'])
def test_snapshot_read_failure_full_fallback(eva,context,fault):
    m=WorldStateManager();rt=IncrementalVisionRuntime(eva,dict())
    x=torch.ones(2,3,40,40);ctx=dict(context,timestamp=100.)
    f=frame_for(torch.zeros_like(x,dtype=torch.uint8),ctx);r=m.precheck(f)
    ctx['state_epoch']=r.diagnostics['state_epoch']
    if fault=='missing':m=None
    if fault=='stale':ctx['frame_id']='future'
    if fault=='epoch':ctx['state_epoch']=-1
    if fault=='aborted':r.diagnostics['transaction_status']='aborted'
    if fault=='mutated':m.store.commit_count+=1
    out,d=rt.forward_from_m1(x,m,ctx)
    assert torch.equal(out[0],eva(x)[0])
    assert d['full_update_reason']=='missing_computation_state' and d['cache_hits']==0


@torch.no_grad()
def test_disabled_legacy_keeps_original_path(eva,context):
    rt=IncrementalVisionRuntime(eva,dict(enabled=False))
    x=torch.ones(2,3,40,40)
    out,d=rt.forward_from_m1(x,None,context)
    assert torch.equal(out[0],eva(x)[0]) and d['fallback_reason']=='disabled'
