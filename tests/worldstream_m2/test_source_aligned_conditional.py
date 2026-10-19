from dataclasses import replace
import numpy as np
import pytest
import torch
from worldstream.m1.state_types import ComputationStateBundle
from worldstream.m1.mask_codec import encode_mask
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.config import M2Config
from worldstream.m2.source_alignment import token_sources,source_windows


def link(shape,source='0',cid='front',displacement=None):
    h,w=shape;yy,xx=np.mgrid[:h,:w];q=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32);p=q.copy()
    if displacement is not None:p+=displacement.reshape(-1,2)
    return dict(source_frame_id=source,source_camera_id=cid,source_timestamp=1.,observation_timestamp=1.5,
        coordinate_space='preprocessed_image',source_points=p,current_points=q,
        validity_lease=dict(source_valid=True,max_observation_age=1.))


def state(runtime,context,shape,candidate=None,dirty=None,displacements=None):
    cache=runtime.conditional_executor.cache
    b=ComputationStateBundle(context['scene_id'],context['frame_id'],context['timestamp'],1,context['execution_signature'])
    b.diagnostics=dict(transaction_status='pending',cache_frame_id=cache.context['frame_id'],
        source_cache_version=cache.version,observed_source_frame_id=cache.context['frame_id'],cache_available=True,full_refresh_hint=False)
    for i,cid in enumerate(context['camera_ids']):
        c=np.ones(shape,bool) if candidate is None else candidate.copy()
        d=np.zeros(shape,bool) if dirty is None else dirty.copy();c&=~d;u=~(c|d)
        b.refresh_contract[cid]=dict(coordinate_space='preprocessed_image',dirty_hint_mask=encode_mask(d),
            reuse_candidate_mask=encode_mask(c),unknown_mask=encode_mask(u),full_refresh_hint=False)
        b.observation_links[cid]=link(shape,cache.context['frame_id'],cid,None if displacements is None else displacements[i])
        b.observation_links[cid]['source_timestamp']=cache.context['timestamp']
        b.observation_links[cid]['observation_timestamp']=context['timestamp']
    return b


def runtime(eva,**kw):
    return IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='window',state_interface='source_linked',
        experimental_conditional_reuse=True,cache_capture='always',**kw))


def test_partial_observed_patch_and_source_conflicts_never_expand():
    mask=np.zeros((16,16),bool);mask[1,1]=True;z=np.zeros_like(mask)
    m=token_sources(link(mask.shape),mask,z,~mask,4,'0','front','1')
    assert m.support_count.sum()==1 and not m.complete.any() and np.all(source_windows(m,2)==-1)
    l=link(mask.shape);l['source_points']=np.array([[1,1],[8,8]],np.float32);l['current_points']=np.array([[1,1],[1,1]],np.float32)
    m=token_sources(l,mask,z,~mask,4,'0','front','1')
    assert m.conflict[0,0] and not m.complete.any()


def test_window_mapping_full_footprint_position_and_padding():
    shape=(40,40);z=np.zeros(shape,bool);ones=~z;delta=np.zeros((*shape,2),np.float32)
    delta[:16,:16,0]=16;delta[:16,16:32,0]=-16
    m=token_sources(link(shape,displacement=delta),ones,z,z,4,'0','front','1')
    windows=source_windows(m,4)
    assert windows[0]==1 and windows[1]==0 and m.complete.all()
    m.complete[1,1]=False
    assert source_windows(m,4)[0]==-1


@pytest.mark.parametrize('reason',['frame','camera','unknown','dirty','out_of_bounds','many_to_one'])
def test_alignment_invalid_sources_rejected(reason):
    shape=(16,16);z=np.zeros(shape,bool);c=~z;d=z.copy();u=z.copy();l=link(shape)
    if reason=='frame':l['source_frame_id']='other'
    if reason=='camera':l['source_camera_id']='other'
    if reason=='unknown':u[:]=True;c[:]=False
    if reason=='dirty':d[:]=True;c[:]=False
    if reason=='out_of_bounds':l['source_points'][:]=100
    if reason=='many_to_one':l['source_points'][:]=1
    m=token_sources(l,c,d,u,4,'0','front','1')
    assert not m.complete.any() and np.all(source_windows(m,4)==-1)


@torch.no_grad()
def test_repeat_unknown_uses_actual_exact_check_and_real_producer_version(eva,context):
    r=runtime(eva);x=torch.randn(2,3,40,40);a=dict(context,timestamp=1.,state_epoch=1)
    r.forward(x,execution_context=a);b=dict(a,frame_id='1',timestamp=1.5)
    s=state(r,b,(40,40),candidate=np.zeros((40,40),bool));calls=[]
    hooks=[m.register_forward_hook(lambda m,a,v,i=i:calls.append(i)) for i,m in enumerate(eva.blocks)]
    actual,diag=r.forward(x,computation_state=s,execution_context=b)
    for h in hooks:h.remove()
    assert torch.equal(actual[0],eva(x)[0])
    assert diag['reuse_mode']=='exact' and diag['actual_skipped_windows']==36
    assert calls==[2,3] and r.cache.images is None  # isolated from formal Exact cache
    assert all(u['computation_frame_id']=='0' and u['computation_version']==1 for u in r.conditional_executor.cache.sources[1])
    assert r.conditional_executor.cache.version==2


@torch.no_grad()
def test_displaced_context_is_conditional_and_tail_always_executes(eva,context):
    r=runtime(eva,conditional_max_patch_error=1e-4);x=torch.randn(2,3,40,40)
    a=dict(context,timestamp=1.,state_epoch=1);r.forward(x,execution_context=a)
    current=x.clone();current[0,:,:16,:16]=x[0,:,:16,16:32];current[0,:,:16,16:32]=x[0,:,:16,:16]
    delta=np.zeros((2,40,40,2),np.float32);delta[0,:16,:16,0]=16;delta[0,:16,16:32,0]=-16
    b=dict(a,frame_id='1',timestamp=1.5);s=state(r,b,(40,40),displacements=delta)
    calls=[];hs=[m.register_forward_hook(lambda m,a,v,i=i:calls.append(i)) for i,m in enumerate(eva.blocks)]
    actual,diag=r.forward(current,computation_state=s,execution_context=b)
    for h in hs:h.remove()
    assert diag['reuse_mode']=='experimental_conditional' and diag['conditional_reused_windows']==2
    assert 0 not in calls and 1 not in calls and calls==[2,3]
    assert not diag['transformer_dependency_certified'] and not diag['task_quality_certified']
    assert not torch.equal(actual[0],eva(current)[0])  # ABS/context cannot be geometrically undone
    units=r.conditional_executor.cache.sources[1]
    assert units[0]['reuse_mode']=='experimental_conditional' and units[0]['computation_frame_id']=='0'
    assert units[0]['computation_window_index']==1 and units[1]['computation_window_index']==0
    # A later same-coordinate input is not re-labelled Exact after approximate ancestry.
    c=dict(b,frame_id='2',timestamp=2.);s2=state(r,c,(40,40))
    _,d2=r.forward(current,computation_state=s2,execution_context=c)
    assert d2['conditional_reused_windows']>=2 and d2['reuse_mode']=='experimental_conditional'
    assert r.conditional_executor.cache.sources[1][0]['computation_window_index']==1


@torch.no_grad()
def test_partial_source_window_and_changed_unknown_fall_back_full(eva,context):
    r=runtime(eva);x=torch.randn(2,3,40,40);a=dict(context,timestamp=1.,state_epoch=1);r.forward(x,execution_context=a)
    c=np.zeros((40,40),bool);c[2,2]=True;b=dict(a,frame_id='1',timestamp=1.5)
    s=state(r,b,(40,40),candidate=c);actual,d=r.forward(x+.1,computation_state=s,execution_context=b)
    assert torch.equal(actual[0],eva(x+.1)[0]) and d['actual_skipped_windows']==0
    assert d['fallback_reason']=='no_complete_source_window'


@torch.no_grad()
def test_conditional_auto_is_closed_and_full_refresh_scene_change_safe(eva,context):
    cfg=dict(mode='auto',reuse_boundary='window',state_interface='source_linked',experimental_conditional_reuse=True)
    r=IncrementalVisionRuntime(eva,cfg);x=torch.randn(2,3,40,40);a=dict(context,timestamp=1.)
    actual,d=r.forward(x,execution_context=a)
    assert d['fallback_reason']=='conditional_auto_unqualified' and d['reuse_mode']=='full'
    assert torch.equal(actual[0],eva(x)[0])
    r=runtime(eva);r.forward(x,execution_context=a);b=dict(a,frame_id='1',timestamp=1.5)
    s=state(r,b,(40,40));s.diagnostics['full_refresh_hint']=True
    _,d=r.forward(x,computation_state=s,execution_context=b)
    assert d['fallback_reason']=='m1_full_refresh' and d['actual_skipped_windows']==0
    _,d=r.forward(x,execution_context=dict(b,scene_id='other',frame_id='2',timestamp=2.))
    assert d['fallback_reason']=='scene_change' and d['actual_skipped_windows']==0


@torch.no_grad()
def test_conditional_publication_oom_keeps_old_snapshot(eva,context,monkeypatch):
    r=runtime(eva);x=torch.randn(2,3,40,40);a=dict(context,timestamp=1.);r.forward(x,execution_context=a)
    c=r.conditional_executor;old_content=c.content;old_version=c.cache.version;old_images=c.cache.images
    def fail(*args,**kw):raise torch.cuda.OutOfMemoryError('injected')
    monkeypatch.setattr(c.cache,'commit',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):c._publish(x,old_content,c.cache.outputs,a,r.signature(a),c.cache.output_stride)
    assert c.content is old_content and c.cache.images is old_images and c.cache.version==old_version


def test_conditional_disabled_by_default_and_configuration_guarded():
    assert not M2Config().experimental_conditional_reuse
    with pytest.raises(ValueError):M2Config(experimental_conditional_reuse=True,reuse_boundary='camera')
    with pytest.raises(ValueError):M2Config(conditional_max_patch_error=float('nan'))


@torch.no_grad()
def test_conditional_cache_capture_never_is_respected(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='window',state_interface='source_linked',
        experimental_conditional_reuse=True,cache_capture='never'))
    x=torch.randn(2,3,40,40)
    for ordinal in range(2):
        value,diag=r.forward(x,execution_context=dict(context,frame_id=str(ordinal),timestamp=1.+ordinal*.5))
        assert torch.equal(value[0],eva(x)[0])
        assert not diag['cache_saved'] and diag['actual_skipped_windows']==0
        assert r.conditional_executor.content is None and r.conditional_executor.cache.images is None
