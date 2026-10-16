import json
import torch
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.camera_executor import camera_cost_key


@torch.no_grad()
def test_auto_only_approved_key_count_can_reuse(eva,context,tmp_path):
    x=torch.randn(2,3,40,40);key=camera_cost_key(eva,x)
    table=tmp_path/'camera.json'
    table.write_text(json.dumps(dict(schema_version=1,records=[dict(key=key,changed_cameras=0,
        quality_certified=True,reuse_wall_ms=.1,full_wall_ms=1,reuse_gpu_ms=.1,full_gpu_ms=1)])))
    r=IncrementalVisionRuntime(eva,dict(mode='auto',reuse_boundary='camera',camera_cost_table=str(table)))
    r.forward(x,execution_context=context)
    _,d=r.forward(x,execution_context=dict(context,frame_id='1',timestamp=1.))
    assert d['actual_reused_cameras']==2 and d['task_quality_certified']
    _,d=r.forward(x,execution_context=dict(context,frame_id='2',timestamp=.5))
    assert d['fallback_reason']=='camera_timestamp_not_increasing'
    x=x.clone();x[0,:,1,1]+=.01
    _,d=r.forward(x,execution_context=dict(context,frame_id='3',timestamp=2.))
    assert d['actual_reused_cameras']==0 and d['fallback_reason']=='unprofiled_camera_cost'


@torch.no_grad()
def test_camera_caller_mutation_does_not_corrupt_history(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    x=torch.randn(2,3,40,40);out,_=r.forward(x,execution_context=context)
    expected=r.camera_executor.cache.features.clone();out[0].zero_()
    assert torch.equal(expected,r.camera_executor.cache.features)


@torch.no_grad()
def test_camera_never_capture_policy(eva,context):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera',cache_capture='never'))
    image=torch.zeros(2,3,40,40)
    for i in range(2):
        result,diagnostic=runtime.forward(image,execution_context=dict(context,frame_id=str(i)))
        assert diagnostic['fallback_reason']=='capture_disabled'
        assert not diagnostic['cache_saved'] and not runtime.camera_executor.cache.bytes
        assert torch.equal(result[0],eva(image)[0])


@torch.no_grad()
def test_invalid_camera_timestamp_cannot_publish_source(eva,context):
    r=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    x=torch.zeros(2,3,40,40)
    _,d=r.forward(x,execution_context=dict(context,timestamp=float('nan')))
    assert d['fallback_reason']=='invalid_camera_timestamp' and not r.camera_executor.cache.bytes
    _,d=r.forward(x,execution_context=dict(context,frame_id='1',timestamp=1.))
    assert d['fallback_reason']=='no_valid_cache' and d['cache_saved']


@torch.no_grad()
def test_camera_non_boolean_quality_flag_fails_closed(eva,context,tmp_path):
    x=torch.zeros(2,3,40,40)
    path=tmp_path/'malformed_quality.json'
    path.write_text(json.dumps(dict(schema_version=1,records=[dict(key=camera_cost_key(eva,x),
        changed_cameras=0,quality_certified='false',reuse_wall_ms=.1,full_wall_ms=1.,reuse_gpu_ms=.1,full_gpu_ms=1.)])))
    r=IncrementalVisionRuntime(eva,dict(mode='auto',reuse_boundary='camera',camera_cost_table=str(path)))
    r.forward(x,execution_context=context)
    _,d=r.forward(x,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='camera_downstream_quality_unverified' and d['actual_reused_cameras']==0


@torch.no_grad()
def test_camera_auto_accounts_for_validation_budget(eva,context,tmp_path):
    x=torch.zeros(2,3,40,40);path=tmp_path/'cost.json'
    path.write_text(json.dumps(dict(schema_version=1,records=[dict(key=camera_cost_key(eva,x),
        changed_cameras=0,quality_certified=True,reuse_wall_ms=.1,full_wall_ms=1.,reuse_gpu_ms=.1,full_gpu_ms=1.)])))
    r=IncrementalVisionRuntime(eva,dict(mode='auto',reuse_boundary='camera',camera_cost_table=str(path)))
    r.forward(x,execution_context=context)
    _,d=r.forward(x,execution_context=dict(context,frame_id='1',necessary_validation_ms=2.))
    assert d['fallback_reason']=='camera_validation_cost_not_beneficial' and d['actual_reused_cameras']==0


@torch.no_grad()
def test_inner_norm_policy_invalidates_camera_cache_and_cost_key(eva,context):
    image=torch.randn(2,3,40,40)
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    runtime.forward(image,execution_context=context)
    old_key=camera_cost_key(eva,image)
    eva.blocks[0].mlp.ffn_ln.eps*=2
    assert camera_cost_key(eva,image)!=old_key
    _,d=runtime.forward(image,execution_context=dict(context,frame_id='1'))
    assert d['fallback_reason']=='execution_signature_mismatch' and d['actual_reused_cameras']==0


@torch.no_grad()
def test_camera_capture_oom_keeps_completed_output_without_second_forward(eva,context,monkeypatch):
    runtime=IncrementalVisionRuntime(eva,dict(mode='exact',reuse_boundary='camera'))
    def fail(*args,**kw):raise torch.cuda.OutOfMemoryError('simulated cache publish allocation')
    monkeypatch.setattr(runtime.camera_executor.cache,'publish',fail)
    calls=[];hook=eva.register_forward_hook(lambda *args:calls.append(1))
    image=torch.randn(2,3,40,40)
    result,d=runtime.forward(image,execution_context=context)
    hook.remove()
    assert len(calls)==1 and not d['cache_saved'] and runtime.camera_executor.cache.bytes==0
    assert torch.equal(result[0],eva(image)[0])
