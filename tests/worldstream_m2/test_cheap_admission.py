import torch
from worldstream.m2.cheap_admission import CheapAdmission


def test_gate_never_accepts_unsampled_changes():
    gate=CheapAdmission();ctx=dict(scene_id='s',camera_ids=['a','b'])
    base=torch.zeros(2,3,40,40)
    assert gate.check(base,ctx,(16,16))['reject']
    changed=base.clone();changed[:,:,1,1]=1
    result=gate.check(changed,ctx,(16,16))
    assert not result['reject'] and result['sample_acceptance'] is False
    changed+=1
    assert gate.check(changed,ctx,(16,16))['reject']


def test_gate_owns_snapshot_and_resets_scene():
    gate=CheapAdmission();ctx=dict(scene_id='s',camera_ids=['a'])
    image=torch.zeros(1,3,33,37);gate.check(image,ctx,(16,16));image+=1
    assert gate.check(image,ctx,(16,16))['reject']
    assert gate.check(image,dict(ctx,scene_id='other'),(16,16))['reason']=='gate_no_compatible_history'


def test_nonfinite_never_creates_sample_acceptance():
    gate=CheapAdmission();ctx=dict(scene_id='s',camera_ids=['a'])
    x=torch.full((1,3,16,16),float('nan'))
    gate.check(x,ctx,(16,16))
    assert gate.check(x,ctx,(16,16))['reject']


def test_sample_snapshot_matches_full_snapshot_rejections():
    gate=CheapAdmission();ctx=dict(scene_id='s',camera_ids=['a','b'])
    generator=torch.Generator().manual_seed(71)
    previous=None
    for step in range(30):
        image=torch.randn(2,3,33,37,generator=generator)
        if step%3 and previous is not None:
            image=previous.clone()
            image[:,:,step%33,step%37]+=1
        result=gate.check(image,ctx,(16,16))
        if previous is not None:
            current=image[:,:,gate.rows,gate.cols]
            old=previous[:,:,gate.rows,gate.cols]
            reference=bool(((current!=old)|~torch.isfinite(current)).any(1).any(-1).all())
            assert result['reject']==reference
        previous=image.clone()
        image.fill_(float('nan'))  # caller mutation must not alter sample history
        assert gate.previous.storage().data_ptr()!=image.storage().data_ptr()
    assert result['snapshot_bytes']<previous.numel()*previous.element_size()


@torch.no_grad()
def test_gate_rejection_reaches_runtime_before_planner(eva,context,monkeypatch):
    from worldstream.m2 import IncrementalVisionRuntime
    runtime=IncrementalVisionRuntime(eva,dict(mode='auto',cheap_gate=True,m1_service='output_only'))
    def forbidden(*a,**kw):raise AssertionError('Rejected input must not plan sparse execution')
    monkeypatch.setattr(runtime.planner,'plan',forbidden)
    x=torch.randn(2,3,40,40)
    runtime.pre_m1_admission(x,context)
    y,d=runtime.forward(x,execution_context=context)
    assert d['fallback_reason']=='cheap_input_rejection' and not d['cache_saved']
    assert torch.equal(y[0],eva(x)[0])
    c=dict(context,frame_id='1')
    x=x+1;runtime.pre_m1_admission(x,c)
    _,d=runtime.forward(x,execution_context=c)
    assert d['fallback_reason']=='cheap_input_rejection'
    assert runtime.cheap_admission.version==2


@torch.no_grad()
def test_zero_budget_does_not_allocate_gate_history(eva,context):
    from worldstream.m2 import IncrementalVisionRuntime
    r=IncrementalVisionRuntime(eva,dict(cheap_gate=True,memory_budget_mb=0))
    d=r.pre_m1_admission(torch.zeros(2,3,40,40),context)
    assert d['reason']=='gate_memory_budget' and d['reject']
    assert r.cheap_admission.previous is None
