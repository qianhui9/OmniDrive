import math
import pytest
import torch
from worldstream.m2 import IncrementalVisionRuntime, M2Config
from worldstream.m2.anchor_risk import feature_statistics,risk_score,world_signal
from worldstream.m2.state_cache import CachePublicationError,NonFiniteCacheError
from .test_source_linked_admission import cold,bundle_for
from .test_wsgrr import region


def test_local_tail_detects_sparse_change_and_camera_max():
    a=torch.ones(2,1,10,4);b=a.clone();b[1,0,0]+=10
    s=feature_statistics(b,a)
    assert s['global_drift']==pytest.approx(math.sqrt(10))
    assert s['local_tail']==pytest.approx(10)
    assert s['hot_fraction']==pytest.approx(.1)
    assert feature_statistics(a,a)['global_drift']==0
    assert feature_statistics(b,a)==s


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_statistics_raise(value):
    a=torch.ones(1,2,2,3);b=a.clone();b[0,0,0,0]=value
    with pytest.raises(NonFiniteCacheError):feature_statistics(b,a)


@pytest.mark.parametrize('weights,scales,threshold', [((1,0,0,0),(1,1,1,1),None),((.2,.5,.2,.1),(.9,1.5,.8,1),.7)])
def test_score_is_one_normalized_nonnegative_sum(weights,scales,threshold):
    c=M2Config(global_risk_weights=weights,global_risk_scales=scales,global_risk_threshold=threshold)
    score,f,z=risk_score(dict(global_drift=.5,local_tail=.7,hot_fraction=.2),.3,1,c)
    assert f==pytest.approx((.5,.84 if weights[1] else 0.,.3,.5))
    assert score==pytest.approx(sum(w*v/s for w,v,s in zip(weights,f,scales)))
    assert z==pytest.approx(tuple(v/s for v,s in zip(f,scales)))


def test_zero_local_weight_omits_topk_hot_mask_and_stack():
    current=torch.randn(2,5,7,8);previous=torch.randn_like(current)
    with torch.autograd.profiler.profile() as profile:
        stats=feature_statistics(current,previous,include_local=False)
    names={event.name for event in profile.function_events}
    assert not names&{'aten::topk','aten::gt','aten::stack'}
    assert 'local_tail' not in stats and 'hot_fraction' not in stats
    complete=feature_statistics(current,previous)
    assert stats['global_drift']==complete['global_drift']
    assert stats['per_camera_global']==complete['per_camera_global']
    c=M2Config(global_risk_weights=(.7100524839296866,0,.33434531972575604,0),
        global_risk_scales=(.9874035567045212,1.9871003338653175,1,2/3))
    score=risk_score(stats,.6,1,c)[0]
    # Preserve the original score's normalize-then-weight evaluation order;
    # (w*v)/s can differ by one Python-float ULP from w*(v/s).
    legacy=sum(w*(v/s) for w,v,s in zip(c.global_risk_weights,
        (complete['global_drift'],complete['local_tail']*(1+complete['hot_fraction']),.6,1/2),c.global_risk_scales))
    assert score==legacy
    for threshold in (score,math.nextafter(score,math.inf),math.nextafter(score,-math.inf)):
        assert (score>=threshold)==(legacy>=threshold)


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_light_statistics_still_reject_nonfinite_inputs(value):
    a=torch.ones(2,2,2,4);b=a.clone();b[0,0,0,0]=value
    with pytest.raises(NonFiniteCacheError):feature_statistics(b,a,include_local=False)


@pytest.mark.parametrize('kw',[dict(global_risk_weights=(0,0,0,0)),dict(global_risk_scales=(1,0,1,1)),dict(global_risk_weights=(-1,0,0,0)),dict(global_risk_threshold=float('nan')),dict(global_tail_fraction=0)])
def test_invalid_risk_parameters(kw):
    with pytest.raises(ValueError):M2Config(**kw)


@torch.no_grad()
def test_actual_m1_mandatory_regions_change_anchor_decision_without_removing_windows(eva,context):
    c=M2Config(global_anchor_interval=3,global_risk_weights=(0,0,1,0),global_risk_threshold=.05)
    image=torch.randn(2,3,40,40)
    outcomes=[]
    for dirty in (False,True):
        rt=IncrementalVisionRuntime(eva,c);cold(rt,image,context)
        ctx=dict(context,frame_id='1',timestamp=1.1);b=bundle_for(rt,ctx)
        if dirty:
            b.region_states=[region(event_flags=['state_jump'])]
        _,d=rt.forward(image,execution_context=ctx,computation_state=b)
        outcomes.append(d)
        assert d['region_refresh']['event_missed_windows']==0
    assert outcomes[0]['global_suffix_reuse_frames']==1
    assert outcomes[1]['global_suffix_executions']==1
    assert outcomes[1]['global_world_signal']['mandatory_ratio']>outcomes[0]['global_world_signal']['mandatory_ratio']
    assert outcomes[1]['executed_windows']>=outcomes[0]['executed_windows']


@torch.no_grad()
def test_policy_transfer_owns_fresh_buffers_and_rejects_old_alias(eva,context):
    rt=IncrementalVisionRuntime(eva);image=torch.randn(2,3,40,40);cold(rt,image,context)
    cache=rt.cache;version=cache.version
    with pytest.raises(CachePublicationError,match='policy_storage_alias'):
        cache.commit(image,cache.outputs,context,cache.signature,block0_input=cache.block0_input,
            window_ages=cache.window_ages,descriptors=cache.descriptors,_transfer_policy=True)
    assert cache.version==version
