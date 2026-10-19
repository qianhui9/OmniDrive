import copy
import numpy as np
import pytest
import torch
from worldstream.m1.state_types import ComputationStateBundle,M1Result
from worldstream.m1.mask_codec import encode_mask
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.m1_adapter import computation_admission
from worldstream.m2.state_cache import StateCache


def bundle_for(runtime,context,unknown=True):
    old=runtime.cache.context
    b=ComputationStateBundle(context["scene_id"],context["frame_id"],context["timestamp"],1,context["execution_signature"])
    b.diagnostics=dict(cache_available=True,source_cache_version=runtime.cache.version,
        cache_frame_id=old["frame_id"],observed_source_frame_id=old["frame_id"],full_refresh_hint=False)
    for cid in context["camera_ids"]:
        zero=np.zeros((40,40),bool);one=~zero
        b.refresh_contract[cid]=dict(coordinate_space="preprocessed_image",full_refresh_hint=False,
            dirty_hint_mask=encode_mask(zero),reuse_candidate_mask=encode_mask(zero if unknown else one),
            unknown_mask=encode_mask(one if unknown else zero))
        b.observation_links[cid]=dict(source_frame_id=old["frame_id"],source_camera_id=cid,
            source_timestamp=old["timestamp"],validity_lease=dict(source_valid=True,max_observation_age=2.))
    return b


@torch.no_grad()
@pytest.mark.parametrize("unknown",[True,False])
@pytest.mark.parametrize("changed",[True,False])
def test_candidates_never_bypass_exact(eva,context,unknown,changed,record_property):
    import json
    observations=[]
    rt=IncrementalVisionRuntime(eva,dict(mode="exact",state_interface="source_linked",cache_capture="always"))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    # Cold cache: no computed source yet, full real execution.
    b=ComputationStateBundle("scene","0",1.,1,"test",diagnostics=dict(full_refresh_hint=True))
    r=M1Result("scene","0",1.,computation_state=b)
    rt.forward(x,r,execution_context=ctx)
    for i in range(1,4):
        ctx=dict(ctx,frame_id=str(i),timestamp=1.+i*.1)
        b=bundle_for(rt,ctx,unknown)
        r=M1Result("scene",str(i),ctx["timestamp"],computation_state=b)
        if changed:
            x=x.clone();x[0,:,0,0]+=.1
        y,d=(rt.forward(x,execution_context=ctx,computation_state=b) if i==2 else
            rt.forward(x,r,execution_context=ctx))
        assert torch.allclose(y[0],eva(x)[0],atol=3e-6,rtol=3e-6)
        assert d["actual_skipped_windows"] == (34 if changed else 36)
        assert rt.cache.sources[0][-1]["computation_frame_id"] == "0"
        assert rt.cache.sources[0][0]["computation_frame_id"] == (str(i) if changed else "0")
        assert d['source_age'] == pytest.approx(i*.1)
        assert d['cache_verification_age'] == pytest.approx(.1)
        assert all(s.get("mode")=="full" for s in d["stages"][2:])
        observations.append(dict(frame=i,unknown=unknown,changed=changed,
            skipped=d['actual_skipped_windows'],verified=d['verified_exact_windows'],
            max_abs=float((y[0]-eva(x)[0]).abs().max()),
            last_window_source=rt.cache.sources[0][-1]['computation_frame_id']))
    record_property('engineering_observations',json.dumps(observations))


@torch.no_grad()
@pytest.mark.parametrize("fault,reason",[("version","m1_cache_source_mismatch"),
    ("epoch","m1_state_epoch_mismatch"),("lease","m1_invalid_lease"),
    ("coordinate","preprocessing_geometry_unavailable"),("partition","invalid_refresh_contract"),
    ("full","m1_full_refresh"),("time","m1_invalid_lease")])
def test_fail_closed(eva,context,fault,reason):
    rt=IncrementalVisionRuntime(eva,dict(mode="exact",cache_capture="always"))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.,state_epoch=1)
    rt.forward(x,execution_context=ctx)
    ctx=dict(ctx,frame_id="1",timestamp=1.5);b=bundle_for(rt,ctx)
    if fault=="version":b.diagnostics["source_cache_version"]=-1
    if fault=="epoch":b.state_epoch=2
    if fault=="lease":b.observation_links["front"]["validity_lease"]["source_valid"]=False
    if fault=="coordinate":b.refresh_contract["front"]["coordinate_space"]="raw_image"
    if fault=="partition":b.refresh_contract["front"]["reuse_candidate_mask"]=encode_mask(np.ones((40,40),bool))
    if fault=="full":b.diagnostics["full_refresh_hint"]=True
    if fault=="time":b.observation_links["front"]["source_timestamp"]=2.
    _,actual=computation_admission(b,ctx,x,rt.cache)
    assert actual==reason


def test_cache_publication_oom_is_atomic(monkeypatch):
    c=StateCache(10000);x=torch.ones(1,3,4,4)
    c.commit(x,{0:torch.ones(1,2)},dict(frame_id="a",camera_ids=["front"]),"sig")
    old=(c.images,c.context,c.sources,c.version)
    def oom(*a,**kw):raise torch.cuda.OutOfMemoryError("injected allocation failure")
    monkeypatch.setattr(torch.Tensor,"clone",oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        c.commit(x,{},dict(frame_id="b",camera_ids=["front"]),"sig")
    assert c.images is old[0] and c.context is old[1] and c.sources is old[2] and c.version==old[3]


def test_cache_metadata_oom_is_atomic():
    class FailingContext(dict):
        def __getitem__(self,key):
            if key=='camera_ids':raise torch.cuda.OutOfMemoryError('metadata allocation failure')
            return super().__getitem__(key)
    c=StateCache(10000);x=torch.ones(1,3,4,4)
    c.commit(x,{0:torch.ones(1,2)},dict(frame_id='a',camera_ids=['front']),'sig')
    old=(c.images,c.outputs,c.context,c.sources,c.version)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        c.commit(x*2,{0:torch.ones(1,2)*2},FailingContext(frame_id='b',camera_ids=['front']),'sig')
    assert c.images is old[0] and c.outputs is old[1] and c.context is old[2] and c.sources is old[3] and c.version==old[4]


@torch.no_grad()
@pytest.mark.parametrize("gate",["dirty","full","aborted","expired"])
def test_refresh_gates_override_identical_input(eva,context,gate):
    from dataclasses import replace
    rt=IncrementalVisionRuntime(eva,dict(mode="exact",cache_capture="always"))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    rt.forward(x,execution_context=ctx)
    rt.config=replace(rt.config,state_interface="source_linked")
    ctx=dict(ctx,frame_id="1",timestamp=1.5);b=bundle_for(rt,ctx)
    r=M1Result("scene","1",1.5,computation_state=b)
    if gate=="dirty":
        dirty=np.zeros((40,40),bool);dirty[0,0]=True
        b.refresh_contract["front"].update(dirty_hint_mask=encode_mask(dirty),unknown_mask=encode_mask(~dirty))
    if gate=="full":r.full_refresh_hint=True
    if gate=="aborted":r.diagnostics["transaction_status"]="aborted"
    if gate=="expired":
        for sources in rt.cache.sources.values():
            for source in sources:source["computation_timestamp"]=-10.
    out,d=rt.forward(x,r,execution_context=ctx)
    assert torch.allclose(out[0],eva(x)[0],atol=3e-6,rtol=3e-6)
    assert d["actual_skipped_windows"]==(34 if gate=="dirty" else 0)


@torch.no_grad()
def test_camera_bundle_force_refresh(eva,context):
    from types import SimpleNamespace
    rt=IncrementalVisionRuntime(eva,dict(mode="exact",reuse_boundary="camera",state_interface="source_linked"))
    x=torch.randn(2,3,40,40);ctx=dict(context,timestamp=1.)
    b=ComputationStateBundle("scene","0",1.,1,"test",diagnostics=dict(full_refresh_hint=True))
    r=M1Result("scene","0",1.,computation_state=b)
    rt.forward(x,r,execution_context=ctx)
    for i in (1,2):
        ctx=dict(ctx,frame_id=str(i),timestamp=1.+i*.1)
        b=bundle_for(SimpleNamespace(cache=rt.camera_executor.cache),ctx)
        r=M1Result("scene",str(i),ctx["timestamp"],full_refresh_hint=i==2,computation_state=b)
        out,d=rt.forward(x,r,execution_context=ctx)
        assert torch.equal(out[0],eva(x)[0])
        assert d["actual_reused_cameras"]==(2 if i==1 else 0)
