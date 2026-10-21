from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs, ComputedFeature
from worldstream.m1.computation_state import validity_lease
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import validate_refresh_partition


@pytest.mark.parametrize("source,observed,now,compatible,conflict,reason", [
    (1.,1.,1.5,True,False,None), (None,1.,1.5,True,False,"source_missing"),
    (1.,1.,3.,True,False,"source_expired"), (2.,1.,1.5,True,False,"source_time_order"),
    (1.,1.,1.5,False,False,"source_incompatible"), (1.,1.,1.5,True,True,"observed_conflict"),
    (1.,1.,float("nan"),True,False,"source_missing")])
def test_lease(source,observed,now,compatible,conflict,reason):
    lease=validity_lease(source,observed,now,1.,compatible,conflict)
    assert lease.source_valid == (reason is None)
    assert lease.revoked_reason == reason


def test_committed_observation_without_computation(config,make_frame):
    manager=WorldStateManager(replace(config,source_linked_state=True,local_support=True,temporal_visibility=True))
    a=make_frame();r=manager.precheck(a)
    assert r.computation_state.diagnostics["observed_source_frame_id"] is None
    manager.commit(a,FullOutputs([]),r)
    b=make_frame(100.5);r=manager.precheck(b)
    bundle=r.computation_state
    assert bundle.diagnostics["observed_source_frame_id"] == a.frame_id
    assert all(not s["computation_available"] and s["computation_frame_id"] is None for s in bundle.source_records)
    link=bundle.observation_links["front"]
    assert len(link["source_points"]) > 0
    assert len(link["photometric_residual"]) == len(link["current_points"])
    c=bundle.refresh_contract["front"]
    assert decode_mask(c['exact_invalidation_mask']).all()
    assert decode_mask(c['reuse_candidate_mask']).any()
    assert validate_refresh_partition(*[decode_mask(c[k]) for k in ("dirty_hint_mask","reuse_candidate_mask","unknown_mask")])
    assert not bundle.diagnostics["transformer_dependency_certified"]
    assert r.physically_certified_anchor_ids == []
    assert bundle.source_records[0]["source_frame_id"] != b.frame_id
    manager.abort_pending()
    assert bundle.diagnostics['transaction_status']=='aborted'
    assert manager.store.last_timestamp == a.timestamp
    c=make_frame(101.);r=manager.precheck(c)
    assert r.full_refresh_hint
    assert not decode_mask(r.computation_state.refresh_contract["front"]["reuse_candidate_mask"]).any()


def test_deferred_validation_maintains_committed_history(config,make_frame):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True,dense_local_support=True)
    m=WorldStateManager(cfg)
    for i in range(4):
        frame=make_frame(100.+i*.5);result=m.precheck_with_observation(frame,validate_observation=False)
        assert not decode_mask(result.computation_state.refresh_contract["front"]["reuse_candidate_mask"]).any()
        m.commit(frame,FullOutputs([]),result)
        assert m.store.commit_count==i+1
        assert m._previous_frame_id==frame.frame_id


def test_source_metadata_is_snapshot(config,make_frame):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True)
    m=WorldStateManager(cfg);a=make_frame();r=m.precheck(a)
    f=ComputedFeature("f","local",(1,),"float32","cpu","layer",source_camera_ids=["front"])
    m.commit(a,FullOutputs([], [f]),r)
    b=make_frame(100.5);r=m.precheck(b)
    assert r.computation_state.source_records[0]["computation_frame_id"]==a.frame_id
    assert not r.computation_state.source_records[0]["computation_available"]
    c=make_frame(101.,scene="new");r=m.precheck(c)
    assert r.full_refresh_hint and all(s["computation_frame_id"] is None for s in r.computation_state.source_records)


def test_atomic_commit_keeps_computation_source(config,make_frame,make_prediction):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True)
    m=WorldStateManager(cfg)
    feature=ComputedFeature("backbone","backbone",(1,),"float32","cpu","eva",
        source_camera_ids=["front"],dependency_scope="full_camera")
    def receipt(frame,version,reused=False):
        return dict(cache_saved=True,cache_boundary='camera',cache_version=version,actual_reused_cameras=int(reused),
            cache_context=dict(frame_id=frame.frame_id,execution_signature=frame.execution_signature),
            output_camera_provenance=[dict(camera_id="front",computation_frame="100.0",
                computation_timestamp=100.,computation_version=1,
                verification_frame=frame.frame_id)])
    a=make_frame();r=m.precheck(a)
    m.commit(a,FullOutputs([make_prediction()],[feature],execution_receipt=receipt(a,1)),r)
    b=make_frame(100.5);r=m.precheck(b)
    assert all(x["computation_frame_id"]!=b.frame_id for x in r.computation_state.source_records)
    m.commit(b,FullOutputs([make_prediction()],[feature],execution_receipt=receipt(b,2,True)),r)
    assert m.store.features["backbone"].source_frame_id==a.frame_id
    c=make_frame(101.);r=m.precheck(c)
    source=next(s for s in r.computation_state.source_records if s["feature_id"]=="backbone")
    assert source["source_frame_id"]==b.frame_id
    assert source["computation_frame_id"]==a.frame_id
    assert source["source_cache_version"]==2 and source["computation_version"]==1
    assert source["computation_available"]
    m.abort_pending()
    assert m._computation_receipt["cache_version"]==2


def test_source_mode_keeps_shadow_ids(config,make_frame,make_prediction):
    base=replace(config,local_support=True,temporal_visibility=True,identity_continuity=True,
        identity_method="temporal_object",identity_tracking_policy="shadow")
    old,new=WorldStateManager(base),WorldStateManager(replace(base,source_linked_state=True))
    for i in range(4):
        for manager in (old,new):
            f=make_frame(100.+i*.5);r=manager.precheck(f)
            manager.commit(f,FullOutputs([make_prediction()]),r)
        assert list(old.store.anchors)==list(new.store.anchors)
        assert old.last_commit_diagnostics.get("identity_continuity")==new.last_commit_diagnostics.get("identity_continuity")
        assert old.get_current_certifications().certified_anchor_ids==new.get_current_certifications().certified_anchor_ids==()


def test_original_entry_source_mode_output_parity(tmp_path,config):
    import importlib.util
    from pathlib import Path
    import torch
    from tools.worldstream_m1.model_smoke import digest
    spec=importlib.util.spec_from_file_location("state_interface_entry",Path("tests/worldstream_m1/test_integration.py"))
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    base=helpers.make_model(tmp_path/"base",None)
    source=helpers.make_model(tmp_path/"source",WorldStateManager(replace(config,
        source_linked_state=True,temporal_visibility=True,local_support=True)))
    for i in range(4):
        metas,data=helpers.inputs(100.+i*.5)
        torch.manual_seed(3);expected=base.simple_test(metas,**data)
        metas,data=helpers.inputs(100.+i*.5)
        torch.manual_seed(3);actual=source.simple_test(metas,**data)
        assert digest(expected)==digest(actual)
        assert torch.equal(base.pts_bbox_head.memory_timestamp,source.pts_bbox_head.memory_timestamp)
    assert source.m1_manager.store.commit_count==4


def test_prefix_cache_does_not_fabricate_final_backbone(config,make_frame):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True)
    m=WorldStateManager(cfg);a=make_frame();r=m.precheck(a)
    f=ComputedFeature("backbone","backbone",(1,),"float32","cpu","eva",source_camera_ids=["front"])
    receipt=dict(cache_saved=True,cache_boundary="window",cache_version=1,
        cache_context=dict(frame_id=a.frame_id,execution_signature=a.execution_signature),
        prefix_window_geometry={0:[100,120]},
        output_window_provenance={0:[dict(camera_id="front",computation_frame_id=a.frame_id,
            computation_timestamp=a.timestamp,computation_version=1)]})
    m.commit(a,FullOutputs([], [f],execution_receipt=receipt),r)
    b=make_frame(100.5);r=m.precheck(b)
    records=r.computation_state.source_records
    assert not next(s for s in records if s["feature_id"]=="backbone")["computation_available"]
    prefix=next(s for s in records if s["feature_id"]=="eva_prefix:0:0")
    assert prefix["computation_available"]
    assert prefix["source_region"]==[(0,0,120,100)]
    assert prefix["computation_frame_id"]==a.frame_id


def test_camera_producer_version_stays_stable_after_recompute_and_reuse(config,make_frame):
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True,
        max_state_age_s=3.,full_refresh_interval=100)
    m=WorldStateManager(cfg)
    feature=ComputedFeature("backbone","backbone",(1,),"float32","cpu","eva",source_camera_ids=["front"])
    snapshots=[]
    for i in range(5):
        frame=make_frame(100.+i*.5)
        result=m.precheck(frame)
        record=next((s for s in result.computation_state.source_records if s["feature_id"]=="backbone"),None)
        if i in (3,4):snapshots.append(record)
        origin,producer_version,reused=(0,1,i==1) if i<2 else (2,3,i>2)
        receipt=dict(cache_saved=True,cache_boundary="camera",cache_version=i+1,
            actual_reused_cameras=int(reused),
            cache_context=dict(frame_id=frame.frame_id,execution_signature=frame.execution_signature),
            output_camera_provenance=[dict(camera_id="front",computation_frame=str(100.+origin*.5),
                computation_timestamp=100.+origin*.5,computation_version=producer_version,
                source_cache_version=producer_version,verification_frame=frame.frame_id)])
        m.commit(frame,FullOutputs([], [feature],execution_receipt=receipt),result)
    assert [s["computation_frame_id"] for s in snapshots]==["101.0","101.0"]
    assert [s["computation_version"] for s in snapshots]==[3,3]
    assert [s["computation_version_namespace"] for s in snapshots]==["m2_camera_execution"]*2
    assert [s["feature_record_version"] for s in snapshots]==[2,2]
    assert [s["source_cache_version"] for s in snapshots]==[3,4]
