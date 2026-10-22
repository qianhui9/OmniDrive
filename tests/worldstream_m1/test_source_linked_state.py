from dataclasses import replace


def test_window_computation_source_does_not_advance_on_verification(config,make_frame):
    cfg=replace(config,source_linked_state=True,temporal_visibility=True,local_support=True,
                max_state_age_s=3.,full_refresh_interval=100)
    manager=WorldStateManager(cfg);snapshots=[]
    for i in range(5):
        frame=make_frame(100.+.5*i);result=manager.precheck(frame)
        snapshots.extend(r for r in result.computation_state.source_records
            if i in (3,4) and r['feature_id']=='eva_prefix:0:0')
        origin,version=(0,1) if i<2 else (2,3)
        h,w=frame.cameras['front'].image.shape[:2]
        receipt=dict(cache_saved=True,cache_boundary='window',cache_version=i+1,
            cache_context=dict(frame_id=frame.frame_id,execution_signature=frame.execution_signature),
            prefix_window_geometry={0:[h,w]},output_window_provenance={0:[dict(camera_id='front',
                computation_frame_id=str(100.+origin*.5),computation_timestamp=100.+origin*.5,
                computation_version=version,producer_layer=0)]})
        manager.commit(frame,FullOutputs([],execution_receipt=receipt),result)
    assert [r['computation_frame_id'] for r in snapshots]==['101.0','101.0']
    assert [r['computation_version'] for r in snapshots]==[3,3]
    assert [r['source_cache_version'] for r in snapshots]==[3,4]


def test_frozen_task_history_reads_verified_pending_snapshot_before_commit(config,make_frame,make_prediction):
    from worldstream.m3.world_state import freeze_verified_state,query_world_state
    from worldstream.m1.geometry import box_geometry
    cfg=replace(config,source_linked_state=True,temporal_visibility=True,local_support=True)
    manager=WorldStateManager(cfg);prediction=make_prediction()
    prediction.geometry_world=box_geometry([0,0,10],[6,6,2])
    manager.initialize(make_frame(),FullOutputs([prediction]))
    frame=make_frame(100.5);result=manager.precheck(frame)
    assert result.valid_anchor_ids
    frozen=freeze_verified_state(manager,frame)
    assert frozen.records and frozen.state_version==1
    assert frozen.records[0]['source_frame']=='100.0'
    manager.commit(frame,FullOutputs([prediction]),result)
    assert frozen.records[0]['source_frame']=='100.0' and manager.store.commit_count==2
    assert frozen.records[0]['physical_geometry_certified'] is False


def test_committed_source_ledger_and_scene_metrics_use_mainline_state(config,make_frame,make_prediction):
    import copy
    from tools.worldstream_m1.scene_metrics import validate_rows,summarize_scene
    manager=WorldStateManager(replace(config,source_linked_state=True,temporal_visibility=True,local_support=True))
    rows=[]
    for timestamp in (100.,100.5):
        frame=make_frame(timestamp);result=manager.precheck(frame)
        manager.commit(frame,FullOutputs([make_prediction()]),result)
        rows.append(dict(precheck=result.to_dict(),commit=copy.deepcopy(manager.last_commit_diagnostics),
                         committed_state=manager.export_state_audit()))
    assert validate_rows(rows)['status']=='PASS'
    metrics=summarize_scene(rows)
    assert metrics['counts']['frames']==2 and metrics['counts']['certified_anchor_events']==0
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
    cfg=replace(config,source_linked_state=True,local_support=True,temporal_visibility=True)
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






def test_original_entry_source_mode_output_parity(tmp_path,config):
    import importlib.util
    from pathlib import Path
    import torch
    from tools.worldstream_m123.benchmark.common import typed_digest as digest
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
