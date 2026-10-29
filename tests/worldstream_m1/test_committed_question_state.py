from dataclasses import replace,FrozenInstanceError
import pytest
from worldstream.m1 import WorldStateManager,FullOutputs
from worldstream.m3.world_state import freeze_committed_state


def test_snapshot_is_current_commit_and_immutable(config,make_frame,make_prediction):
    manager=WorldStateManager(config)
    first=make_frame();result=manager.precheck(first)
    with pytest.raises(ValueError):freeze_committed_state(manager,first)
    manager.commit(first,FullOutputs([make_prediction()]),result)
    before=freeze_committed_state(manager,first)
    assert before.frame_id==first.frame_id and before.version==1
    assert before.anchors and any(a.observed for a in before.anchors)
    assert all(a.source_frame==first.frame_id for a in before.anchors if a.observed)
    second=make_frame(100.5);result=manager.precheck(second)
    with pytest.raises(ValueError):freeze_committed_state(manager,second)
    manager.commit(second,FullOutputs([make_prediction()]),result)
    after=freeze_committed_state(manager,second)
    assert after.version==2 and before.version==1 and before.frame_id!=after.frame_id
    with pytest.raises(FrozenInstanceError):before.frame_id='mutated'
    with pytest.raises(ValueError):freeze_committed_state(manager,first)


def test_camera_exposure_time_is_not_lidar_frame_time(config,make_frame,make_prediction):
    manager=WorldStateManager(config)
    frame=make_frame()
    frame.cameras['front'].timestamp=frame.timestamp-.015
    result=manager.precheck(frame)
    manager.commit(frame,FullOutputs([make_prediction()]),result)
    snapshot=freeze_committed_state(manager,frame)
    assert snapshot.anchors and snapshot.anchors[0].observed
    assert snapshot.anchors[0].cameras==('front',)


def test_current_detector_evidence_does_not_need_reusable_texture(config,make_frame,make_prediction):
    import numpy as np
    manager=WorldStateManager(config);frame=make_frame()
    frame.cameras['front'].image[:]=100
    result=manager.precheck(frame);manager.commit(frame,FullOutputs([make_prediction(semantic='car')]),result)
    assert not next(iter(manager.store.anchors.values())).observation_support
    assert not freeze_committed_state(manager,frame).anchors[0].observed
    current=freeze_committed_state(manager,frame,current_detection_support=True)
    assert current.anchors[0].observed and current.anchors[0].cameras==('front',)
    assert 'not_reuse_certificate' in current.anchors[0].observation_kind
    assert not next(iter(manager.store.anchors.values())).observation_support


def test_geometric_current_support_does_not_promote_history(config,make_frame,make_prediction):
    manager=WorldStateManager(config);frame=make_frame();frame.cameras['front'].image[:]=100
    result=manager.precheck(frame);manager.commit(frame,FullOutputs([make_prediction(semantic='car')]),result)
    frame2=make_frame(100.5);result=manager.precheck(frame2);manager.commit(frame2,FullOutputs([]),result)
    current=freeze_committed_state(manager,frame2,current_detection_support=True)
    assert current.anchors and not any(a.observed for a in current.anchors)


def test_current_support_requires_valid_visible_camera_and_actual_source(config,make_frame,make_prediction):
    manager=WorldStateManager(config);frame=make_frame();frame.cameras['front'].image[:]=100
    result=manager.precheck(frame);manager.commit(frame,FullOutputs([make_prediction(x=100.,semantic='car')]),result)
    assert not freeze_committed_state(manager,frame,current_detection_support=True).anchors[0].observed
    frame.cameras['front'].timing_reliable=False
    assert not freeze_committed_state(manager,frame,current_detection_support=True).anchors[0].observed


def test_competition_snapshot_cannot_create_positive_anchor(config,make_frame):
    manager=WorldStateManager(config);frame=make_frame();result=manager.precheck(frame)
    manager.commit(frame,FullOutputs([]),result)
    candidate=dict(candidate_id='decoded:0',semantic_class='pedestrian',center_ego=(0.,0.,10.),
        score=.2,source_frame=frame.frame_id,source_timestamp=frame.timestamp,
        evidence_origin='actual_current_decoded_perception')
    snapshot=freeze_committed_state(manager,frame,competition_candidates=[candidate])
    assert not snapshot.anchors and not manager.store.anchors
    assert snapshot.competition_evidence and not snapshot.competition_evidence[0].answer_witness_eligible


def test_competition_snapshot_rejects_stale_or_unprojected_candidates(config,make_frame):
    manager=WorldStateManager(config);frame=make_frame();result=manager.precheck(frame)
    manager.commit(frame,FullOutputs([]),result)
    candidate=dict(candidate_id='decoded:0',semantic_class='car',center_ego=(0.,0.,10.),score=.2,
        source_frame=frame.frame_id,source_timestamp=frame.timestamp,evidence_origin='actual_current_decoded_perception')
    for change in (dict(source_frame='future'),dict(source_timestamp=frame.timestamp+1),
                   dict(evidence_origin='oracle'),dict(score=.1),dict(center_ego=(0.,0.,-10.))):
        assert not freeze_committed_state(manager,frame,competition_candidates=[dict(candidate,**change)]).competition_evidence


def test_adapter_readout_preserves_M1_filter_but_exposes_competition(config,make_frame):
    import torch
    from worldstream.m1.omnidrive_adapter import outputs_from_model
    class Boxes:
        corners=torch.ones(1,8,3)
        gravity_center=torch.tensor([[0.,0.,10.]])
        dims=torch.ones(1,3)
        tensor=torch.zeros(1,7)
    sink=[];result=dict(boxes_3d=Boxes(),scores_3d=torch.tensor([.2]),labels_3d=torch.tensor([8]))
    outputs=outputs_from_model(make_frame(),[result],[],{},config,competition_sink=sink)
    assert not outputs.anchors and len(sink)==1 and sink[0]['semantic_class']=='pedestrian'
