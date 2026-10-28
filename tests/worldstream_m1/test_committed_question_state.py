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
