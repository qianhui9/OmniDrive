from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1.state_types import WorldAnchor,Geometry,Frame
from worldstream.m1.task_facts import extract_task_facts


def inputs():
    anchor=WorldAnchor('a','scene','object','car',Geometry(np.array([[1.,-1.,0.],[3.,1.,2.]]),np.array([2.,0.,1.])),
        100.,source_score=1.,geometry_uncertainty=0.,history_observations=100,
        source_frame_id='source',source_execution_signature='sig',evidence_origin='actual_perception')
    frame=Frame('scene','current',100.5,np.eye(4),{},'sig')
    check=dict(valid=True,cameras={'CAM_FRONT':dict(valid=True)})
    return anchor,frame,check


def test_high_score_zero_declared_uncertainty_and_observation_do_not_certify_physics():
    facts=extract_task_facts(*inputs())
    assert {f.fact_type for f in facts}=={'object_persistence','relative_direction','metric_distance','motion_relation'}
    assert not any(f.certified for f in facts)
    assert all(f.verified_timestamp is None for f in facts)
    assert all(f.uncertainty['independent_localization_error_bound_m'] is None for f in facts)
    distance=next(f for f in facts if f.fact_type=='metric_distance')
    assert distance.fact_value==pytest.approx(np.sqrt(5))
    assert distance.rejection_reason=='independent_metric_error_bound_missing'
    direction=next(f for f in facts if f.fact_type=='relative_direction')
    assert direction.fact_value['lateral']==0 and not direction.certified


@pytest.mark.parametrize('updates',[dict(scene_id='other'),dict(source_timestamp=100.5),
    dict(source_timestamp=99.),dict(evidence_origin='oracle'),dict(source_execution_signature='old')])
def test_stale_and_oracle_provenance_stay_unknown(updates):
    anchor,frame,check=inputs()
    assert all(f.rejection_reason=='stale_unsupported_provenance_or_geometry'
        for f in extract_task_facts(replace(anchor,**updates),frame,check))


def test_untrusted_motion_not_translated_into_fact():
    anchor,frame,check=inputs()
    anchor.velocity_world=np.array([3.,0.,0.]);anchor.velocity_reliable=False
    motion=next(f for f in extract_task_facts(anchor,frame,check) if f.fact_type=='motion_relation')
    assert motion.fact_value is None and not motion.certified
