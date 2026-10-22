"""Ordinary association is not a physical certificate; pending sources stay immutable."""
from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import M1Config,WorldStateManager,FullOutputs
from worldstream.m1.association import legal_assignment,association_ambiguities,assignment_edge_alternatives


def test_global_assignment_alternative_differs_from_local_margin():
    costs=np.array([[1.,2.],[1.1,np.inf]])
    pairs=legal_assignment(costs);local=association_ambiguities(pairs,costs)
    global_proof=assignment_edge_alternatives(pairs,costs)
    assert local[0,1]['ambiguous']
    assert global_proof[0,1]['cardinality_lost']==1
    assert global_proof[0,1]['global_cost_margin_m'] is None
    ties=np.ones((2,2));proof=assignment_edge_alternatives(legal_assignment(ties),ties)
    assert all(p['global_cost_margin_m']==0 for p in proof.values())

def test_external_prediction_and_fork_arrays_are_isolated(config,make_frame,make_prediction):
    m=WorldStateManager(config);prediction=make_prediction();m.initialize(make_frame(),FullOutputs([prediction]))
    anchor=next(iter(m.store.anchors.values()));before=anchor.geometry_world.center.copy()
    prediction.geometry_world.center[:]=88
    assert np.array_equal(anchor.geometry_world.center,before)
    with pytest.raises(ValueError):anchor.geometry_world.center[0]=8
    with pytest.raises(ValueError):anchor.geometry_world.center.setflags(write=True)
    fork=m.store.fork();fork.anchors[anchor.anchor_id].geometry_world.center=np.ones(3)
    assert np.array_equal(anchor.geometry_world.center,before)
    fork.anchors[anchor.anchor_id].observation_support['front'].descriptor=np.zeros((2,2))
    assert anchor.observation_support['front'].descriptor.shape!=(2,2)

def test_source_replacement_during_pending_rejects(config,make_frame,make_prediction):
    m=WorldStateManager(config);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    f=make_frame(100.5);r=m.precheck(f)
    next(iter(m.store.anchors.values())).geometry_world.center=np.zeros(3)
    with pytest.raises(ValueError,match='source state'):m.commit(f,FullOutputs([make_prediction()]),r)
    assert m._pending_store is None

def test_new_detection_drops_old_certificate(config,make_frame,make_prediction):
    m=WorldStateManager(config);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    a=next(iter(m.store.anchors.values()));a.last_certified_timestamp=100.;a.certified_observation_count=1
    version=a.state_version
    f=make_frame(100.5);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(.1)]),r)
    a=next(iter(m.store.anchors.values()))
    assert a.state_version!=version and a.last_certified_timestamp is None and a.physical_certification is None
    assert a.certified_observation_count==1

def test_tensor_acquisition_cannot_mutate_committed_source(config,make_frame,make_prediction):
    cfg=replace(config,capture_tensors=True,max_tensor_bytes=1_000_000)
    m=WorldStateManager(cfg);m.initialize(make_frame(),FullOutputs([make_prediction()]))
    r=m.precheck(make_frame(100.5));candidate=r.valid_feature_candidates[0]
    got=m.get_candidate_feature(candidate['feature_id'],candidate['computation_version'])
    original=m.store.features[candidate['feature_id']]
    assert got is not original
    with pytest.raises(ValueError):got.tensor[:]=0
    fork=m.store.fork();fork.features[candidate['feature_id']].source_camera_ids.append('injected')
    assert 'injected' not in original.source_camera_ids


def test_geometric_association_does_not_sign_physical_identity(config,make_frame,make_prediction):
    manager=WorldStateManager(config)
    manager.initialize(make_frame(),FullOutputs([make_prediction()]))
    frame=make_frame(100.5);result=manager.precheck(frame)
    manager.commit(frame,FullOutputs([make_prediction()]),result)
    assert manager.last_commit_diagnostics['association_pairs']
    assert manager.last_commit_diagnostics['association_is_identity_certification'] is False
    assert not manager.get_current_certifications().certified_anchor_ids
