"""Unqualified producer outputs cannot silently change ordinary tracking state."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.state_store import StateStore


def fake_producer(edges):
    def produce(anchors,predictions,costs,frame,*args,**kwargs):
        return {(int(i),int(j)):dict(status='candidate' if (i,j) in edges else 'unknown',
            rejection_reasons=[] if (i,j) in edges else ['no_observation'],evidence=None,
            measurements=dict(views=[],source_class=anchors[i].semantic_class),producer_cpu_ms=0.,
            recovered_short_gap=False,resolved_competition=False,source_frame_digest='untrusted')
            for i,j in zip(*np.nonzero(np.isfinite(costs))) if anchors[i].anchor_type=='object'}
    return produce


def manager(config,make_frame,make_prediction,count=1,identity=True):
    cfg=replace(config,identity_continuity=identity,identity_method='temporal',temporal_visibility=True)
    m=WorldStateManager(cfg)
    m.initialize(make_frame(support=False),FullOutputs([make_prediction(x=i,semantic='car') for i in range(count)]))
    return m


def test_two_predictions_cannot_inherit_one_unqualified_proposal_anchor(config,make_frame,make_prediction,monkeypatch):
    m=manager(config,make_frame,make_prediction)
    monkeypatch.setattr('worldstream.m1.identity_temporal.temporal_identity_pairs',fake_producer({(0,0),(0,1)}))
    f=make_frame(100.5,support=False);r=m.precheck(f)
    m.commit(f,FullOutputs([make_prediction(.2,semantic='car'),make_prediction(1.2,semantic='car')]),r)
    committed=m.last_commit_diagnostics['committed_anchor_ids']
    assert len(committed)==len(set(committed))==2
    assert len(m.store.anchors)==2
    assert not m.get_current_certifications().certified_anchor_ids


def test_unqualified_competing_proposal_does_not_override_geometry(config,make_frame,make_prediction,monkeypatch):
    a=manager(config,make_frame,make_prediction);b=manager(config,make_frame,make_prediction,identity=False)
    monkeypatch.setattr('worldstream.m1.identity_temporal.temporal_identity_pairs',fake_producer({(0,1)}))
    for m in (a,b):
        f=make_frame(100.5,support=False);r=m.precheck(f)
        m.commit(f,FullOutputs([make_prediction(.1,semantic='car'),make_prediction(1.3,semantic='car')]),r)
    assert a.last_commit_diagnostics['association_pairs']==b.last_commit_diagnostics['association_pairs']
    assert a.last_commit_diagnostics['committed_anchor_ids']==b.last_commit_diagnostics['committed_anchor_ids']


def test_two_sources_cannot_duplicate_current_prediction(config,make_frame,make_prediction,monkeypatch):
    m=manager(config,make_frame,make_prediction,count=2)
    monkeypatch.setattr('worldstream.m1.identity_temporal.temporal_identity_pairs',fake_producer({(0,0),(1,0)}))
    f=make_frame(100.5,support=False);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(.3,semantic='car')]),r)
    pairs=m.last_commit_diagnostics['association_pairs']
    assert len({j for _,j in pairs})==len(pairs)


def test_shadow_hypotheses_are_defensive_and_commit_atomic(config,make_frame,make_prediction,monkeypatch):
    m=manager(config,make_frame,make_prediction)
    monkeypatch.setattr('worldstream.m1.identity_temporal.temporal_identity_pairs',fake_producer({(0,0)}))
    f=make_frame(100.5,support=False);r=m.precheck(f)
    m.commit(f,FullOutputs([make_prediction(.1,semantic='car')]),r)
    saved=m.get_identity_hypotheses();assert saved
    saved[0]['status']='forged'
    assert m.get_identity_hypotheses()[0]['status']!='forged'
    before=copy.deepcopy(m.get_identity_hypotheses());ids=list(m.store.anchors)
    f=make_frame(101.,support=False);r=m.precheck(f)
    def fail(*a):raise RuntimeError('rollback')
    monkeypatch.setattr(StateStore,'put_feature',fail)
    with pytest.raises(RuntimeError):m.commit(f,FullOutputs([make_prediction(.2,semantic='car')]),r)
    assert m.get_identity_hypotheses()==before and list(m.store.anchors)==ids


@pytest.mark.parametrize('policy',['shadow','certified_only','experimental'])
def test_explicit_tracking_policy_keeps_conflicts_one_to_one_and_unqualified(config,make_frame,make_prediction,monkeypatch,policy):
    cfg=replace(config,identity_tracking_policy=policy)
    if policy!='shadow':
        with pytest.raises(ValueError,match='Retired'):manager(cfg,make_frame,make_prediction,count=2)
        return
    m=manager(cfg,make_frame,make_prediction,count=2)
    monkeypatch.setattr('worldstream.m1.identity_temporal.temporal_identity_pairs',fake_producer({(0,0),(0,1),(1,0)}))
    f=make_frame(100.5,support=False);r=m.precheck(f)
    m.commit(f,FullOutputs([make_prediction(.3,semantic='car'),make_prediction(1.3,semantic='car')]),r)
    pairs=m.last_commit_diagnostics['association_pairs']
    assert len({i for i,j in pairs})==len(pairs)==len({j for i,j in pairs})
    assert len(set(m.last_commit_diagnostics['committed_anchor_ids']))==2
    assert not m.get_current_certifications().certified_anchor_ids
    assert not any(a.identity_link_qualified for a in m.store.anchors.values())
    if policy!='experimental':assert not any(h['applied_to_tracking'] for h in m.get_identity_hypotheses())
    m.reset('other');assert not m.get_identity_hypotheses()



def test_qualified_selective_readout_and_tracking_provenance_agree(config,make_frame,make_prediction):
    # The qualification is a controlled unit fixture, not a real-data result.
    from test_identity_surface import identity_manager
    m,p,f,_=identity_manager(config,make_frame,make_prediction)
    r=m.precheck(f);m.commit(f,FullOutputs([p]),r)
    certified=m.get_current_certifications().certified_anchor_ids
    assert certified
    for aid in certified:
        a=m.store.anchors[aid]
        assert a.identity_link_qualified and a.association_basis=='qualified_identity'
        assert a.association_history[-1]['certified']
