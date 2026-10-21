"""Regression checks for historical-only quality gating and exact fast association."""
from dataclasses import replace
import numpy as np
import pytest
from worldstream.m1 import M1Config, FullOutputs, WorldStateManager, Geometry, PerceptionAnchor, WorldAnchor
from worldstream.m1.association import associate
from worldstream.m1.geometry import predict_geometry, sample_polyline, box_geometry


def scalar_cost(anchors, predictions, timestamp, cfg):
    costs=np.full((len(anchors),len(predictions)),np.inf)
    for i,a in enumerate(anchors):
        if timestamp-a.source_timestamp>cfg.max_state_age_s: continue
        g=predict_geometry(a,timestamp)
        for j,p in enumerate(predictions):
            if (a.anchor_type,a.semantic_class)!=(p.anchor_type,p.semantic_class):continue
            h=p.geometry_world
            if a.anchor_type=='object':
                if g.size is None or h.size is None:continue
                if np.any(np.maximum(g.size/h.size,h.size/g.size)>cfg.box_size_ratio):continue
                d=np.linalg.norm(g.center-h.center)
                gate=cfg.association_gate_m-a.geometry_uncertainty-cfg.velocity_uncertainty_mps*(timestamp-a.source_timestamp)
            else:
                x,y=sample_polyline(g.points),sample_polyline(h.points)
                dx,dy=x[-1]-x[0],y[-1]-y[0]
                if abs(np.dot(dx,dy))/max(np.linalg.norm(dx)*np.linalg.norm(dy),1e-8)<cfg.map_direction_cos:continue
                d=min(np.linalg.norm(x-y,axis=1).mean(),np.linalg.norm(x-y[::-1],axis=1).mean());gate=cfg.map_gate_m
            if d<=gate:costs[i,j]=d
    return costs


@pytest.mark.parametrize('seed',range(5))
def test_vector_association_matches_scalar(seed):
    rng=np.random.RandomState(seed); anchors=[];predictions=[];cfg=M1Config()
    for j in range(40):
        kind='object' if j%2 else 'map'; cls=str(j%3)
        center=rng.normal(size=3)*3
        if kind=='object':g=box_geometry(center,rng.uniform(.5,3,3))
        else:
            points=np.cumsum(rng.normal(size=(12,3)),axis=0)+center;g=Geometry(points,points.mean(0))
        anchors.append(WorldAnchor(str(j),'s',kind,cls,g,100. if j%5 else 98.))
        shift=rng.normal(size=3)*.4
        h=Geometry(g.points+shift,g.center+shift,None if g.size is None else g.size*rng.uniform(.8,1.2))
        if kind=='map' and j%4:h.points=h.points[::-1]
        predictions.append(PerceptionAnchor(kind,cls,h))
    expected=scalar_cost(anchors,predictions,100.5,cfg)
    matches,costs=associate(anchors,predictions,100.5,cfg)
    np.testing.assert_allclose(costs,expected,atol=1e-14,rtol=1e-14)
    # The old maximum-cardinality objective is retired. The cost equivalence
    # above remains exact; allocation now has explicit unmatched options.
    assert len({i for i,j in matches})==len(matches)==len({j for i,j in matches})
    assert all(np.isfinite(expected[i,j]) for i,j in matches)


def test_quality_is_historical_and_commit_only(config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,certification_min_source_score=.7,certification_min_history=2,
                               certification_max_innovation_m=.5))
    p=make_prediction();p.source_score=.9
    m.initialize(make_frame(),FullOutputs([p]))
    r=m.precheck(make_frame(100.5));a=next(iter(m.store.anchors.values()))
    assert a.history_observations==1 and a.last_innovation_m is None
    assert not r.valid_anchor_ids and 'history_insufficient' in r.diagnostics['anchors'][a.anchor_id]['reasons']
    m.commit(make_frame(100.5),FullOutputs([p]),r)
    r=m.precheck(make_frame(101.));a=next(iter(m.store.anchors.values()))
    assert a.history_observations==2 and a.last_innovation_m==0
    assert r.valid_anchor_ids
    versions=m.store.feature_versions()
    m.precheck(make_frame(101.1))
    assert a.history_observations==2
    assert all(m.store.feature_versions()[k]['computation_version']==v['computation_version']
               for k,v in versions.items())
    p.source_score=.4
    m.commit(make_frame(101.1),FullOutputs([p]),m.last_result)
    r=m.precheck(make_frame(101.5))
    assert not r.valid_anchor_ids and 'source_score_insufficient' in r.diagnostics['anchors'][a.anchor_id]['reasons']


@pytest.mark.parametrize('score',[None,float('nan'),.2])
def test_unknown_or_low_source_score_fails_closed(score,config,make_frame,make_prediction):
    m=WorldStateManager(replace(config,certification_min_source_score=.5))
    p=make_prediction();p.source_score=score;m.initialize(make_frame(),FullOutputs([p]))
    r=m.precheck(make_frame(100.5));assert not r.valid_anchor_ids


def test_parallel_camera_evidence_equals_serial(config,make_frame):
    from copy import deepcopy
    snapshots=[]
    for workers in (1,3):
        m=WorldStateManager(replace(config,temporal_visibility=True,local_support=True,temporal_workers=workers))
        for t in (100.,100.5):
            f=make_frame(t,support=False)
            f.cameras={str(i):deepcopy(f.cameras['front']) for i in range(3)}
            for cid,camera in f.cameras.items():camera.camera_id=cid
            r=m.precheck(f);m.commit(f,FullOutputs([]),r)
        snapshots.append({k:(e.stable_mask.copy(),e.supported.copy(),e.current_points.copy()) for k,e in m.last_evidence.items()})
    for cid in snapshots[0]:
        for a,b in zip(snapshots[0][cid],snapshots[1][cid]):np.testing.assert_array_equal(a,b)
