from dataclasses import replace
import copy
import numpy as np
from worldstream.m1 import WorldStateManager, FullOutputs
from worldstream.m1.association_observation import refine
from worldstream.m1.state_types import AnchorObservation
from worldstream.m1.observation_validator import ObservationValidator
from worldstream.m1.geometry import project_anchor, box_geometry


def setup(config, make_frame, make_anchor, make_prediction):
    frame=make_frame(t=100.5)
    cfg=replace(config,uncertainty_px=0.)
    anchors=[make_anchor(-.7,semantic='traffic_cone'),make_anchor(.7,semantic='traffic_cone')]
    preds=[make_prediction(-.7,semantic='traffic_cone'),make_prediction(.7,semantic='traffic_cone')]
    validator=ObservationValidator(cfg)
    for a,p in zip(anchors,preds):
        a.geometry_world=box_geometry(a.geometry_world.center,[.8,.8,.8])
        p.geometry_world=copy.deepcopy(a.geometry_world)
        a.source_execution_signature=frame.execution_signature
    descriptors=[]
    for a in anchors:
        r=project_anchor(a,frame.cameras['front'],cfg)
        descriptors.append(validator.descriptor(frame.cameras['front'].image,r.region))
    assert all(d is not None for d in descriptors)
    for i,a in enumerate(anchors):
        a.observation_support={'front':AnchorObservation('front',(0,0,1,1),descriptors[1-i],100.,1.)}
    return frame,cfg,anchors,preds,validator


def test_visual_unique_bijection_for_non_tracking_class(config,make_frame,make_anchor,make_prediction):
    f,c,a,p,v=setup(config,make_frame,make_anchor,make_prediction)
    pairs,changed,d=refine(a,p,np.ones((2,2)),[],f,100.,c,v)
    assert set(pairs)=={(0,1),(1,0)} and changed==set(pairs)
    assert d['mismatching_edges']==2 and not d['certified']
    assert all(not x.identity_link_qualified and x.certified_observation_count==0 for x in a)


def test_visual_unknown_illegal_reappearance_keep_baseline(config,make_frame,make_anchor,make_prediction):
    f,c,a,p,v=setup(config,make_frame,make_anchor,make_prediction)
    for fault in ('missing','illegal','reappearance','namespace'):
        old=copy.deepcopy(a);costs=np.ones((2,2))
        if fault=='missing':old[0].observation_support={}
        if fault=='illegal':costs[0,1]=np.inf
        if fault=='reappearance':old[0].source_timestamp=99.
        if fault=='namespace':old[0].source_execution_signature='other'
        pairs,changed,_=refine(old,p,costs,[(0,0),(1,1)],f,100.,c,v)
        assert pairs==[(0,0),(1,1)] and not changed


def test_default_geometry_and_physical_counts(config,make_frame,make_prediction):
    results=[]
    for enabled in (False,True):
        m=WorldStateManager(replace(config,association_observation_guard=enabled))
        for t in (100.,100.5):
            f=make_frame(t=t);r=m.precheck(f);m.commit(f,FullOutputs([make_prediction(semantic='traffic_cone')]),r)
        results.append(m)
    assert results[0].last_commit_diagnostics['association_pairs']==results[1].last_commit_diagnostics['association_pairs']
    assert all(a.certified_observation_count==0 for m in results for a in m.store.anchors.values())


def test_conflict_requires_bound_mature_history(config,make_frame,make_anchor):
    from worldstream.m1.dqtrack_bridge import identity_conflicts
    a=make_anchor();a.source_frame_id='old';f=make_frame(t=100.5)
    old=dict(status='aligned',frame_id='old',timestamp=100.,namespace=['run','scene'],tracking_id='1',
        aligned_observations=2,expires_at=101.)
    new=dict(old,frame_id=f.frame_id,timestamp=f.timestamp,tracking_id='2')
    for fault in ('ok','newborn','expired','missing','run','binding'):
        a.external_tracking=copy.deepcopy(old);h=copy.deepcopy(new)
        if fault=='newborn':a.external_tracking['aligned_observations']=1
        if fault=='expired':a.external_tracking['expires_at']=100.1
        if fault=='missing':h={}
        if fault=='run':h['namespace']=['other','scene']
        if fault=='binding':a.external_tracking['frame_id']='unrelated'
        found=identity_conflicts([a],{0:h},[(0,0)],f,100.,2.)
        assert bool(found)==(fault=='ok')


def test_visual_commit_changes_only_legal_pairs_without_certification(config,make_frame,make_prediction):
    from worldstream.m1.geometry import region_slices
    cfg=replace(config,association_observation_guard=True,uncertainty_px=0.)
    m=WorldStateManager(cfg)
    preds=[make_prediction(-.7,semantic='traffic_cone'),make_prediction(.7,semantic='traffic_cone')]
    for p in preds:p.geometry_world=box_geometry(p.geometry_world.center,[.8,.8,.8])
    f=make_frame();r=m.precheck(f);m.commit(f,FullOutputs(preds),r)
    anchors=list(m.store.anchors.values());ids=[a.anchor_id for a in anchors]
    f=make_frame(t=100.5);cam=f.cameras['front']
    regions=[region_slices(a.observation_support['front'].region,cam.image.shape) for a in anchors]
    first,second=[cam.image[region].copy() for region in regions]
    assert first.shape==second.shape
    cam.image[regions[0]]=second;cam.image[regions[1]]=first
    r=m.precheck(f);m.commit(f,FullOutputs(preds),r)
    assert m.last_commit_diagnostics['association_pairs']==[[ids[0],1],[ids[1],0]]
    assert all(not a.identity_link_qualified and a.certified_observation_count==0
        and a.association_basis=='uncertified_visual_geometry' for a in m.store.anchors.values())
