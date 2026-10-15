from dataclasses import replace
import copy
import cv2
import numpy as np
from worldstream.m1 import M1Config, CameraFrame, FullOutputs, WorldStateManager
from worldstream.m1.temporal_visibility import build_evidence
from worldstream.m1.novelty_detector import unexplained_regions


def pair(dx=4):
    rng=np.random.RandomState(40)
    image=cv2.GaussianBlur(rng.randint(0,256,(160,240),np.uint8),(3,3),.6)
    k=np.array([[120.,0,120],[0,120.,80],[0,0,1.]])
    a=CameraFrame('c',image,k,np.eye(4),100.)
    moved=cv2.warpAffine(image,np.array([[1.,0,dx],[0,1,0]]),(240,160))
    pose=np.eye(4); pose[0,3]=dx/12  # plane at z=10
    b=CameraFrame('c',moved,k,pose,100.5)
    return a,b


def cfg(**kw):
    return M1Config(temporal_visibility=True,local_support=True,**kw)


def test_bidirectional_geometry_and_boundary():
    a,b=pair(); e=build_evidence(a,b,cfg())
    assert e.supported.sum()>100
    assert np.median(e.fb_error_px[e.supported])<.1
    assert np.allclose(np.median(e.current_points[e.supported]-e.source_points[e.supported],axis=0),[4,0],atol=.1)
    assert e.stable_mask.any() and not e.stable_mask[:,:4].any()
    assert not e.transformer_dependency_certified


def test_obvious_occlusion_not_explained():
    a,b=pair(); b.image[40:120,80:160]=0
    e=build_evidence(a,b,cfg())
    assert not e.stable_mask[45:115,85:155].any()
    assert e.stable_mask[:,170:].any()  # local failure does not invalidate unrelated evidence


def test_wrong_geometry_and_texture_fail_closed():
    a,b=pair(); b.world_to_camera=np.eye(4)
    e=build_evidence(a,b,cfg())
    assert not e.stable_mask.any()
    a.image[:]=100; b.image[:]=100
    assert not build_evidence(a,b,cfg()).stable_mask.any()


def test_masks_measured_not_rectangle_and_partition():
    a,b=pair(); c=cfg(); e=build_evidence(a,b,c)
    meta,invalid=unexplained_regions(b,[(0,0,240,160)],[],c,temporal_evidence=e)
    assert np.array_equal(~invalid,e.stable_mask)
    assert 0 < meta['verified_stable_area_ratio'] < .9
    assert abs(sum(meta[k] for k in ['verified_stable_area_ratio','changed_area_ratio','unknown_area_ratio'])-1)<1e-8
    _,missing=unexplained_regions(b,[(0,0,240,160)],[],c)
    assert missing.all()
    _,refresh=unexplained_regions(b,[],[],c,True,e)
    assert refresh.all()


def test_scene_and_source_version_isolation(config,make_frame,make_prediction):
    c=replace(config,temporal_visibility=True,local_support=True)
    m=WorldStateManager(c); f=make_frame(support=False)
    m.initialize(f,FullOutputs([make_prediction()]))
    before=m.store.feature_versions(); old_image=m._previous_cameras['front'].image.copy()
    r=m.precheck(make_frame(100.5,support=False))
    assert m._previous_cameras['front'].timestamp==100.
    assert m.store.feature_versions().keys()==before.keys()
    for k,v in before.items(): assert m.store.feature_versions()[k]['computation_version']==v['computation_version']
    f.cameras['front'].image[:]=0
    assert np.array_equal(old_image,m._previous_cameras['front'].image)
    r=m.precheck(make_frame(100.6,scene='new',support=False))
    assert not r.valid_anchor_ids and m.last_masks['front'].all()
    assert not m._previous_cameras


def test_unknown_dynamic_motion_remains_invalid(config,make_frame,make_prediction):
    c=replace(config,temporal_visibility=True,local_support=True)
    m=WorldStateManager(c)
    m.initialize(make_frame(support=False),FullOutputs([make_prediction(semantic='car')]))
    r=m.precheck(make_frame(100.5,support=False))
    assert not r.valid_anchor_ids
    assert 'motion_unknown' in next(iter(r.diagnostics['anchors'].values()))['reasons']


def test_invalid_calibration_time_and_old_source():
    a,b=pair(); b.intrinsic[:]=0
    assert not build_evidence(a,b,cfg()).stable_mask.any()
    a,b=pair(); b.timestamp=102
    assert not build_evidence(a,b,cfg()).stable_mask.any()
    a,b=pair(); b.timing_reliable=False
    assert not build_evidence(a,b,cfg()).stable_mask.any()


def test_local_support_ablation_preserves_anchor_decisions(config,make_frame,make_prediction):
    results=[]
    for local in (True,False):
        m=WorldStateManager(replace(config,temporal_visibility=True,local_support=local))
        m.initialize(make_frame(support=False),FullOutputs([make_prediction()]))
        r=m.precheck(make_frame(100.5,support=False)); results.append(r.valid_anchor_ids)
    assert results[0]==results[1]


def test_cached_camera_projection_equivalent(config,make_frame,make_anchor):
    from worldstream.m1.geometry import project_anchor,valid_projection_camera
    rng=np.random.RandomState(42)
    cam=make_frame(100.5).cameras['front']
    for _ in range(40):
        a=make_anchor(x=float(rng.uniform(-30,30)))
        normal=project_anchor(a,cam,config)
        cached=project_anchor(a,cam,config,valid_projection_camera(cam))
        assert normal.region==cached.region and normal.reason==cached.reason and normal.valid==cached.valid
        assert np.allclose(normal.uv,cached.uv,equal_nan=True)


def test_feature_oracle_bilinear_alignment():
    from tools.worldstream_m1.evaluate_feature_oracle import bilinear,errors
    yy,xx=np.mgrid[:3,:4]
    f=np.stack([xx,yy],axis=-1)
    values,valid=bilinear(f,np.array([[8.,8.],[24.,24.],[32.,24.]]),(48,64))
    assert valid.all()
    assert np.allclose(values,[[0,0],[1,1],[1.5,1]])
    l2,cos=errors(np.array([[1.,2.]]),np.array([[1.,2.]]))
    assert l2[0]==0 and np.isclose(cos[0],1)


def test_exact_mask_roundtrip_and_export(config,make_frame,make_prediction):
    from worldstream.m1.mask_codec import encode_mask,decode_mask
    mask=np.random.RandomState(4).rand(53,67)>.5
    assert np.array_equal(mask,decode_mask(encode_mask(mask)))
    manager=WorldStateManager(replace(config,temporal_visibility=True,local_support=True))
    manager.initialize(make_frame(support=False),FullOutputs([make_prediction()]))
    manager.precheck(make_frame(100.5,support=False))
    result=manager.export_m2_inputs()
    assert np.array_equal(manager.last_masks['front'],decode_mask(result.invalidation_regions['front']['exact_invalidation_mask']))


def test_local_background_without_object_anchor_survives(config,make_frame):
    m=WorldStateManager(replace(config,temporal_visibility=True,local_support=True))
    m.initialize(make_frame(support=False),FullOutputs([]))
    r=m.precheck(make_frame(100.5,support=False))
    assert not r.full_refresh_hint
    assert not m.last_masks['front'].all()
