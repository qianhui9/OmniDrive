from dataclasses import replace
import numpy as np
import cv2
import pytest
from worldstream.m1 import M1Config,CameraFrame,FullOutputs,WorldStateManager
from worldstream.m1.dense_support import dense_support,ray_geometry


def cameras(dx=3):
    rng=np.random.RandomState(701)
    image=cv2.GaussianBlur(rng.randint(0,256,(128,192),np.uint8),(3,3),.7)
    k=np.array([[120.,0,96],[0,120,64],[0,0,1.]])
    old=CameraFrame('c',image,k,np.eye(4),10.)
    new_image=cv2.warpAffine(image,np.array([[1.,0,dx],[0,1,0]]),(192,128))
    pose=np.eye(4);pose[0,3]=dx/12
    return old,CameraFrame('c',new_image,k,pose,10.5)


def cfg(**kw):return M1Config(temporal_visibility=True,local_support=True,dense_local_support=True,**kw)


def test_dense_plane_and_rigid_geometry():
    old,new=cameras();mask,source,d=dense_support(old,new,cfg())
    assert mask.sum()>100 and d['dense_flow_ms']>0
    yy,xx=np.where(mask)
    np.testing.assert_allclose(np.median(source[yy,xx]-np.stack((xx,yy),axis=1),axis=0),[-3,0],atol=.2)
    assert not mask[:,:7].any()


def test_occlusion_and_guard_remove_support():
    old,new=cameras();new.image[35:95,65:130]=0
    mask,_,_=dense_support(old,new,cfg())
    assert not mask[30:100,60:135].any()
    narrow,_,_=dense_support(old,new,cfg(dense_guard_radius_px=2))
    assert not (mask & ~narrow).any() and mask.sum()<narrow.sum()


def test_texture_missing_calibration_and_timing_fail_closed():
    old,new=cameras();old.image[:]=100;new.image[:]=100
    assert not dense_support(old,new,cfg())[0].any()
    for change in ['calibration','timing','age']:
        old,new=cameras()
        if change=='calibration':new.intrinsic[:]=0
        if change=='timing':new.timing_reliable=False
        if change=='age':new.timestamp=15.
        mask,source,_=dense_support(old,new,cfg());assert not mask.any() and source is None


def test_wrong_static_geometry_rejected():
    old,new=cameras();new.world_to_camera=np.eye(4)
    assert not dense_support(old,new,cfg())[0].any()


def test_ray_depth_and_rotation():
    old,new=cameras();source=np.array([[40.,40],[100,80],[150,60]])
    assert ray_geometry(old,new,source,source+[3,0]).all()
    assert not ray_geometry(old,new,source,source-[3,0]).any()
    assert not ray_geometry(old,new,source,source+[3,5]).any()
    assert not ray_geometry(old,new,source,source).any() # unobservable translation depth
    new.world_to_camera=np.eye(4)
    assert ray_geometry(old,new,source,source).all()


def test_dense_does_not_change_anchor_certification(config,make_frame,make_prediction):
    decisions=[]
    for dense in (False,True):
        c=replace(config,temporal_visibility=True,local_support=True,dense_local_support=dense)
        m=WorldStateManager(c);f=make_frame(support=False);m.initialize(f,FullOutputs([make_prediction()]))
        r=m.precheck(make_frame(100.5,support=False));decisions.append((r.valid_anchor_ids,r.invalid_anchor_ids))
        assert m._previous_cameras['front'].timestamp==100.
    assert decisions[0]==decisions[1]


@pytest.mark.parametrize('kwargs',[dict(dense_guard_radius_px=0),dict(dense_guard_radius_px=1.5),dict(dense_local_support=True)])
def test_dense_config_validation(kwargs):
    with pytest.raises(ValueError):M1Config(**kwargs)


def test_motion_guard_only_removes_uncertain_support(config,make_frame,make_prediction):
    masks=[]
    for guard in (False,True):
        m=WorldStateManager(replace(config,temporal_visibility=True,local_support=True,
            dense_local_support=True,dense_motion_guard=guard,dense_guard_radius_px=2))
        prediction=make_prediction(semantic='car',velocity=np.array([1.,0,0]))
        prediction.velocity_reliable=False
        m.initialize(make_frame(support=False),FullOutputs([prediction]))
        result=m.precheck(make_frame(100.5,support=False))
        assert not result.valid_anchor_ids
        masks.append(m.last_masks['front'].copy())
    assert np.all(masks[1] | ~masks[0])
    assert masks[1].sum()>masks[0].sum()


def test_swept_guard_preserves_unclipped_offscreen_geometry(config,make_frame,make_prediction):
    from worldstream.m1 import WorldAnchor
    from worldstream.m1.geometry import project_anchor
    from worldstream.m1.dense_support import swept_dynamic_region
    f=make_frame(100.5);p=make_prediction(x=100.,semantic='car',velocity=np.zeros(3))
    a=WorldAnchor('x','s','object','car',p.geometry_world,100.,velocity_world=np.zeros(3))
    projection=project_anchor(a,f.cameras['front'],config)
    region=swept_dynamic_region(a,f.cameras['front'],projection,config)
    assert projection.region[0]==120. and region[0]>120.


def test_pixel_motion_veto_does_not_relabel_anchor_evidence(config,make_frame,make_prediction):
    from worldstream.m1.geometry import box_geometry
    decisions=[]
    for guard in (False,True):
        m=WorldStateManager(replace(config,temporal_visibility=True,local_support=True,
            dense_local_support=True,dense_motion_guard=guard,dense_guard_radius_px=2))
        static=make_prediction();static.geometry_world=box_geometry([0.,0,10.],[6.,6.,2.])
        moving=make_prediction(x=6.,semantic='car',velocity=np.array([-3.,0,0]));moving.velocity_reliable=False
        m.initialize(make_frame(support=False),FullOutputs([static,moving]))
        result=m.precheck(make_frame(100.5,support=False))
        decisions.append(result.valid_anchor_ids)
    assert decisions[0] and decisions[0]==decisions[1]


@pytest.mark.parametrize('radius',[2,4,8])
def test_exclusion_roi_is_exact_with_neighbourhood_halo(radius):
    old,new=cameras();c=cfg(dense_guard_radius_px=radius)
    reference,src,_=dense_support(old,new,c)
    excluded=np.ones(new.image.shape,bool);excluded[30:100,50:140]=False
    actual,mapping,_=dense_support(old,new,c,exclusion=excluded)
    np.testing.assert_array_equal(actual&~excluded,reference&~excluded)
    np.testing.assert_array_equal(mapping,src)
    empty,mapping,diag=dense_support(old,new,c,exclusion=np.ones_like(excluded))
    assert not empty.any() and mapping is None and diag['dense_skipped_fully_excluded']
