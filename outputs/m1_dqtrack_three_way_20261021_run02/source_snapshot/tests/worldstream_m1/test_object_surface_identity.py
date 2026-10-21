"""Signed geometry fixtures exercise claims, never real-world qualification."""
from dataclasses import replace
import json
import numpy as np
import pytest
from worldstream.m1 import FullOutputs,WorldStateManager,WorldAnchor
from worldstream.m1.geometry import box_geometry,project_anchor,state_geometry_version
from worldstream.m1.identity_continuity import identity_protocol_digest,IDENTITY_CLAIM
from worldstream.m1.identity_object import PRODUCER,DEPENDENCY
from worldstream.m1.identity_temporal import camera_snapshot_digest,certify_temporal_identity
from worldstream.m1.input_contract import frame_digest
from worldstream.m1.state_types import PhysicalEvidence


def object_receipt(config,make_frame,make_prediction):
    cfg=replace(config,temporal_visibility=True,identity_continuity=True,identity_method='temporal_object')
    m=WorldStateManager(cfg);old=make_frame(support=False);p=make_prediction(semantic='car')
    p.geometry_world=box_geometry([0,0,10],[8,6,2]);m.initialize(old,FullOutputs([p]))
    a=next(iter(m.store.anchors.values()));f=make_frame(100.5,support=False);cam=f.cameras['front'];cam.world_to_camera[0,3]=-1
    b=WorldAnchor(a.anchor_id,f.scene_id,'object','car',box_geometry([.8,0,10],[8,6,2]),f.timestamp,
        source_frame_id=f.frame_id,source_state_epoch=a.source_state_epoch,source_execution_signature=f.execution_signature)
    xyz=np.array([[-2,-2,10],[2,-2,10],[-2,2,10],[2,2,10]],float)
    ring=np.array([[-12,-10,20],[12,-10,20],[-12,10,20],[12,10,20]],float)
    def uv(x,c):
        z=x@c.world_to_camera[:3,:3].T+c.world_to_camera[:3,3];h=z@c.intrinsic.T;return (h[:,:2]/h[:,2:]).tolist()
    obs=a.observation_support['front'];source=old.cameras['front']
    view=dict(status='supported',camera='front',observation_model='translation_compensated_box_surface_ring',
        source_points=uv(xyz,source),current_points=uv(xyz+[.8,0,0],cam),
        ring_source_points=uv(ring,source),ring_current_points=uv(ring,cam),
        source_region=list(obs.region),current_region=list(project_anchor(b,cam,cfg).region),
        source_timestamp=source.timestamp,current_timestamp=cam.timestamp,source_coordinate_space=source.coordinate_space,
        current_coordinate_space=cam.coordinate_space,source_descriptor=obs.descriptor.tolist(),
        fb_error_px=[0.]*4,patch_residual=[0.]*4,ring_fb_error_px=[0.]*4,ring_patch_residual=[0.]*4,
        total_tracks=20,background_gap_m=[10.,10.],source_intrinsic=source.intrinsic.tolist(),
        source_world_to_camera=source.world_to_camera.tolist(),source_camera_snapshot_sha256=camera_snapshot_digest(source))
    measurement=dict(source_class='car',current_class='car',source_center=a.geometry_world.center.tolist(),
        current_center=b.geometry_world.center.tolist(),displacement_m=.8,views=[view],dependency=DEPENDENCY,
        producer_config=cfg.to_dict(),descriptor_config={k:getattr(cfg,k) for k in ('descriptor','descriptor_size','min_texture_std','min_region_area_px')},
        source_camera_sha256={'front':camera_snapshot_digest(source)},competing_edges=[])
    e=PhysicalEvidence(a.anchor_id,a.source_frame_id,f.frame_id,f.scene_id,a.source_state_epoch,state_geometry_version(a),
        state_geometry_version(b),m._previous_frame_digest,frame_digest(f),f.execution_signature,f.timestamp,PRODUCER,json.dumps(measurement))
    return m,a,b,f,e


@pytest.mark.parametrize('mutation',[None,'background','motion','source_digest','crop','descriptor','camera_space','time',
    'geometry','duplicate_ring','ring_fb','ring_outside_image','gap','conflict','forbidden'])
def test_object_surface_receipt_is_strict_and_never_qualifies_without_independent_logs(config,make_frame,make_prediction,mutation):
    m,a,b,f,e=object_receipt(config,make_frame,make_prediction);data=json.loads(e.measurements_json);v=data['views'][0]
    if mutation=='background':v['source_points']=v['ring_source_points'];v['current_points']=v['ring_current_points']
    elif mutation=='motion':v['current_points']=(np.asarray(v['current_points'])+[5,0]).tolist()
    elif mutation=='source_digest':v['source_camera_snapshot_sha256']='wrong'
    elif mutation=='crop':v['source_region'][0]+=1
    elif mutation=='descriptor':v['source_descriptor'][0][0]+=1
    elif mutation=='camera_space':v['source_coordinate_space']='other'
    elif mutation=='time':v['source_timestamp']=f.timestamp
    elif mutation=='geometry':e=replace(e,source_state_version='wrong')
    elif mutation=='duplicate_ring':v['ring_source_points']=[v['ring_source_points'][0]]*4
    elif mutation=='ring_fb':v['ring_fb_error_px'][0]=-.1
    elif mutation=='ring_outside_image':
        v['ring_source_points']=[[0.,10.],[119.,10.],[0.,90.],[119.,90.]]
        v['ring_current_points']=(np.asarray(v['ring_source_points'])+[-3.,0.]).tolist()
    elif mutation=='gap':v['background_gap_m']=[100.,100.]
    elif mutation=='conflict':data['views'].append(dict(status='contradicted'))
    elif mutation=='forbidden':data['GT_instance']='illegal'
    e=m._physical_issuer._mint_measured(replace(e,measurements_json=json.dumps(data)))
    r=certify_temporal_identity(a,b,f,e,m._physical_issuer,m._previous_frame_digest,identity_protocol_digest(m.config,f))
    assert not r.certified and r.certified_claim is None
    if mutation is None:assert r.rejection_reasons==('independent_producer_qualification_missing',),r.rejection_reasons
    else:assert set(r.rejection_reasons)-{'independent_producer_qualification_missing'}
