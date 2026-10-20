"""Experimental same-RGB foreground *proposal*, never an instance mask.

Box-constrained triangulation alone admits background. Require observable rays
and measured depth separation from a surrounding ring in both actual views.
Static object/background ambiguity, missing ring or conflicting views return
Unknown. Qualification against independent labels is still mandatory.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .geometry import oriented_box_membership, project_anchor, valid_projection_camera, state_geometry_version
from .input_contract import frame_digest
from .physical_certifier import PRODUCER_ID
from .state_types import PhysicalEvidence

_PRODUCER_CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def producer_protocol_digest(config, frame):
    # Qualification must not transfer across changed association/observation
    # policies or producer code. Exclude output/performance settings and the
    # qualification artifact itself to avoid a circular qualification hash.
    excluded={'physical_qualification_path','physical_qualification_sha256',
        'logging','log_path','profiling','visualization','capture_tensors',
        'capture_backbone','max_tensor_bytes','storage_device','temporal_workers'}
    protocol=dict(producer=PRODUCER_ID,producer_code_sha256=_PRODUCER_CODE_SHA256,
        measurement_code={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
        evidence_config={k:v for k,v in config.to_dict().items() if k not in excluded},
        gap_m=config.foreground_depth_gap_m,
        min_points=config.support_min_points, angle_deg=config.min_triangulation_angle_deg,
        reprojection_px=config.flow_epipolar_max_px, fb_px=config.flow_fb_max_px,
        coordinate_space={cid:(c.coordinate_space,list(c.image.shape)) for cid,c in frame.cameras.items()},
        proposal_dependency='RGB_correspondence_with_predicted_box_proposal')
    return hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()


def triangulate_correspondences(previous, current, evidence):
    """Return measured world points, parallax, reprojection and depth validity."""
    p=np.asarray(evidence.source_points,float);q=np.asarray(evidence.current_points,float)
    n=len(p);xyz=np.full((n,3),np.nan);angles=np.zeros(n);error=np.full(n,np.inf)
    depths=np.full((n,2),np.nan);valid=np.zeros(n,bool)
    if not n or not all(valid_projection_camera(c) and c.timing_reliable for c in (previous,current)):
        return xyz,angles,error,depths,valid
    rays0=np.c_[p,np.ones(n)]@np.linalg.inv(previous.intrinsic).T@previous.world_to_camera[:3,:3]
    rays1=np.c_[q,np.ones(n)]@np.linalg.inv(current.intrinsic).T@current.world_to_camera[:3,:3]
    cosine=np.sum(rays0*rays1,1)/(np.linalg.norm(rays0,axis=1)*np.linalg.norm(rays1,axis=1))
    angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    homogeneous=cv2.triangulatePoints(previous.intrinsic@previous.world_to_camera[:3],
        current.intrinsic@current.world_to_camera[:3],p.T.copy(),q.T.copy())
    with np.errstate(invalid='ignore',divide='ignore'):xyz=(homogeneous[:3]/homogeneous[3:]).T
    errors=[]
    for k,(cam,uv) in enumerate(((previous,p),(current,q))):
        points=xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3]
        depths[:,k]=points[:,2];uvh=points@cam.intrinsic.T
        with np.errstate(invalid='ignore',divide='ignore'):reproj=uvh[:,:2]/uvh[:,2:]
        errors.append(np.linalg.norm(reproj-uv,axis=1))
    error=np.maximum(*errors)
    valid=(np.isfinite(xyz).all(1)&np.isfinite(error)&(depths>.5).all(1)&(depths<200).all(1))
    return xyz,angles,error,depths,valid


def _inside(points, region):
    return ((points[:,0]>=region[0])&(points[:,0]<region[2])&
            (points[:,1]>=region[1])&(points[:,1]<region[3]))


def propose_foreground_correspondence(anchor, current_anchor, frame, previous_cameras,
        evidence_by_camera, config, issuer=None, source_frame_digest=None,
        actual_source_frame_id=None, association_alternatives=0):
    tick=perf_counter();views=[];reasons=[];point_records=[];selected_records=[]
    if (anchor.anchor_type!='object' or current_anchor.anchor_type!='object'
            or anchor.source_frame_id!=actual_source_frame_id or not source_frame_digest):
        reasons.append('actual_source_frame_unavailable')
    if association_alternatives:reasons.append('identity_ambiguous')
    conflicts=0;contradictions=0
    for cid,cam in frame.cameras.items():
        old=previous_cameras.get(cid);e=evidence_by_camera.get(cid);obs=anchor.observation_support.get(cid)
        view=dict(camera=cid,reason='source_or_correspondence_missing',candidate_points=0,ring_points=0)
        views.append(view)
        if old is None or e is None or obs is None:continue
        if (abs(old.timestamp-obs.timestamp)>1e-5 or abs(e.source_timestamp-old.timestamp)>1e-5
                or e.timestamp!=cam.timestamp or e.camera_id!=cid):
            view['reason']='source_time_or_camera_mismatch';continue
        projection=project_anchor(current_anchor,cam,config)
        if projection.region is None:view['reason']='target_outside_view';continue
        p=np.asarray(e.source_points);q=np.asarray(e.current_points)
        if not len(p):view['reason']='no_tracks_or_low_texture';continue
        if (p.ndim!=2 or p.shape[1:]!=(2,) or q.shape!=p.shape
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or len(np.unique(p,axis=0))!=len(p) or len(np.unique(q,axis=0))!=len(q)):
            view['reason']='nonunique_or_invalid_measured_tracks'
            conflicts+=1
            continue
        if (old.camera_id!=cid or cam.camera_id!=cid or obs.camera_id!=cid
                or old.coordinate_space!=cam.coordinate_space or not old.timing_reliable or not cam.timing_reliable):
            view['reason']='source_coordinate_or_timing_mismatch'
            conflicts+=1
            continue
        xyz,angles,error,depths,valid=triangulate_correspondences(old,cam,e)
        raw=e.raw_correspondence_supported if e.raw_correspondence_supported is not None else e.supported
        measured=e.correspondence_valid&e.geometry_valid&raw
        observable=(valid&measured&(angles>=config.min_triangulation_angle_deg)
                    &(error<=config.flow_epipolar_max_px)&(e.fb_error_px<=config.flow_fb_max_px))
        inside0=_inside(p,obs.region);inside1=_inside(q,projection.region)
        contradictions+=int((inside0&inside1&e.correspondence_valid&~e.geometry_valid).sum())
        box_points=(observable&inside0&inside1&oriented_box_membership(xyz,anchor.geometry_world,0.)
                    &oriented_box_membership(xyz,current_anchor.geometry_world,0.))
        selected=np.flatnonzero(box_points)
        roi_count=int((inside0&inside1&measured).sum())
        view.update(roi_tracks=roi_count,observable_roi_points=int((observable&inside0&inside1).sum()),candidate_points=len(selected))
        if not len(selected):
            view['reason']='parallax_or_box_membership_insufficient'
            if roi_count>=config.support_min_points and (observable&inside0&inside1).sum()>=config.support_min_points:
                conflicts+=1
            continue
        r=np.array(obs.region);s=np.array(projection.region)
        margin0=np.maximum((r[2:]-r[:2])*.5,12);margin1=np.maximum((s[2:]-s[:2])*.5,12)
        ring0=np.r_[r[:2]-margin0,r[2:]+margin0];ring1=np.r_[s[:2]-margin1,s[2:]+margin1]
        ring=observable&_inside(p,ring0)&_inside(q,ring1)&~inside0&~inside1
        ridx=np.flatnonzero(ring);view['ring_points']=len(ridx)
        size=np.maximum(r[2:]-r[:2],1)
        quadrants=len(np.unique(np.clip(((p[selected]-r[:2])/size*2).astype(int),0,1),axis=0))
        view['spatial_quadrants']=quadrants
        records=[dict(camera=cid,source_uv=p[i].tolist(),current_uv=q[i].tolist(),world_xyz=xyz[i].tolist(),
            parallax_deg=float(angles[i]),reprojection_error_px=float(error[i]),fb_error_px=float(e.fb_error_px[i])) for i in selected[:64]]
        point_records.extend(records)
        if len(ridx)<config.support_min_points:
            view['reason']='background_depth_unobservable';continue
        gap=np.percentile(depths[ridx],25,axis=0)-np.percentile(depths[selected],90,axis=0)
        view['foreground_background_depth_gap_m']=gap.tolist()
        if np.any(gap<config.foreground_depth_gap_m):
            view['reason']='target_and_background_indistinguishable';continue
        if len(selected)<config.support_min_points or quadrants<3:
            view['reason']='insufficient_object_spatial_support';continue
        view['reason']='candidate_same_RGB_not_qualified_identity'
        for record in records:record['depth_gap_m']=float(gap.min())
        selected_records.extend(records)
    supported_views=[v for v in views if v['reason']=='candidate_same_RGB_not_qualified_identity']
    if not supported_views:reasons.append('no_distinguishable_foreground_view')
    if conflicts:reasons.append('multi_camera_geometry_conflict')
    if contradictions:reasons.append('static_ray_hypothesis_contradicted_or_unresolved')
    receipt=None
    if not reasons and selected_records and issuer is not None:
        measurement=dict(world_points=[r['world_xyz'] for r in selected_records],
            fb_error_px=[r['fb_error_px'] for r in selected_records],parallax_deg=[r['parallax_deg'] for r in selected_records],
            reprojection_error_px=[r['reprojection_error_px'] for r in selected_records],
            foreground_background_depth_gap_m=[r['depth_gap_m'] for r in selected_records],
            measured_tracks=selected_records,
            spatial_quadrants=max(v['spatial_quadrants'] for v in supported_views),
            association_alternatives=association_alternatives,conflicting_camera_count=conflicts,
            static_ray_contradictions=contradictions,dependency='RGB_correspondence_with_predicted_box_proposal')
        receipt=issuer._mint_measured(PhysicalEvidence(anchor.anchor_id,anchor.source_frame_id,frame.frame_id,
            frame.scene_id,anchor.source_state_epoch,state_geometry_version(anchor),state_geometry_version(current_anchor),
            source_frame_digest,frame_digest(frame),frame.execution_signature,frame.timestamp,PRODUCER_ID,
            json.dumps(measurement,sort_keys=True,allow_nan=False)))
    return dict(status='candidate' if not reasons else 'unknown',rejection_reasons=reasons,
        foreground_ownership_certified=False,independent_sensor=False,views=views,
        candidate_points=point_records,selected_points=selected_records,evidence=receipt,
        producer_cpu_ms=(perf_counter()-tick)*1000,protocol_digest=producer_protocol_digest(config,frame))
