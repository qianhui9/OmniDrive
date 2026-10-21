"""Finite actual-image identity proposals with measured motion and exclusivity.

RGB track continuity is conditional evidence, not instance truth. No static-ray,
stationarity, foreground-mask, centre or velocity claim follows from this path.
Independent claim/population-specific qualification remains mandatory.
"""
import copy
import hashlib
from dataclasses import replace
import json
from time import perf_counter
import cv2
import numpy as np
from .geometry import project_anchor, state_geometry_version, region_slices, valid_projection_camera
from .input_contract import frame_digest, reject_oracle_metadata
from .state_types import PhysicalEvidence, PhysicalCertificationResult, WorldAnchor

PRODUCER='finite_actual_image_identity_v1'
POPULATION='finite_motion_observed_exclusive_identity_pairs'
DEPENDENCY='Actual_source_images_crop_LK_affine_and_optional_temporal_loop_not_instance_ownership'


def camera_snapshot_digest(camera):
    digest=hashlib.sha256(repr((camera.camera_id,camera.coordinate_space,camera.timestamp,camera.timing_reliable)).encode())
    for value in (camera.image,camera.intrinsic,camera.world_to_camera,camera.raw_to_image,camera.raw_intrinsic):
        if value is None:digest.update(b'None;');continue
        a=np.ascontiguousarray(value);digest.update(repr((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def _inside(p,r):
    return (p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])


def quadrants(p,r):
    return len(np.unique(np.clip(((p-np.asarray(r[:2]))/np.maximum(np.asarray(r[2:])-r[:2],1)*2).astype(int),0,1),axis=0)) if len(p) else 0


def _lk(old,new,p):
    if not len(p):return np.empty((0,2)),np.empty(0),np.empty(0,bool)
    args=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,ok,error=cv2.calcOpticalFlowPyrLK(old,new,np.asarray(p,np.float32).reshape(-1,1,2),None,**args)
    if q is None:return np.zeros_like(p),np.full(len(p),np.inf),np.zeros(len(p),bool)
    q=np.where(np.isfinite(q),q,np.asarray(p).reshape(-1,1,2)).astype(np.float32)
    back,rev,_=cv2.calcOpticalFlowPyrLK(new,old,q,None,**args)
    if back is None:return q[:,0],np.full(len(p),np.inf),np.zeros(len(p),bool)
    fb=np.linalg.norm(back[:,0]-p,axis=1);q=q[:,0]
    valid=ok[:,0].astype(bool)&rev[:,0].astype(bool)&np.isfinite(fb)&(fb>=0)&(fb<=.75)&(error[:,0]<=12)
    h,w=new.shape;valid&=np.isfinite(q).all(1)&(q[:,0]>=3)&(q[:,0]<w-3)&(q[:,1]>=3)&(q[:,1]<h-3)
    cells=np.rint(q/2).astype(int);_,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
    valid&=counts[inv]==1
    return q,fb,valid


def _aligned_descriptor(image,matrix,region,validator):
    h,w=image.shape[:2]
    aligned=cv2.warpAffine(image,np.asarray(matrix,float),(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return validator.descriptor(aligned,region)


def _measure_view(anchor,target,old,cam,validator,config,depth_history=None,search_motion=None):
    from .temporal_visibility import gray_image
    from .geometry_support import measured_patch_support
    cid=cam.camera_id;obs=anchor.observation_support.get(cid)
    v=dict(camera=cid,status='unknown',reason='source_observation_unavailable')
    if obs is None:return v
    if (old.camera_id!=cid or obs.camera_id!=cid or old.coordinate_space!=cam.coordinate_space
            or old.image.shape[:2]!=cam.image.shape[:2] or not old.timing_reliable or not cam.timing_reliable
            or not all(valid_projection_camera(c) for c in (old,cam))
            or not np.isfinite([old.timestamp,obs.timestamp,cam.timestamp]).all()
            or old.timestamp!=obs.timestamp or not 0<cam.timestamp-old.timestamp<=min(config.max_state_age_s,1.)):
        return dict(v,reason='source_time_camera_or_coordinate_mismatch')
    source_descriptor=validator.descriptor(old.image,obs.region)
    if source_descriptor is None or not np.array_equal(source_descriptor,obs.descriptor):
        return dict(v,reason='actual_source_crop_descriptor_mismatch')
    projection=project_anchor(target,cam,config)
    if not projection.valid:return dict(v,reason='current_projection_unavailable')
    a,b=gray_image(old),gray_image(cam);mask=np.zeros(a.shape,np.uint8);mask[region_slices(obs.region,a.shape)]=255
    p=cv2.goodFeaturesToTrack(a,maxCorners=96,qualityLevel=.01,minDistance=4,mask=mask,blockSize=5)
    if p is None:return dict(v,reason='RGB_tracks_missing')
    p=p[:,0];p=p[_inside(p,obs.region)]
    from .correspondence_recovery import retain_depth_points
    p,depth_seed_diagnostics=retain_depth_points(p,depth_history,old,config,obs.region)
    v['depth_feature_seeding']=depth_seed_diagnostics
    motion=None
    if config.ego_motion_mode != 'off':
        from .ego_motion import track_with_prediction
        if config.multi_hypothesis_tracking:
            from .search_motion import image_search_initial
            initial=image_search_initial(anchor,old,cam,p,search_motion,config)
            q,fb,valid,motion=track_with_prediction(old,cam,p,config,depth_history,static_geometry=False,search_initial=initial)
        else:q,fb,valid,motion=track_with_prediction(old,cam,p,config,depth_history)
        valid&=(q[:,0]>=3)&(q[:,0]<b.shape[1]-3)&(q[:,1]>=3)&(q[:,1]<b.shape[0]-3)
    else:q,fb,valid=_lk(a,b,p)
    if not len(p):return dict(v,reason='RGB_tracks_missing')
    _,_,photo,residual=measured_patch_support(a,b,p,q,valid,config)
    inside=_inside(q,projection.region);measured=photo&valid&(residual<=.04);selected=measured&inside
    n=int(selected.sum());total=len(p)
    v.update(total_tracks=total,measured_tracks=int(measured.sum()),tracks=n,
        source_region=list(obs.region),current_region=list(projection.region),source_timestamp=old.timestamp,current_timestamp=cam.timestamp,
        source_coordinate_space=old.coordinate_space,current_coordinate_space=cam.coordinate_space)
    if motion is not None:
        from .state_types import json_value
        v['ego_motion']=json_value(motion)
    if config.identity_diagnostics:
        from .state_types import json_value
        v['track_diagnostics']=json_value(dict(source_points=p.tolist(),current_points=q.tolist(),
            fb_error_px=fb.tolist(),correspondence_valid=valid.tolist(),patch_supported=photo.tolist(),
            patch_residual=residual.tolist(),inside_current=inside.tolist(),selected=selected.tolist()))
    if n<4 or n/total<.8 or quadrants(p[selected],obs.region)<3 or quadrants(q[selected],projection.region)<3:
        contradictory=(int(measured.sum())>=4 and float((measured&~inside).sum())/total>=.8)
        return dict(v,status='contradicted' if contradictory else 'unknown',reason='tracks_exclude_target' if contradictory else 'crop_track_or_spatial_support_insufficient',
            exclusion_source_points=p[measured].tolist(),exclusion_current_points=q[measured].tolist(),
            exclusion_fb_error_px=fb[measured].tolist(),exclusion_patch_residual=residual[measured].tolist())
    pp,qq=p[selected],q[selected];design=np.c_[pp,np.ones(n)]
    matrix,_,rank,_=np.linalg.lstsq(design,qq,rcond=None);matrix=matrix.T
    errors=np.linalg.norm(design@matrix.T-qq,axis=1)
    if rank<3 or not np.isfinite(matrix).all() or (errors>.75).any():
        return dict(v,reason='incoherent_local_motion')
    determinant=float(np.linalg.det(matrix[:,:2]))
    if not .5<=determinant<=2.:return dict(v,reason='folded_or_compressed_motion')
    descriptor=_aligned_descriptor(b,matrix,obs.region,validator)
    appearance=validator.residual(obs.descriptor,descriptor)
    if not np.isfinite(appearance) or appearance>.18:
        return dict(v,status='contradicted',reason='motion_aligned_appearance_conflict')
    return dict(v,status='supported',reason='measured_motion_observed_crop',
        source_points=pp.tolist(),current_points=qq.tolist(),fb_error_px=fb[selected].tolist(),
        patch_residual=residual[selected].tolist(),motion_matrix=matrix.tolist(),affine_residual_px=errors.tolist(),
        source_descriptor=obs.descriptor.tolist(),current_descriptor=descriptor.tolist(),descriptor_residual=appearance,
        source_quadrants=quadrants(pp,obs.region),current_quadrants=quadrants(qq,projection.region),
        appearance_mode='motion_aligned_source_crop')


def _cycle(view,source_record,earlier_record,frame,aid):
    from .temporal_visibility import gray_image
    cid=view['camera'];old=earlier_record['cameras'].get(cid);source=source_record['cameras'].get(cid)
    earlier=earlier_record['anchors'].get(aid)
    if old is None or source is None or earlier is None or cid not in earlier.observation_support:return None
    if (camera_snapshot_digest(old)!=earlier_record['camera_sha256'].get(cid)
            or camera_snapshot_digest(source)!=source_record['camera_sha256'].get(cid)):return None
    if (old.camera_id!=cid or old.coordinate_space!=source.coordinate_space
            or not np.isfinite([old.timestamp,source.timestamp,frame.cameras[cid].timestamp]).all()
            or not 0<source.timestamp-old.timestamp<=1 or not 0<frame.cameras[cid].timestamp-old.timestamp<=1
            or old.image.shape[:2]!=source.image.shape[:2] or not old.timing_reliable):return None
    p=np.asarray(view['source_points'],np.float32);q=np.asarray(view['current_points'],float)
    back,fb0,valid0=_lk(gray_image(source),gray_image(old),p)
    long,fb1,valid1=_lk(gray_image(old),gray_image(frame.cameras[cid]),back)
    error=np.linalg.norm(long-q,axis=1)
    good=valid0&valid1&_inside(back,earlier.observation_support[cid].region)&(error<=.75)
    if good.sum()<4 or good.mean()<.8 or quadrants(p[good],view['source_region'])<3:return None
    return dict(older_frame_id=earlier_record['frame_id'],older_frame_digest=earlier_record['frame_digest'],
        older_timestamp=old.timestamp,older_region=list(earlier.observation_support[cid].region),
        source_points=p[good].tolist(),older_points=back[good].tolist(),current_points=q[good].tolist(),
        independently_tracked_current_points=long[good].tolist(),loop_error_px=error[good].tolist(),
        backward_fb_error_px=fb0[good].tolist(),long_fb_error_px=fb1[good].tolist())


def measure_pair(anchor,target,frame,history,config,validator,search_motion=None):
    reasons=[];record=next((r for r in reversed(history) if r['frame_id']==anchor.source_frame_id),None)
    m=dict(source_class=anchor.semantic_class,current_class=target.semantic_class,
        source_center=anchor.geometry_world.center.tolist(),current_center=target.geometry_world.center.tolist(),
        displacement_m=float(np.linalg.norm(target.geometry_world.center-anchor.geometry_world.center)),views=[],dependency=DEPENDENCY)
    if config.identity_method=='temporal_object':
        from .identity_object import DEPENDENCY as OBJECT_DEPENDENCY
        m['dependency']=OBJECT_DEPENDENCY
    m['descriptor_config']={k:getattr(config,k) for k in ('descriptor','descriptor_size','min_texture_std','min_region_area_px')}
    m['producer_config']=config.to_dict()
    if (record is None or record['scene_id']!=frame.scene_id or record['epoch']!=anchor.source_state_epoch
            or record['execution_signature']!=frame.execution_signature or not record['frame_digest']):
        reasons.append('actual_source_frame_unavailable_or_execution_mismatch')
    age=frame.timestamp-anchor.source_timestamp
    if not np.isfinite(age) or not 0<age<=min(config.max_state_age_s,1.):reasons.append('stale_or_future_actual_source')
    if record is not None:
        saved=record['anchors'].get(anchor.anchor_id)
        if (saved is None or state_geometry_version(saved)!=state_geometry_version(anchor)
                or any(cid not in saved.observation_support or obs.region!=saved.observation_support[cid].region
                    or not np.array_equal(obs.descriptor,saved.observation_support[cid].descriptor)
                    or obs.timestamp!=saved.observation_support[cid].timestamp for cid,obs in anchor.observation_support.items())):
            reasons.append('actual_anchor_observation_binding_mismatch')
        if any(camera_snapshot_digest(c)!=record['camera_sha256'].get(cid) for cid,c in record['cameras'].items()):
            reasons.append('actual_source_camera_snapshot_mismatch')
        m['source_camera_sha256']=dict(record['camera_sha256'])
    if not reasons:
        for cid,cam in frame.cameras.items():
            old=record['cameras'].get(cid)
            if old is not None:
                depth=record.get('depth_history',{}).get(cid)
                if depth is not None and any(depth.get(k)!=record.get(k) for k in
                        ('scene_id','frame_id','frame_digest','epoch','execution_signature','commit_version')):
                    reasons.append('actual_depth_history_binding_mismatch');continue
                view=_measure_view(anchor,target,old,cam,validator,config,depth,search_motion)
                if (config.identity_method=='temporal_object' and config.identity_object_support
                        and ((config.identity_object_required and view['status']=='supported') or (view['status']=='unknown'
                            and view['reason'] in ('crop_track_or_spatial_support_insufficient','incoherent_local_motion','folded_or_compressed_motion')))):
                    from .identity_object import measure_object_view
                    view=measure_object_view(anchor,target,old,cam,validator,config,view)
                m['views'].append(view)
        if not any(v['status']=='supported' for v in m['views']):reasons.append('measured_motion_identity_observation_missing')
        if any(v['status']=='contradicted' for v in m['views']):reasons.append('observable_camera_conflict')
    return dict(status='candidate' if not reasons else 'unknown',rejection_reasons=reasons,measurements=m),record


def temporal_identity_pairs(anchors,predictions,costs,frame,history,config,validator,issuer,
                            legacy_evidence=None,legacy_frame_id=None,legacy_digest=None,search_hypotheses=None):
    tick=perf_counter();proposals={};records={}
    for i,j in zip(*np.nonzero(np.isfinite(costs))):
        anchor=anchors[i];pred=predictions[j]
        if anchor.anchor_type!='object':continue
        target=WorldAnchor(anchor.anchor_id,frame.scene_id,pred.anchor_type,pred.semantic_class,pred.geometry_world,frame.timestamp,
            velocity_world=pred.velocity_world,velocity_reliable=pred.velocity_reliable,geometry_uncertainty=pred.geometry_uncertainty,
            source_frame_id=frame.frame_id,source_state_epoch=anchor.source_state_epoch,source_execution_signature=frame.execution_signature)
        available=history if config.identity_use_history else history[-1:]
        proposal,record=measure_pair(anchor,target,frame,available,config,validator,(search_hypotheses or {}).get(i))
        proposal['acceptance_path']='motion_observed'
        # Preserve the existing independently testable selective population as
        # an alternative two-view producer, with its original receipt/validator.
        # No thresholds or evidence fields are translated into the motion claim.
        binding_failures={'source_time_camera_or_coordinate_mismatch','actual_source_crop_descriptor_mismatch'}
        if (proposal['status']=='unknown' and proposal['rejection_reasons']==['measured_motion_identity_observation_missing']
                and not any(v['reason'] in binding_failures for v in proposal['measurements']['views'])
                and not np.isfinite(np.delete(costs[i],j)).any() and not np.isfinite(np.delete(costs[:,j],i)).any()
                and legacy_evidence is not None and not config.identity_object_required):
            from .identity_continuity import propose_identity_continuity
            fallback=propose_identity_continuity(anchor,target,frame,legacy_evidence,config,validator,issuer,
                legacy_digest,legacy_frame_id)
            if fallback['status']=='candidate':proposal=dict(fallback,acceptance_path='selective_two_view_retained')
        proposals[int(i),int(j)]=proposal;records[int(i),int(j)]=(record,target)
    for (i,j),proposal in proposals.items():
        competitors=[(a,b) for a,b in proposals if (a==i or b==j) and (a,b)!=(i,j)]
        proposal.update(evidence=proposal.get('evidence'),recovered_short_gap=bool(history and anchors[i].source_frame_id!=history[-1]['frame_id']),
            resolved_competition=False,source_frame_digest=records[i,j][0]['frame_digest'] if records[i,j][0] else None)
        proposal['measurements']['competing_edges']=[dict(source_anchor_id=anchors[a].anchor_id,current_prediction_index=b,
            measured_status=proposals[a,b]['status'],reasons=list(proposals[a,b]['rejection_reasons']),
            positive_exclusion=any(v['reason']=='tracks_exclude_target' for v in proposals[a,b]['measurements']['views']),
            exclusion_views=[v for v in proposals[a,b]['measurements']['views'] if v['reason']=='tracks_exclude_target']) for a,b in competitors]
        if proposal['status']!='candidate':continue
        reasons=proposal['rejection_reasons'];record,target=records[i,j]
        if competitors:
            if not config.identity_resolve_competition:reasons.append('competition_resolution_disabled')
            # A competitor failing to provide tracks is not positive exclusion.
            if any(not any(v['reason']=='tracks_exclude_target' for v in proposals[a,b]['measurements']['views']) for a,b in competitors):
                reasons.append('unresolved_measured_identity_competition')
            if not reasons:
                earlier=[r for r in history if r['timestamp']<record['timestamp'] and r['scene_id']==frame.scene_id
                         and r['epoch']==record['epoch'] and r['execution_signature']==frame.execution_signature] if config.identity_use_history else []
                cycles=[]
                for v in proposal['measurements']['views']:
                    if v['status']!='supported':continue
                    cycle=next((c for old in reversed(earlier) if (c:=_cycle(v,record,old,frame,anchors[i].anchor_id)) is not None),None)
                    if cycle is not None:cycles.append(dict(camera=v['camera'],**cycle))
                proposal['measurements']['temporal_cycles']=cycles
                if not cycles:reasons.append('competition_temporal_support_missing')
                else:proposal['resolved_competition']=True
        proposal['measurements']['resolved_competition']=proposal['resolved_competition']
        if reasons:proposal['status']='unknown';continue
        if proposal['acceptance_path']=='selective_two_view_retained':continue
        producer=PRODUCER
        if config.identity_method=='temporal_object':
            from .identity_object import PRODUCER as producer
        evidence=PhysicalEvidence(anchors[i].anchor_id,anchors[i].source_frame_id,frame.frame_id,frame.scene_id,
            anchors[i].source_state_epoch,state_geometry_version(anchors[i]),state_geometry_version(target),
            record['frame_digest'],frame_digest(frame),frame.execution_signature,frame.timestamp,producer,
            json.dumps(proposal['measurements'],sort_keys=True,allow_nan=False))
        proposal['evidence']=issuer._mint_measured(evidence)
    elapsed=(perf_counter()-tick)*1000
    for p in proposals.values():p['producer_cpu_ms']=elapsed/max(len(proposals),1);p['claim']='object_identity_continuity'
    return proposals


def certify_temporal_identity(anchor,current,frame,evidence,issuer,source_digest,protocol_digest):
    from .identity_continuity import IDENTITY_CLAIM
    reasons=[];provenance=dict(identity_only=True,stationarity_certified=False,geometry_attributes_certified=False,
        motion_attributes_certified=False,foreground_ownership_certified=False,claim_scope='Same physical entity between actual source/current observations only')
    if not isinstance(evidence,PhysicalEvidence) or issuer is None or not issuer.authentic(evidence):
        reasons.append('trusted_measured_identity_evidence_missing')
    else:
        try:
            m=json.loads(evidence.measurements_json);reject_oracle_metadata(m)
            from .identity_object import PRODUCER as OBJECT_PRODUCER,DEPENDENCY as OBJECT_DEPENDENCY,POPULATION as OBJECT_POPULATION
            object_method=evidence.producer_id==OBJECT_PRODUCER
            if (evidence.producer_id not in (PRODUCER,OBJECT_PRODUCER)
                    or m['dependency']!=(OBJECT_DEPENDENCY if object_method else DEPENDENCY)):
                reasons.append('identity_producer_or_dependency_mismatch')
            if (anchor.anchor_id!=current.anchor_id or evidence.anchor_id!=anchor.anchor_id
                    or anchor.scene_id!=frame.scene_id or current.scene_id!=frame.scene_id or evidence.scene_id!=frame.scene_id):
                reasons.append('anchor_or_scene_binding_mismatch')
            if (not source_digest or evidence.source_frame_digest!=source_digest or evidence.source_frame_id!=anchor.source_frame_id
                    or evidence.current_frame_id!=frame.frame_id or evidence.source_frame_id==frame.frame_id
                    or evidence.current_frame_digest!=frame_digest(frame)):reasons.append('source_or_current_frame_mismatch')
            if (type(evidence.source_epoch) is not int or evidence.source_epoch<1 or evidence.source_epoch!=anchor.source_state_epoch
                    or current.source_state_epoch!=anchor.source_state_epoch):reasons.append('source_epoch_mismatch')
            if (evidence.source_state_version!=state_geometry_version(anchor) or evidence.current_state_version!=state_geometry_version(current)
                    or current.source_frame_id!=frame.frame_id or current.source_timestamp!=frame.timestamp):reasons.append('state_version_mismatch')
            if (evidence.execution_signature!=frame.execution_signature or anchor.source_execution_signature!=frame.execution_signature
                    or current.source_execution_signature!=frame.execution_signature):reasons.append('execution_mismatch')
            age=frame.timestamp-anchor.source_timestamp
            if (not np.isfinite(age) or not 0<age<=1 or evidence.evidence_timestamp!=frame.timestamp):reasons.append('stale_or_future_identity_evidence')
            if (anchor.anchor_type!='object' or current.anchor_type!='object' or anchor.semantic_class!=current.semantic_class
                    or m['source_class']!=anchor.semantic_class or m['current_class']!=current.semantic_class
                    or not np.array_equal(m['source_center'],anchor.geometry_world.center)
                    or not np.array_equal(m['current_center'],current.geometry_world.center)
                    or not np.isfinite(m['displacement_m']) or not np.isclose(m['displacement_m'],np.linalg.norm(current.geometry_world.center-anchor.geometry_world.center))):
                reasons.append('object_measurement_binding_mismatch')
            from .config import M1Config
            from .observation_validator import ObservationValidator
            # Descriptor representation/size is bound explicitly, then rebuilt
            # from current pixels. No asserted residual or quadrant is trusted.
            representation=m.get('descriptor_config',{})
            producer_config=M1Config(**m['producer_config'])
            from .identity_continuity import identity_protocol_digest
            if (identity_protocol_digest(producer_config,frame)!=protocol_digest
                    or representation!={k:getattr(producer_config,k) for k in ('descriptor','descriptor_size','min_texture_std','min_region_area_px')}):
                reasons.append('measured_protocol_config_mismatch')
            validator=ObservationValidator(producer_config)
            supported=[]
            for v in m['views']:
                if v['status']=='contradicted':reasons.append('observable_camera_conflict')
                if v['status']!='supported':continue
                cid=v['camera'];cam=frame.cameras[cid];obs=anchor.observation_support[cid]
                p=np.asarray(v['source_points'],float);q=np.asarray(v['current_points'],float);n=len(p)
                fb=np.asarray(v['fb_error_px'],float);photo=np.asarray(v['patch_residual'],float);matrix=np.asarray(v.get('motion_matrix',np.eye(2,3)),float)
                object_view=v.get('observation_model')=='translation_compensated_box_surface_ring'
                r=np.asarray(v['source_region'],float);s=np.asarray(v['current_region'],float)
                if (p.shape!=(n,2) or q.shape!=p.shape or n<4 or fb.shape!=(n,) or photo.shape!=(n,) or matrix.shape!=(2,3)
                        or r.shape!=(4,) or s.shape!=(4,) or not all(np.isfinite(a).all() for a in (p,q,fb,photo,matrix,r,s))
                        or (fb<0).any() or (fb>.75).any() or (photo<0).any() or (photo>.04).any()
                        or len(np.unique(p,axis=0))!=n or len(np.unique(q,axis=0))!=n
                        or type(v['total_tracks']) is not int or v['total_tracks']<n or (not object_view and n/v['total_tracks']<.8)):
                    reasons.append('invalid_measured_motion_tracks');continue
                if (not np.array_equal(r,obs.region) or not np.array_equal(v['source_descriptor'],obs.descriptor)
                        or obs.camera_id!=cid or cam.camera_id!=cid or v['source_timestamp']!=obs.timestamp
                        or not np.isfinite([obs.timestamp,cam.timestamp]).all() or v['current_timestamp']!=cam.timestamp
                        or not cam.timing_reliable or v['source_coordinate_space']!=cam.coordinate_space
                        or v['current_coordinate_space']!=cam.coordinate_space or (r[2:]<=r[:2]).any() or (s[2:]<=s[:2]).any()
                        or not _inside(p,r).all() or not _inside(q,s).all() or quadrants(p,r)<3 or quadrants(q,s)<3):
                    reasons.append('camera_crop_or_spatial_binding_mismatch')
                projection=project_anchor(current,cam,producer_config)
                if not projection.valid or not np.array_equal(s,projection.region):reasons.append('current_crop_geometry_binding_mismatch')
                if object_view:
                    from .identity_object import validate_object_view
                    if (not object_method or producer_config.identity_method!='temporal_object'
                            or not validate_object_view(anchor,current,cam,v,producer_config,m)):
                        reasons.append('object_surface_ring_measurements_invalid')
                    supported.append(v);continue
                if (not .5<=np.linalg.det(matrix[:,:2])<=2 or (np.linalg.norm(np.c_[p,np.ones(n)]@matrix.T-q,axis=1)>.75).any()):
                    reasons.append('motion_model_not_supported')
                from .temporal_visibility import gray_image
                descriptor=_aligned_descriptor(gray_image(cam),matrix,r,validator)
                if (v.get('appearance_mode')!='motion_aligned_source_crop' or descriptor is None
                        or not np.allclose(descriptor,v['current_descriptor'],atol=1e-5)
                        or validator.residual(obs.descriptor,descriptor)>.18):reasons.append('motion_aligned_appearance_not_supported')
                supported.append(v)
            if not supported:reasons.append('identity_observation_missing')
            if m['competing_edges']:
                if not m.get('resolved_competition') or not m.get('temporal_cycles'):reasons.append('competition_not_resolved')
                for edge in m['competing_edges']:
                    if edge.get('measured_status')!='unknown' or not edge.get('positive_exclusion'):reasons.append('competitor_not_positively_excluded')
                    excluded=False
                    for v in edge.get('exclusion_views',[]):
                        p=np.asarray(v['exclusion_source_points'],float);q=np.asarray(v['exclusion_current_points'],float)
                        fb=np.asarray(v['exclusion_fb_error_px'],float);photo=np.asarray(v['exclusion_patch_residual'],float);n=len(p)
                        if (n>=4 and p.shape==(n,2) and q.shape==p.shape and fb.shape==(n,) and photo.shape==(n,)
                                and all(np.isfinite(a).all() for a in (p,q,fb,photo)) and (fb>=0).all() and (fb<=.75).all()
                                and (photo>=0).all() and (photo<=.04).all() and _inside(p,v['source_region']).all()
                                and type(v['total_tracks']) is int and v['total_tracks']>=n
                                and int((~_inside(q,v['current_region'])).sum())/v['total_tracks']>=.8
                                and len(np.unique(p,axis=0))==n and len(np.unique(q,axis=0))==n):excluded=True
                    if not excluded:reasons.append('competitor_exclusion_measurements_invalid')
                for c in m.get('temporal_cycles',[]):
                    p=np.asarray(c['source_points'],float);q=np.asarray(c['current_points'],float);long=np.asarray(c['independently_tracked_current_points'],float)
                    b=np.asarray(c['older_points'],float);n=len(p)
                    arrays=[np.asarray(c[k],float) for k in ('loop_error_px','backward_fb_error_px','long_fb_error_px')]
                    view=next((v for v in supported if v['camera']==c['camera']),None)
                    bound=bool(view is not None and all(any(np.allclose(pp,sp,atol=1e-5) and np.allclose(qq,sq,atol=1e-5)
                        for sp,sq in zip(view['source_points'],view['current_points'])) for pp,qq in zip(p,q)))
                    if (n<4 or any(a.shape!=(n,2) or not np.isfinite(a).all() for a in (p,q,long,b))
                            or any(a.shape!=(n,) or not np.isfinite(a).all() or (a<0).any() or (a>.75).any() for a in arrays)
                            or (np.linalg.norm(long-q,axis=1)>.75).any() or not _inside(b,c['older_region']).all()
                            or not np.isfinite(c['older_timestamp']) or view is None
                            or not 0<view['source_timestamp']-c['older_timestamp']<=1
                            or not 0<view['current_timestamp']-c['older_timestamp']<=1 or not c['older_frame_digest'] or not bound
                            or len(np.unique(p,axis=0))!=n or not np.allclose(arrays[0],np.linalg.norm(long-q,axis=1),atol=1e-5)):
                        reasons.append('temporal_loop_measurements_invalid')
            reasons.extend(issuer.qualification_reasons(evidence,frame,protocol_digest,IDENTITY_CLAIM,OBJECT_POPULATION if object_method else POPULATION))
            provenance.update(producer=evidence.producer_id,protocol_digest=protocol_digest,receipt=evidence.authentication_tag,
                source_frame_digest=evidence.source_frame_digest,current_frame_digest=evidence.current_frame_digest,
                execution_signature=evidence.execution_signature,risk_qualification_sha256=issuer.qualification_sha256,scene_level_risk=issuer.qualification_risk(),
                supported_views=len(supported),same_sensor_RGB=True,independent_sensor=False)
        except (KeyError,ValueError,TypeError,IndexError,OverflowError,np.linalg.LinAlgError,cv2.error):
            reasons.append('malformed_or_forbidden_identity_evidence')
    return PhysicalCertificationResult(not reasons,IDENTITY_CLAIM if not reasons else None,tuple(sorted(set(reasons))),provenance,
        evidence.source_state_version if isinstance(evidence,PhysicalEvidence) else None,
        evidence.current_state_version if isinstance(evidence,PhysicalEvidence) else None,anchor.source_frame_id,frame.frame_id,
        anchor.anchor_id,frame.scene_id,frame.timestamp,anchor.source_state_epoch)
