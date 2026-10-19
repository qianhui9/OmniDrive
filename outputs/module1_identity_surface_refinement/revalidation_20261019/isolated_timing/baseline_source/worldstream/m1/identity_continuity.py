"""Selective identity proposal. No stationarity, foreground or metric claim.

Isolation + measured crop/track support define an empirically testable population.
They do NOT establish true identity without independent producer qualification.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .geometry import project_anchor, state_geometry_version
from .input_contract import frame_digest, reject_oracle_metadata
from .state_types import PhysicalEvidence, PhysicalCertificationResult

IDENTITY_CLAIM='object_identity_continuity'
IDENTITY_PRODUCER='isolated_observed_identity_pairs_v1'
IDENTITY_POPULATION='isolated_observed_identity_pairs'
MAX_DISPLACEMENT_M=.5


def identity_protocol_digest(config, frame):
    excluded={'physical_qualification_path','physical_qualification_sha256','logging','log_path','profiling',
              'visualization','capture_tensors','capture_backbone','storage_device','max_tensor_bytes','temporal_workers'}
    # Bind transitive measurement/selection implementations, not just a version label.
    files=['identity_continuity.py','association.py','geometry.py','temporal_visibility.py','geometry_support.py',
           'observation_validator.py','physical_certifier.py','state_manager.py']
    code={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}
    value=dict(producer=IDENTITY_PRODUCER,code=code,max_displacement_m=MAX_DISPLACEMENT_M,
        config={k:v for k,v in config.to_dict().items() if k not in excluded},
        cameras={cid:[c.coordinate_space,list(c.image.shape)] for cid,c in frame.cameras.items()})
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def propose_identity_continuity(anchor,current_anchor,frame,evidence_by_camera,config,validator,
        issuer,source_frame_digest,actual_source_frame_id,competing_costs=()):
    tick=perf_counter();reasons=[];views=[]
    if (anchor.anchor_type!='object' or current_anchor.anchor_type!='object'
            or anchor.semantic_class!=current_anchor.semantic_class):reasons.append('incompatible_object_class')
    if anchor.source_frame_id!=actual_source_frame_id or not source_frame_digest:
        reasons.append('actual_source_frame_unavailable')
    age=frame.timestamp-anchor.source_timestamp
    if not np.isfinite(age) or not 0<age<=config.max_state_age_s:reasons.append('missing_frame_or_stale_source')
    displacement=float(np.linalg.norm(current_anchor.geometry_world.center-anchor.geometry_world.center))
    if not np.isfinite(displacement) or displacement>MAX_DISPLACEMENT_M:reasons.append('outside_selective_displacement_population')
    alternatives=[float(v) for v in competing_costs if np.isfinite(v)]
    if alternatives:reasons.append('unresolved_geometric_identity_competition')
    # Cheap failures prove violation of this population's necessary conditions.
    if not reasons:
        for cid,cam in frame.cameras.items():
            old=anchor.observation_support.get(cid);e=evidence_by_camera.get(cid)
            if old is None or e is None:continue
            projection=project_anchor(current_anchor,cam,config)
            if not projection.valid:continue
            view=dict(camera=cid,reason='temporal_source_mismatch');views.append(view)
            if (e.camera_id!=cid or abs(e.source_timestamp-old.timestamp)>1e-5
                    or e.timestamp!=cam.timestamp or not cam.timing_reliable):continue
            p=e.source_points;q=e.current_points;r=np.asarray(old.region);s=np.asarray(projection.region)
            inside=(p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])
            inside&=(q[:,0]>=s[0])&(q[:,0]<s[2])&(q[:,1]>=s[1])&(q[:,1]<s[3])
            raw=e.raw_correspondence_supported
            if raw is None:raw=e.supported
            selected=inside&raw&e.correspondence_valid&e.geometry_valid
            count=int(selected.sum());total=int(inside.sum())
            size=np.maximum(r[2:]-r[:2],1)
            quadrants=len(np.unique(np.clip(((p[selected]-r[:2])/size*2).astype(int),0,1),axis=0)) if count else 0
            descriptor=validator.descriptor(cam.image,projection.region)
            residual=validator.residual(old.descriptor,descriptor)
            view.update(source_points=p[selected].tolist(),current_points=q[selected].tolist(),
                fb_error_px=e.fb_error_px[selected].tolist(),source_region=r.tolist(),current_region=s.tolist(),
                source_timestamp=old.timestamp,current_timestamp=cam.timestamp,
                source_descriptor=np.asarray(old.descriptor).tolist(),current_descriptor=None if descriptor is None else descriptor.tolist(),
                descriptor_residual=residual,tracks=count,total_tracks=total,quadrants=quadrants)
            passed=(count>=config.support_min_points and total>0 and count/total>=config.support_point_fraction
                    and quadrants/4>=config.support_quadrant_fraction and residual<=config.residual_threshold)
            view['reason']='measured_identity_population_candidate' if passed else 'observable_crop_or_tracks_insufficient'
        if not any(v['reason']=='measured_identity_population_candidate' for v in views):
            reasons.append('measured_identity_observation_missing')
        if any(v['reason']!='measured_identity_population_candidate' for v in views):
            reasons.append('common_view_observation_conflict_or_unknown')
    evidence=None
    measurements=dict(source_class=anchor.semantic_class,current_class=current_anchor.semantic_class,
        source_center=anchor.geometry_world.center.tolist(),current_center=current_anchor.geometry_world.center.tolist(),
        displacement_m=displacement,competing_legal_costs=alternatives,views=views,
        dependency='Actual_detection_pair_and_same_RGB_tracks_not_instance_mask')
    if not reasons:
        evidence=issuer._mint_measured(PhysicalEvidence(anchor.anchor_id,anchor.source_frame_id,frame.frame_id,
            frame.scene_id,anchor.source_state_epoch,state_geometry_version(anchor),state_geometry_version(current_anchor),
            source_frame_digest,frame_digest(frame),frame.execution_signature,frame.timestamp,IDENTITY_PRODUCER,
            json.dumps(measurements,sort_keys=True,allow_nan=False)))
    return dict(status='candidate' if not reasons else 'unknown',rejection_reasons=reasons,evidence=evidence,
        measurements=measurements,producer_cpu_ms=(perf_counter()-tick)*1000,
        # Protocol construction is measured separately by the caller once/frame.
        claim=IDENTITY_CLAIM)


def certify_identity_evidence(anchor,current_anchor,frame,evidence,issuer,source_digest,protocol_digest):
    reasons=[];provenance=dict(identity_only=True,stationarity_certified=False,geometry_attributes_certified=False,
        motion_attributes_certified=False,foreground_ownership_certified=False,same_sensor_RGB=True,independent_sensor=False)
    if not isinstance(evidence,PhysicalEvidence) or issuer is None or not issuer.authentic(evidence):
        reasons.append('trusted_measured_identity_evidence_missing')
    else:
        try:
            m=json.loads(evidence.measurements_json);reject_oracle_metadata(m)
            if evidence.producer_id!=IDENTITY_PRODUCER:reasons.append('identity_producer_mismatch')
            if (anchor.anchor_id!=current_anchor.anchor_id or evidence.anchor_id!=anchor.anchor_id
                    or evidence.scene_id!=frame.scene_id or anchor.scene_id!=frame.scene_id or current_anchor.scene_id!=frame.scene_id):
                reasons.append('anchor_or_scene_binding_mismatch')
            if (not source_digest or evidence.source_frame_digest!=source_digest or evidence.source_frame_id!=anchor.source_frame_id
                    or evidence.current_frame_id!=frame.frame_id or evidence.source_frame_id==frame.frame_id):
                reasons.append('source_or_current_frame_mismatch')
            if (evidence.source_epoch!=anchor.source_state_epoch or current_anchor.source_state_epoch!=anchor.source_state_epoch
                    or not isinstance(evidence.source_epoch,int) or evidence.source_epoch<1):reasons.append('source_epoch_mismatch')
            if (evidence.source_state_version!=state_geometry_version(anchor) or evidence.current_state_version!=state_geometry_version(current_anchor)
                    or current_anchor.source_frame_id!=frame.frame_id or current_anchor.source_timestamp!=frame.timestamp):
                reasons.append('state_version_or_current_computation_mismatch')
            if (evidence.execution_signature!=frame.execution_signature or anchor.source_execution_signature!=frame.execution_signature
                    or current_anchor.source_execution_signature!=frame.execution_signature or evidence.current_frame_digest!=frame_digest(frame)):
                reasons.append('observation_or_execution_mismatch')
            if not 0<frame.timestamp-anchor.source_timestamp<=1 or evidence.evidence_timestamp!=frame.timestamp:
                reasons.append('stale_or_future_identity_evidence')
            if (anchor.anchor_type!='object' or current_anchor.anchor_type!='object' or anchor.semantic_class!=current_anchor.semantic_class
                    or m['source_class']!=anchor.semantic_class or m['current_class']!=current_anchor.semantic_class):
                reasons.append('incompatible_object_class')
            if (not np.array_equal(m['source_center'],anchor.geometry_world.center)
                    or not np.array_equal(m['current_center'],current_anchor.geometry_world.center)
                    or np.linalg.norm(current_anchor.geometry_world.center-anchor.geometry_world.center)>MAX_DISPLACEMENT_M):
                reasons.append('outside_selective_identity_population')
            if m['competing_legal_costs']:reasons.append('identity_competition')
            if m['dependency']!='Actual_detection_pair_and_same_RGB_tracks_not_instance_mask':reasons.append('circular_or_self_asserted_identity')
            supported=0
            for v in m['views']:
                p=np.asarray(v['source_points'],float);q=np.asarray(v['current_points'],float);fb=np.asarray(v['fb_error_px'],float)
                a=np.asarray(v['source_descriptor'],float);b=np.asarray(v['current_descriptor'],float)
                if (p.ndim!=2 or p.shape[1:]!=(2,) or q.shape!=p.shape or len(p)<4 or fb.shape!=(len(p),)
                        or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(fb).all() or (fb>.75).any()
                        or a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all()
                        or np.mean(np.abs(a-b))>.18 or v['quadrants']<3):
                    reasons.append('measured_identity_tracks_or_crop_insufficient');continue
                cid=v['camera'];cam=frame.cameras[cid];old=anchor.observation_support[cid]
                if (v['source_timestamp']!=old.timestamp or v['current_timestamp']!=cam.timestamp or not cam.timing_reliable
                        or not np.array_equal(a,old.descriptor)):
                    reasons.append('camera_or_source_observation_mismatch');continue
                for pts,region in ((p,v['source_region']),(q,v['current_region'])):
                    if not ((pts[:,0]>=region[0])&(pts[:,0]<region[2])&(pts[:,1]>=region[1])&(pts[:,1]<region[3])).all():
                        reasons.append('tracks_outside_claimed_crop')
                supported+=1
            if not supported:reasons.append('identity_observation_missing')
            reasons.extend(issuer.qualification_reasons(evidence,frame,protocol_digest,IDENTITY_CLAIM,IDENTITY_POPULATION))
            provenance.update(producer=IDENTITY_PRODUCER,protocol_digest=protocol_digest,receipt=evidence.authentication_tag,
                qualification='Pinned independently audited scene/log risk artifact required',supported_views=supported)
        except (ValueError,TypeError,KeyError,IndexError,OverflowError):reasons.append('malformed_or_forbidden_identity_evidence')
    return PhysicalCertificationResult(not reasons,IDENTITY_CLAIM if not reasons else None,tuple(sorted(set(reasons))),provenance,
        evidence.source_state_version if isinstance(evidence,PhysicalEvidence) else None,
        evidence.current_state_version if isinstance(evidence,PhysicalEvidence) else None,anchor.source_frame_id,frame.frame_id,
        anchor.anchor_id,frame.scene_id,frame.timestamp,anchor.source_state_epoch)
