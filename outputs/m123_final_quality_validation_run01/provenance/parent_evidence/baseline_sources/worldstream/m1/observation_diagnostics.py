"""Read-only necessary-condition diagnostics; no acceptance gate is weakened."""
import numpy as np
from .geometry import state_geometry_version


def attach_observation_funnel(manager, anchor, camera, projection, occluded, validation):
    cid=camera.camera_id;obs=anchor.observation_support.get(cid)
    old=manager._previous_cameras.get(cid) if anchor.source_frame_id==manager._previous_frame_id else None
    source_digest=manager._previous_frame_digest if old is not None else None
    e=manager.last_evidence.get(cid);checks={}
    def check(name,value,reason):
        value=None if value is None else bool(value)
        checks[name]=dict(passed=value,reason='ok' if value is True else reason)
    check('historical_frame_source',bool(old is not None and source_digest),'actual_source_frame_unavailable')
    check('source_epoch',anchor.source_state_epoch==manager.store.generation,'source_epoch_mismatch')
    check('state_version',anchor.state_version==state_geometry_version(anchor),'state_version_mismatch')
    check('execution_signature',anchor.source_execution_signature==manager._pending_execution_signature,'execution_signature_mismatch')
    check('historical_crop',obs is not None,'historical_crop_missing')
    reconstructed=None
    if old is not None and obs is not None:
        rebuilt=manager.validator.descriptor(old.image,obs.region)
        reconstructed=bool(rebuilt is not None and np.array_equal(rebuilt,obs.descriptor))
    check('source_descriptor_reconstruction',reconstructed,'source_descriptor_missing_or_mismatch')
    check('current_projection',bool(projection.valid),projection.reason)
    region=validation.region if len(validation.region)==4 else projection.region
    descriptor=manager.validator.descriptor(camera.image,region) if region is not None else None
    check('current_crop_descriptor',descriptor is not None,'low_texture_or_invalid_crop')
    check('descriptor_residual',bool(validation.observation_consistent),'observation_mismatch')
    check('camera_time_and_coordinate',None if old is None else bool(old.camera_id==cid and old.coordinate_space==camera.coordinate_space
        and old.timing_reliable and camera.timing_reliable and obs is not None and obs.timestamp==old.timestamp),'camera_time_or_coordinate_unknown')
    check('projected_occlusion',not occluded,'projected_occlusion')
    tracks={};measurements={}
    if e is not None and obs is not None and projection.region is not None:
        p=np.asarray(e.source_points);q=np.asarray(e.current_points);n=len(p)
        r=np.asarray(obs.region);s=np.asarray(projection.region)
        inside=((p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])&
                (q[:,0]>=s[0])&(q[:,0]<s[2])&(q[:,1]>=s[1])&(q[:,1]<s[3]))
        raw=e.raw_correspondence_supported if e.raw_correspondence_supported is not None else e.supported
        masks=dict(crop_correspondence=inside,
            bidirectional=inside&e.correspondence_valid&(e.fb_error_px<=manager.config.flow_fb_max_px),
            calibrated_geometry=inside&e.correspondence_valid&e.geometry_valid,
            local_photo_geometry=inside&e.correspondence_valid&e.geometry_valid&raw,
            after_motion_occlusion_exclusion=inside&e.supported)
        for k,mask in masks.items():tracks[k]=dict(total=n,passed=int(mask.sum()),rejected=int(n-mask.sum()))
        chosen=masks['local_photo_geometry'];total=int(inside.sum());count=int(chosen.sum())
        unique=bool(len(np.unique(p[chosen],axis=0))==count and len(np.unique(q[chosen],axis=0))==count) if count else None
        quadrants=len(np.unique(np.clip(((p[chosen]-r[:2])/np.maximum(r[2:]-r[:2],1)*2).astype(int),0,1),axis=0)) if count else 0
        check('track_uniqueness',unique,'tracks_missing_or_repeated')
        check('track_spatial_distribution',bool(count and quadrants/4>=manager.config.support_quadrant_fraction),'spatial_support_insufficient')
        check('track_support_fraction',bool(total and count/total>=manager.config.support_point_fraction),'local_track_support_insufficient')
        def stats(values):
            v=np.asarray(values);v=v[np.isfinite(v)]
            return dict(n=len(v),median=float(np.median(v)) if len(v) else None,p95=float(np.percentile(v,95)) if len(v) else None)
        measurements=dict(tracks=tracks,local_support_fraction=count/total if total else None,quadrants=quadrants,
            fb_error_px=stats(e.fb_error_px[inside]),patch_residual=stats(e.patch_residual[inside]),
            actual_source_timestamp=e.source_timestamp,current_camera_timestamp=e.timestamp)
        check('measurement_source_time',bool(old is not None and e.source_timestamp==old.timestamp and e.timestamp==camera.timestamp),
              'measured_track_source_mismatch')
    else:
        for k in ('track_uniqueness','track_spatial_distribution','track_support_fraction','measurement_source_time'):
            check(k,None,'tracks_or_source_crop_unavailable')
    validation.evidence.update(observation_funnel=checks,observation_measurements=measurements,
        source_crop_region=list(obs.region) if obs is not None else None,
        source_frame_id=anchor.source_frame_id,source_frame_digest=source_digest,
        source_state_version=anchor.state_version,source_epoch=anchor.source_state_epoch,
        source_execution_signature=anchor.source_execution_signature,
        finite_claims=dict(foreground_ownership='unknown',identity='unknown',stationarity='unqualified_class_prior_only',
                           metric_geometry='uncertified',neural_equivalence=False))
