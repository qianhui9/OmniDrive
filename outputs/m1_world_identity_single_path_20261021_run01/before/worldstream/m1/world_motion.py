"""Conservative world motion from committed, qualified and calibrated centres.

This module never elevates detector velocity or geometric track IDs to evidence.
Identity qualification alone does not calibrate detector localization error.
"""
import copy
import numpy as np
from .geometry import state_geometry_version


def update_world_motion(prior, current, config):
    current.motion_history=[]
    current.motion_prediction=dict(reliable=False,reason='qualified_identity_history_unavailable')
    if prior is None:return
    certificate=current.physical_certification
    from .identity_continuity import IDENTITY_CLAIM
    if (not current.identity_link_qualified or certificate is None or not certificate.certified
            or certificate.certified_claim!=IDENTITY_CLAIM
            or certificate.source_state_version!=state_geometry_version(prior)
            or certificate.current_state_version!=state_geometry_version(current)
            or certificate.source_frame_id!=prior.source_frame_id
            or certificate.current_frame_id!=current.source_frame_id
            or not certificate.evidence_provenance.get('risk_qualification_sha256')):
        return
    if (prior.scene_id!=current.scene_id or prior.source_state_epoch!=current.source_state_epoch
            or prior.source_execution_signature!=current.source_execution_signature):return
    if not prior.geometry_uncertainty_calibrated or not current.geometry_uncertainty_calibrated:
        current.motion_prediction['reason']='world_position_uncertainty_uncalibrated'
        return
    dt=current.source_timestamp-prior.source_timestamp
    if not np.isfinite(dt) or not 0<dt<=config.max_frame_gap_s:
        current.motion_prediction['reason']='motion_time_gap_invalid';return
    def sample(a):
        return dict(frame_id=a.source_frame_id,timestamp=a.source_timestamp,center=a.geometry_world.center.tolist(),
            sigma_m=a.geometry_uncertainty,state_version=state_geometry_version(a),epoch=a.source_state_epoch,
            execution_signature=a.source_execution_signature,scene_id=a.scene_id)
    history=copy.deepcopy(prior.motion_history) or [sample(prior)]
    if history[-1]!=sample(prior):
        current.motion_prediction['reason']='motion_source_version_mismatch';return
    history.append(sample(current));history=history[-config.association_history_limit:]
    current.motion_history=history
    if len(history)<3:
        current.motion_prediction['reason']='motion_history_insufficient';return
    times=np.asarray([r['timestamp'] for r in history]);steps=np.diff(times)
    if (not np.isfinite(steps).all() or (steps<=0).any() or (steps>config.max_frame_gap_s).any()
            or len({r['frame_id'] for r in history})!=len(history)):
        current.motion_prediction['reason']='motion_history_time_invalid';return
    centers=np.asarray([r['center'] for r in history]);sigma=np.asarray([r['sigma_m'] for r in history])
    if not np.isfinite(centers).all() or not np.isfinite(sigma).all() or (sigma<0).any():return
    velocities=np.diff(centers,axis=0)/steps[:,None]
    variances=(sigma[1:]**2+sigma[:-1]**2)/steps**2
    weights=1/np.maximum(variances,1e-8)
    velocity=np.average(velocities,axis=0,weights=weights)
    # Turning/acceleration expands prediction uncertainty instead of broadening
    # the fixed association gate. Uncertain trajectories cannot become reliable.
    scatter=(velocities-velocity).T@(velocities-velocity)/len(velocities)
    covariance=scatter+np.eye(3)*max(float(np.max(variances)),1e-8)
    reliable=bool(np.sqrt(np.linalg.eigvalsh(covariance).max())<=config.velocity_uncertainty_mps)
    current.motion_prediction=dict(reliable=reliable,reason='qualified_calibrated_motion' if reliable else 'motion_direction_or_speed_uncertain',
        velocity_world=velocity.tolist(),velocity_covariance=covariance.tolist(),
        source_state_version=state_geometry_version(current),source_timestamp=current.source_timestamp,
        position_sigma_m=current.geometry_uncertainty,history_frames=len(history),
        localization_calibrated=True,identity_qualification_sha256=certificate.evidence_provenance['risk_qualification_sha256'])


def prediction(anchor, timestamp):
    m=anchor.motion_prediction
    dt=timestamp-anchor.source_timestamp
    if (not m.get('reliable') or not 0<=dt<=1.
            or m.get('source_state_version')!=state_geometry_version(anchor)):
        return None
    v=np.asarray(m.get('velocity_world'));cov=np.asarray(m.get('velocity_covariance'))
    if (v.shape!=(3,) or cov.shape!=(3,3) or not np.isfinite(v).all() or not np.isfinite(cov).all()
            or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<0):return None
    covariance=cov*dt**2+np.eye(3)*m['position_sigma_m']**2
    return anchor.geometry_world.center+v*dt,covariance
