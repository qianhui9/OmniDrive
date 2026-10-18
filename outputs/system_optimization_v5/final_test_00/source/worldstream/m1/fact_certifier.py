"""Fail-closed relative-direction research certifier (not a runtime policy).

Candidate extraction, online evidence and independent deployment validation are
separate contracts. A successful per-observation check is feasibility evidence;
it never issues a quality/cost certificate or enables M3 consumption.
"""
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Optional

import numpy as np

from .task_facts import TaskFact as FactCandidate


@dataclass(frozen=True)
class DirectionCalibration:
    calibration_id: str
    execution_signature: str
    calibration_scenes: tuple
    radius_xy_m: Optional[tuple]
    alpha: float = .05
    method: str = 'scene_max_split_conformal'
    source: str = 'offline_calibration_only'


@dataclass(frozen=True)
class FactCertification:
    certified: bool
    fact_type: str
    fact_value: object
    evidence_source: dict
    uncertainty_interval: Optional[dict]
    source_frame: Optional[str]
    current_frame: str
    identity_support: dict
    foreground_support: dict
    freshness: dict
    certification_latency: float  # milliseconds
    rejection_reason: str
    coverage_scope: str = 'statistical; not a deterministic per-target physical guarantee'
    deployment_approved: bool = False


def calibrate_scene_maxima(scene_errors, execution_signature, calibration_id, alpha=.05):
    """Conformal interval radius using scenes as exchangeability units.

    Calibration GT may be used OFFLINE to build this artifact. It is never an
    input to certify_relative_direction. A quantile above n returns an unbounded
    interval, rather than incorrectly clipping k to the largest calibration
    error. Coverage still requires exchangeable scenes and the same population
    and error/association protocol; independent risk validation is mandatory.
    """
    if not 0 < alpha < 1: raise ValueError('Invalid alpha')
    maxima=[]
    for scene, errors in sorted(scene_errors.items()):
        values=np.asarray(errors,dtype=float)
        if values.ndim!=2 or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Expected nonempty finite xy residuals per scene')
        maxima.append(float(np.abs(values).max()))
    rank=math.ceil((len(maxima)+1)*(1-alpha))
    radius=(float(sorted(maxima)[rank-1]),)*2 if rank<=len(maxima) else None
    return DirectionCalibration(calibration_id,execution_signature,tuple(sorted(scene_errors)),radius,alpha)


def _forbidden(value):
    if isinstance(value, dict):
        return any(str(key).lower() in ('gt','ground_truth','oracle','annotation') or _forbidden(item)
                   for key,item in value.items())
    if isinstance(value,(tuple,list)): return any(_forbidden(item) for item in value)
    return isinstance(value,str) and any(term in value.lower() for term in ('oracle','ground_truth','annotation','offline_gt'))


def certify_relative_direction(candidate, frame, center_xy, evidence, calibration,
                               associations=(), max_age_s=1.):
    """Check evidence for ONE uniquely associated historical target.

    Evidence must be produced upstream from current/source observations, not
    inferred from detector score or ROI texture. The present OmniDrive adapter
    has no independent instance foreground producer, so its evidence rejects.
    associations is the complete proposed (source anchor,current target) set;
    mutually incompatible matches reject together, independent of call order.
    """
    tick=perf_counter();interval=None;reason='ok';value=None
    identity=evidence.get('identity',{});foreground=evidence.get('foreground',{})
    age=frame.timestamp-candidate.source_timestamp
    freshness=dict(age_s=float(age),max_age_s=max_age_s,scene_equal=candidate.scene_id==frame.scene_id)
    if _forbidden(evidence) or _forbidden(candidate.computation_source):
        reason='forbidden_gt_online_input'
    elif candidate.fact_type!='relative_direction': reason='unsupported_fact_type'
    elif not (0<max_age_s<=1. and math.isfinite(age) and 0<age<=max_age_s and candidate.scene_id==frame.scene_id):
        reason='stale_or_scene_change'
    elif (not candidate.source_frame or candidate.source_frame==frame.frame_id or
          candidate.computation_source.get('origin') not in ('actual_perception','historical_full_model_detection_replay') or
          candidate.computation_source.get('source_execution_signature')!=frame.execution_signature):
        reason='source_version_or_provenance_mismatch'
    elif not (identity.get('unique') is True and identity.get('bidirectional') is True and
              identity.get('source_anchor')==candidate.anchor_id and identity.get('current_target') and
              identity.get('source_frame')==candidate.source_frame and identity.get('current_frame')==frame.frame_id and
              identity.get('source')=='foreground_owned_bidirectional_correspondence'):
        reason='identity_unverified'
    elif (len({target for source,target in associations if source==candidate.anchor_id})!=1 or
          len({source for source,target in associations if target==identity['current_target']})!=1 or
          (candidate.anchor_id,identity['current_target']) not in associations):
        reason='incompatible_or_ambiguous_association'
    elif not (foreground.get('source_owned') is True and foreground.get('current_owned') is True and
              foreground.get('occlusion_rejected') is True and foreground.get('evidence_digest') and
              foreground.get('source_frame')==candidate.source_frame and foreground.get('current_frame')==frame.frame_id and
              foreground.get('source') in ('independent_instance_segmentation','independent_sensor_instance_support')):
        reason='foreground_unverified'
    elif not isinstance(calibration,DirectionCalibration): reason='calibrated_coordinate_error_bound_missing'
    elif (calibration.source!='offline_calibration_only' or calibration.method!='scene_max_split_conformal' or
          not calibration.calibration_id or not 0<calibration.alpha<=.05 or calibration.execution_signature!=frame.execution_signature or
          frame.scene_id in calibration.calibration_scenes): reason='calibration_scope_mismatch'
    elif calibration.radius_xy_m is None: reason='insufficient_independent_calibration_scenes'
    else:
        center=np.asarray(center_xy,dtype=float);radius=np.asarray(calibration.radius_xy_m,dtype=float)
        if center.shape!=(2,) or radius.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(radius).all() or (radius<0).any():
            reason='invalid_coordinate_interval'
        else:
            lower=center-radius;upper=center+radius
            interval=dict(x=[float(lower[0]),float(upper[0])],y=[float(lower[1]),float(upper[1])],
                          calibration_id=calibration.calibration_id,alpha=calibration.alpha)
            if ((lower<=0)&(upper>=0)).any(): reason='direction_interval_crosses_zero'
            else:
                value=dict(longitudinal=int(np.sign(center[0])),lateral=int(np.sign(center[1])))
                if candidate.fact_value!=value:reason='candidate_value_interval_mismatch'
    return FactCertification(reason=='ok',candidate.fact_type,value,dict(evidence),interval,
        candidate.source_frame,frame.frame_id,dict(identity),dict(foreground),freshness,
        (perf_counter()-tick)*1000,reason)
