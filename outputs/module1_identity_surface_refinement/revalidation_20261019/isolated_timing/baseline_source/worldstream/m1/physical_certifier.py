"""Claim-scoped verification of authenticated measured online evidence.

Same-RGB evidence needs independently scored producer qualification. No approved
qualification is shipped. Object continuity never certifies metric centre/speed.
"""
from dataclasses import asdict, replace
import hashlib
import hmac
import json
import math
import secrets
import numpy as np
from .geometry import state_geometry_version, oriented_box_membership
from .input_contract import frame_digest, reject_oracle_metadata
from .state_types import PhysicalEvidence, PhysicalCertificationResult

CONTINUITY_CLAIM = 'static_candidate_identity_continuity_with_local_ray_consistency'
PRODUCER_ID = 'same_rgb_box_contrast_correspondence_v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class EvidenceIssuer:
    """Local execution provenance, not security against arbitrary Python code.

    Runtime issuance occurs only in the measured producer. The pinned artifact
    supplies OFFLINE qualification and never current-frame GT measurements.
    """
    def __init__(self, qualification=None, pinned_sha256=None):
        self._key = secrets.token_bytes(32)
        self._qualification_json = None
        if qualification is not None:
            encoded = _canonical(qualification)
            if hashlib.sha256(encoded.encode()).hexdigest() != pinned_sha256:
                raise ValueError('Qualification artifact does not match its audited digest')
            self._qualification_json = encoded

    def _mint_measured(self, evidence):
        value = asdict(evidence);value.pop('authentication_tag')
        return replace(evidence,authentication_tag=hmac.new(self._key,_canonical(value).encode(),hashlib.sha256).hexdigest())

    def authentic(self, evidence):
        if not isinstance(evidence,PhysicalEvidence):return False
        value=asdict(evidence);tag=value.pop('authentication_tag')
        try:expected=hmac.new(self._key,_canonical(value).encode(),hashlib.sha256).hexdigest()
        except (TypeError,ValueError):return False
        return isinstance(tag,str) and hmac.compare_digest(tag,expected)

    def qualification_reasons(self, evidence, frame, protocol_digest,
                              claim=CONTINUITY_CLAIM, population='measured_box_contrast_candidates'):
        if self._qualification_json is None:return ['independent_producer_qualification_missing']
        q=json.loads(self._qualification_json);reasons=[]
        if q.get('producer_id')!=evidence.producer_id or q.get('claim')!=claim:
            reasons.append('qualification_claim_or_producer_mismatch')
        if q.get('protocol_digest')!=protocol_digest or q.get('population')!=population:
            reasons.append('qualification_population_or_protocol_mismatch')
        units=q.get('validation_scene_units',[]);scenes=[u.get('scene_id') for u in units]
        logs=[u.get('log_token') for u in units]
        if (not units or len(set(scenes))!=len(scenes) or any(not s for s in scenes)
                or any(u.get('accepted_entities',0)<1 or u.get('unscorable_accepted',1)!=0
                       or not isinstance(u.get('false_acceptances'),int) or u['false_acceptances']<0
                       or u['false_acceptances']>u['accepted_entities'] for u in units)):
            reasons.append('independent_validation_units_invalid');return reasons
        if (len(set(logs))!=len(logs) or any(not log for log in logs)
                or set(scenes)&set(q.get('calibration_scenes',[]))
                or set(logs)&set(q.get('calibration_logs',[]))):
            reasons.append('validation_scene_or_log_independence_invalid')
        if frame.scene_id in set(scenes)|set(q.get('calibration_scenes',[])):
            reasons.append('qualification_scene_reused_for_evaluation')
        from scipy.stats import beta
        # Conservative scene event: any wrong accepted entity in that scene.
        # Repeated frames/entities never increase the independent sample count.
        errors=sum(u['false_acceptances']>0 for u in units);n=len(units)
        upper=float(beta.ppf(.95,errors+1,n-errors)) if errors<n else 1.
        if n<3 or upper>.05:reasons.append('scene_level_physical_risk_not_qualified')
        if q.get('independence_basis')!='disjoint_new_M1_scenes_and_logs':
            reasons.append('scene_independence_not_established')
        return reasons


def certify_physical_evidence(anchor, current_anchor, frame, evidence=None, issuer=None,
                              source_frame_digest=None, protocol_digest=None, claim=CONTINUITY_CLAIM):
    """Validate the named limited claim; centre/distance/velocity stay unapproved."""
    from .identity_continuity import IDENTITY_CLAIM, certify_identity_evidence
    if claim==IDENTITY_CLAIM:
        return certify_identity_evidence(anchor,current_anchor,frame,evidence,issuer,source_frame_digest,protocol_digest)
    reasons=[];provenance=dict(same_sensor_RGB=True,independent_sensor=False,
                               geometry_attributes_certified=False,motion_attributes_certified=False)
    if claim!=CONTINUITY_CLAIM:reasons.append('unsupported_claim_requires_calibrated_metric_or_motion_bounds')
    if not isinstance(evidence,PhysicalEvidence):
        reasons.extend(['measured_identity_correspondence_missing','measured_foreground_ownership_missing'])
    elif issuer is None or not issuer.authentic(evidence):reasons.append('untrusted_or_modified_evidence_receipt')
    else:
        try:
            m=json.loads(evidence.measurements_json);reject_oracle_metadata(m)
            if evidence.producer_id!=PRODUCER_ID:reasons.append('untrusted_producer')
            if (evidence.anchor_id!=anchor.anchor_id or current_anchor.anchor_id!=anchor.anchor_id
                    or evidence.scene_id!=frame.scene_id or anchor.scene_id!=frame.scene_id
                    or current_anchor.scene_id!=frame.scene_id):
                reasons.append('anchor_or_scene_binding_mismatch')
            if (evidence.source_frame_id!=anchor.source_frame_id or evidence.current_frame_id!=frame.frame_id
                    or evidence.source_frame_id==evidence.current_frame_id or not source_frame_digest
                    or evidence.source_frame_digest!=source_frame_digest):reasons.append('source_frame_not_accessible_or_mismatched')
            if evidence.source_epoch!=anchor.source_state_epoch or current_anchor.source_state_epoch!=anchor.source_state_epoch:
                reasons.append('source_epoch_mismatch')
            if not isinstance(evidence.source_epoch,int) or evidence.source_epoch<1:
                reasons.append('invalid_source_epoch')
            if (current_anchor.source_frame_id!=frame.frame_id or current_anchor.source_timestamp!=frame.timestamp
                    or current_anchor.source_execution_signature!=frame.execution_signature):
                reasons.append('current_computed_state_binding_mismatch')
            if (evidence.source_state_version!=state_geometry_version(anchor)
                    or evidence.current_state_version!=state_geometry_version(current_anchor)):
                reasons.append('state_geometry_version_mismatch')
            if (evidence.execution_signature!=frame.execution_signature
                    or anchor.source_execution_signature!=frame.execution_signature
                    or evidence.current_frame_digest!=frame_digest(frame)):
                reasons.append('execution_or_observation_digest_mismatch')
            age=frame.timestamp-anchor.source_timestamp
            if not math.isfinite(age) or not 0<age<=1. or evidence.evidence_timestamp!=frame.timestamp:
                reasons.append('stale_or_future_evidence')
            points=np.asarray(m.get('world_points',[]),float)
            fb=np.asarray(m.get('fb_error_px',[]),float);angles=np.asarray(m.get('parallax_deg',[]),float)
            reproj=np.asarray(m.get('reprojection_error_px',[]),float)
            contrast=np.asarray(m.get('foreground_background_depth_gap_m',[]),float)
            n=len(points)
            if (points.ndim!=2 or points.shape!=(n,3) or n<4 or not np.isfinite(points).all()
                    or any(x.shape!=(n,) or not np.isfinite(x).all() for x in (fb,angles,reproj))
                    or (fb>.75).any() or (angles<.5).any() or (reproj>1.).any()):
                reasons.append('measured_correspondence_or_observability_insufficient')
            elif not (oriented_box_membership(points,anchor.geometry_world).all()
                      and oriented_box_membership(points,current_anchor.geometry_world).all()):
                reasons.append('measured_points_contradict_bound_object_geometry')
            if (contrast.ndim!=1 or len(contrast)<4 or not np.isfinite(contrast).all()
                    or np.median(contrast)<=0 or m.get('spatial_quadrants',0)<3):
                reasons.append('foreground_background_not_distinguished')
            if m.get('association_alternatives',1)!=0:reasons.append('identity_ambiguous')
            if m.get('conflicting_camera_count',1)!=0:reasons.append('multi_camera_evidence_conflict')
            if m.get('static_ray_contradictions',1)!=0:reasons.append('static_hypothesis_contradicted')
            if m.get('dependency')!='RGB_correspondence_with_predicted_box_proposal':
                reasons.append('self_asserted_or_circular_evidence_dependency')
            reasons.extend(issuer.qualification_reasons(evidence,frame,protocol_digest))
            provenance.update(producer=evidence.producer_id,source_frame_digest=evidence.source_frame_digest,
                receipt_digest=evidence.authentication_tag,measurement_count=n,protocol_digest=protocol_digest,
                claim_scope='Qualified identity continuity and local ray consistency; no metric centre or speed guarantee')
        except (ValueError,TypeError,KeyError,OverflowError):reasons.append('malformed_or_forbidden_online_evidence')
    return PhysicalCertificationResult(not reasons,claim if not reasons else None,tuple(sorted(set(reasons))),provenance,
        evidence.source_state_version if isinstance(evidence,PhysicalEvidence) else None,
        evidence.current_state_version if isinstance(evidence,PhysicalEvidence) else None,
        anchor.source_frame_id,frame.frame_id,anchor.anchor_id,frame.scene_id,frame.timestamp,anchor.source_state_epoch)


def certify_available_state(anchor, frame, observation_supported, evidence=None, issuer=None,
                            current_anchor=None, source_frame_digest=None, protocol_digest=None):
    """Compatibility diagnostics; genuine measured evidence defaults absent."""
    result=certify_physical_evidence(anchor,current_anchor or anchor,frame,evidence,issuer,source_frame_digest,protocol_digest)
    value=asdict(result);value['rejection_reasons']=list(result.rejection_reasons)
    value.update(foreground_ownership_certified=result.certified,identity_certified=result.certified,
        geometry_uncertainty_status='uncalibrated',motion_status='unknown',producer_available=evidence is not None)
    if evidence is None:
        value['rejection_reasons']+=['independent_identity_missing','independent_foreground_ownership_missing',
            'calibrated_localization_bound_missing','independent_motion_evidence_missing']
    if not observation_supported and evidence is None:value['rejection_reasons'].append('observation_not_supported')
    return value
