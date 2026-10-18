"""Strict world-state boundary for the evidence actually produced by this adapter.

There is currently no independent foreground/identity or calibrated motion and
localization producer. Neither an ROI match, a class prior, nor a user-supplied
boolean is an admissible replacement. This reports the missing dependencies;
it does not issue hypothetical certificates that downstream code could consume.
"""


def certify_available_state(anchor, frame, observation_supported):
    reasons = ['independent_identity_missing', 'independent_foreground_ownership_missing',
               'calibrated_localization_bound_missing', 'independent_motion_evidence_missing']
    if not observation_supported:
        reasons.append('observation_not_supported')
    if (anchor.scene_id != frame.scene_id or anchor.source_frame_id is None
            or anchor.source_execution_signature != frame.execution_signature
            or not 0 < frame.timestamp - anchor.source_timestamp):
        reasons.append('source_not_fresh_or_accessible')
    return dict(certified=False, rejection_reasons=reasons,
                evidence_source='available_online_RGB_and_predicted_geometry_only',
                foreground_ownership_certified=False, identity_certified=False,
                geometry_uncertainty_status='uncalibrated',
                motion_status='unknown', producer_available=False)
