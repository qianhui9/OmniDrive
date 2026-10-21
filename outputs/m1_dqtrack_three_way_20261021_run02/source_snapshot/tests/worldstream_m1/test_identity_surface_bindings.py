"""Adversarial input contracts, using controlled qualification only."""
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from worldstream.m1 import FullOutputs
from worldstream.m1.identity_continuity import IDENTITY_CLAIM, identity_protocol_digest, propose_identity_continuity
from worldstream.m1.physical_certifier import EvidenceIssuer, _canonical, certify_physical_evidence
from worldstream.m1.surface_visibility import surface_visibility_support
from test_identity_surface import identity_manager, identity_receipt, triple


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), 99.0])
def test_identity_tracks_require_finite_actual_source_time(config, make_frame, make_prediction, timestamp):
    m, old, current, frame, _, _ = identity_receipt(config, make_frame, make_prediction)
    evidence = copy.deepcopy(m.last_evidence)
    evidence['front'].source_timestamp = timestamp
    proposal = propose_identity_continuity(old, current, frame, evidence, m.config, m.validator,
        m._physical_issuer, m._previous_frame_digest, m._previous_frame_id)
    assert proposal['status'] == 'unknown'
    assert proposal['evidence'] is None


@pytest.mark.parametrize('mutation', ['source_region', 'duplicate_tracks', 'negative_fb', 'camera_id', 'view_conflict'])
def test_identity_receipt_revalidates_observation_population(config, make_frame, make_prediction, mutation):
    m, old, current, frame, _, evidence = identity_receipt(config, make_frame, make_prediction)
    measurements = json.loads(evidence.measurements_json)
    view = measurements['views'][0]
    if mutation == 'source_region':
        view['source_region'] = [0, 0, 120, 100]
    elif mutation == 'duplicate_tracks':
        view['source_points'] = [view['source_points'][0]] * len(view['source_points'])
        view['current_points'] = [view['current_points'][0]] * len(view['current_points'])
    elif mutation == 'negative_fb':
        view['fb_error_px'][0] = -1.0
    elif mutation == 'camera_id':
        old.observation_support['front'].camera_id = 'other'
    else:
        view['reason'] = 'temporal_source_mismatch'
    receipt = m._physical_issuer._mint_measured(replace(evidence, measurements_json=json.dumps(measurements)))
    result = certify_physical_evidence(old, current, frame, receipt, m._physical_issuer,
        m._previous_frame_digest, identity_protocol_digest(m.config, frame), IDENTITY_CLAIM)
    assert not result.certified


@pytest.mark.parametrize('mutation', ['units_not_list', 'unit_not_object', 'boolean_errors', 'fractional_entities',
    'calibration_not_list', 'duplicate_logs'])
def test_malformed_risk_qualification_rejects_without_raising(config, make_frame, make_prediction, mutation):
    m, old, current, frame, qualification, evidence = identity_receipt(config, make_frame, make_prediction)
    if mutation == 'units_not_list':
        qualification['validation_scene_units'] = None
    elif mutation == 'unit_not_object':
        qualification['validation_scene_units'][0] = None
    elif mutation == 'boolean_errors':
        qualification['validation_scene_units'][0]['false_acceptances'] = False
    elif mutation == 'fractional_entities':
        qualification['validation_scene_units'][0]['accepted_entities'] = 1.5
    elif mutation == 'calibration_not_list':
        qualification['calibration_scenes'] = 'cal'
    else:
        qualification['validation_scene_units'][0]['log_token'] = qualification['validation_scene_units'][1]['log_token']
    issuer = EvidenceIssuer(qualification, hashlib.sha256(_canonical(qualification).encode()).hexdigest())
    receipt = issuer._mint_measured(evidence)
    result = certify_physical_evidence(old, current, frame, receipt, issuer,
        m._previous_frame_digest, identity_protocol_digest(m.config, frame), IDENTITY_CLAIM)
    assert not result.certified


def test_commit_readout_requires_committed_execution_signature(config, make_frame, make_prediction):
    m, prediction, frame, _ = identity_manager(config, make_frame, make_prediction)
    result = m.precheck(frame)
    m.commit(frame, FullOutputs([prediction]), result)
    assert m.get_current_certifications().certified_anchor_ids
    m.store.execution_signature = 'different-model-execution'
    assert not m.get_current_certifications().certified_anchor_ids


@pytest.mark.parametrize('mutation', ['camera_id', 'coordinate_space', 'nan_timestamp'])
def test_surface_history_requires_same_camera_contract(mutation):
    older, previous, current, _, history = triple()
    if mutation == 'camera_id':
        older.camera_id = 'other'
    elif mutation == 'coordinate_space':
        older.coordinate_space = 'unrelated_preprocessed_image'
    else:
        older.timestamp = float('nan')
    source = history[2]
    # triple's measured previous-to-older map also represents current-to-previous
    # on its constant-velocity plane; no artificial depth labels enter runtime.
    visible, _ = surface_visibility_support(previous, current, source,
        np.ones(current.image.shape[:2], bool), history)
    assert not visible.any()
