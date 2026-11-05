import math
from time import perf_counter

def target_binding_risk(profile, witness, witnesses, current, previous, temporal=False):
    if temporal:
        raise ValueError('R0 does not enable temporal risk')
    from .answer_validity import fact_directions, fact_region
    tick = perf_counter()
    distance = lambda a: math.hypot(*a.center_ego[:2])
    nearest = profile.selector == 'nearest'
    competitors = []
    for a in witnesses:
        if a.anchor_id == witness.anchor_id:
            continue
        margin = distance(a) - distance(witness)
        changes_answer = a.semantic_class != witness.semantic_class if profile.intent == 'category' else fact_directions(a.center_ego) != fact_directions(witness.center_ego)
        if nearest and margin < max(2.0, 0.15 * distance(witness)) and changes_answer:
            competitors.append(dict(anchor_id=a.anchor_id, distance_gap_m=margin, angular_gap_rad=abs(math.atan2(a.center_ego[1], a.center_ego[0]) - math.atan2(witness.center_ego[1], witness.center_ego[0]))))
    boundary = nearest and (profile.direct and abs(abs(witness.center_ego[1]) - 4.0) < 0.75 or (profile.intent == 'position' and any((abs(abs(v) - 1.0) < 0.75 for v in witness.center_ego[:2]))))
    current_risk = min(1.0, 0.45 * bool(competitors) + 0.35 * bool(boundary) + 0.2 * (1.0 - witness.score) + 0.1 * (len(witness.cameras) == 1))
    out = dict(current_risk=current_risk, temporal_risk=0.0, current_competitors=competitors, boundary=bool(boundary), temporal_enabled=temporal, temporal_competitors=[], current_binding_wall_ms=(perf_counter() - tick) * 1000, temporal_risk_wall_ms=0.0, temporal_status='disabled', score_kind='UNCALIBRATED_BINDING_RISK', continuity=dict(current_observation_required=True, matched=False, identity_qualified=False, admission_credit=0.0, reason='No calibrated positive-evidence admission credit'))
    tick = perf_counter()
    out['temporal_risk_wall_ms'] = (perf_counter() - tick) * 1000 if temporal and nearest else 0.0
    out['total_risk'] = min(1.0, out['current_risk'] + out['temporal_risk'])
    return out
