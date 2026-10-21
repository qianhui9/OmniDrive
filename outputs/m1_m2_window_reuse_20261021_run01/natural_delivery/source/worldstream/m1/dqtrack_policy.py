"""Uncertified external-identity proposals on existing legal geometry edges.

No GT, no velocity reliability or physical evidence enters this module. Online
use requires pinned independent qualification; offline research uses the same
pure proposal function with an explicitly labelled research-only profile.
"""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import numpy as np


def validate_profile(profile,config):
    p=dict(profile)
    if (not np.isfinite(p['min_score']) or not 0<=p['min_score']<=1
            or type(p['min_observations']) is not int or p['min_observations']<3
            or not np.isfinite(p['max_hint_age_s']) or not 0<p['max_hint_age_s']<=config.max_state_age_s):
        raise ValueError('Invalid bounded external identity profile')
    return p


def qualified_profile(config):
    try:
        path=config.dqtrack_policy_qualification_path
        if not path:return None,dict(status='inactive',reason='independent_qualification_missing')
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config.dqtrack_policy_qualification_sha256:
            raise ValueError('qualification_digest_mismatch')
        artifact=json.loads(raw)
        if artifact['policy_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('policy_source_mismatch')
        dev=set(artifact['development_scene_ids']);hold=set(artifact['independent_scene_ids'])
        if len(dev)<4 or len(hold)<4 or dev&hold or not artifact['independent_exposure_audit_passed']:
            raise ValueError('independent_scene_evidence_insufficient')
        for split in ('development','independent'):
            old=artifact[split]['conservative'];new=artifact[split]['candidate']
            if (new['wrong_continuations']>=old['wrong_continuations']
                    or new['fragments']>old['fragments'] or new['correct_continuations']<old['correct_continuations']
                    or artifact[split]['m1_overhead_fraction']>.1):
                raise ValueError('no_joint_risk_recall_fragmentation_cost_benefit')
        if artifact['geometry']!={k:getattr(config,k) for k in ('association_gate_m','box_size_ratio','max_state_age_s')}:
            raise ValueError('geometry_configuration_mismatch')
        return validate_profile(artifact['profile'],config),dict(status='qualified',artifact_sha256=hashlib.sha256(raw).hexdigest())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return None,dict(status='inactive',reason=str(exc))


def external_key(hint):
    return (tuple(hint.get('namespace',[])),hint.get('tracking_id'))


def recent_metadata(prior,hint,timestamp,profile):
    """Keep timestamped recent data; a retained record is never a current hint."""
    old=prior.external_tracking if prior is not None else {}
    if hint:
        same=bool(old and external_key(old)==external_key(hint))
        consecutive=same and old.get('status')=='aligned' and timestamp-old['timestamp']<=profile['max_hint_age_s']
        return deepcopy(dict(hint,aligned_observations=old.get('aligned_observations',1)+1 if consecutive else 1,
                    expires_at=timestamp+profile['max_hint_age_s']))
    if old and timestamp<=old.get('expires_at',-np.inf):
        return deepcopy(dict(old,status='recent_only',rejection_reason='current_frame_alignment_missing'))
    return {}


def propose(anchors,predictions,accepted,costs,hints,frame,last_timestamp,config,profile):
    profile=validate_profile(profile,config)
    diagnostics=dict(status='research_or_qualified_proposal',proposals=[],rejected=[],
        score_semantics='empirical threshold, not calibrated identity probability',
        velocity_feature='unavailable: external velocity not independently verified')
    if config.identity_continuity or config.foreground_correspondence or config.search_dynamic_motion:
        diagnostics.update(status='inactive',reason='physical_or_experimental_identity_policy_active')
        return list(accepted),set(),diagnostics
    eligible_old={};eligible_new={}
    for i,old in enumerate(anchors):
        h=old.external_tracking;reasons=[]
        if old.anchor_type!='object' or old.scene_id!=frame.scene_id:continue
        if not h:reasons.append('external_history_missing')
        elif (h.get('status')!='aligned' or frame.timestamp-h['timestamp']>profile['max_hint_age_s']
              or h.get('expires_at',-np.inf)<frame.timestamp):reasons.append('lost_occluded_or_expired_hint')
        elif h.get('aligned_observations',1)<profile['min_observations']:reasons.append('newborn_or_short_track')
        elif h['tracking_score']<profile['min_score']:reasons.append('low_empirical_score')
        if frame.timestamp-old.source_timestamp>config.max_state_age_s or old.source_timestamp<last_timestamp:
            reasons.append('state_expired_or_reappeared')
        if reasons:diagnostics['rejected'].append(dict(anchor_id=old.anchor_id,reasons=reasons));continue
        eligible_old.setdefault(external_key(h),[]).append(i)
    for j,h in hints.items():
        if (h.get('status')=='aligned' and h['timestamp']==frame.timestamp
                and h['namespace'][1]==frame.scene_id and h['tracking_score']>=profile['min_score']):
            eligible_new.setdefault(external_key(h),[]).append(j)
        else:diagnostics['rejected'].append(dict(prediction_index=j,reasons=['low_score_or_not_current_hint']))
    external=[]
    for key,rows in eligible_old.items():
        cols=eligible_new.get(key,[])
        if len(rows)!=1 or len(cols)!=1:
            diagnostics['rejected'].append(dict(reasons=['external_identity_nonunique_lost_or_switched']))
            continue
        i,j=rows[0],cols[0]
        if not np.isfinite(costs[i,j]):
            diagnostics['rejected'].append(dict(anchor_id=anchors[i].anchor_id,prediction_index=j,reasons=['illegal_geometry']));continue
        # Refuse identity overrides with unknown competitors. Every endpoint in
        # the legal connected component must have a unique usable external ID.
        row_ids={i};col_ids={j}
        while True:
            nc=set(np.flatnonzero(np.isfinite(costs[list(row_ids)]).any(axis=0)))|col_ids
            nr=set(np.flatnonzero(np.isfinite(costs[:,list(nc)]).any(axis=1)))|row_ids
            if nr==row_ids and nc==col_ids:break
            row_ids,col_ids=nr,nc
        usable_rows={v[0] for v in eligible_old.values() if len(v)==1}
        usable_cols={v[0] for v in eligible_new.values() if len(v)==1}
        if not row_ids<=usable_rows or not col_ids<=usable_cols:
            diagnostics['rejected'].append(dict(anchor_id=anchors[i].anchor_id,prediction_index=j,reasons=['unknown_competitors']));continue
        component=[]
        for ri in row_ids:
            targets=eligible_new.get(external_key(anchors[ri].external_tracking),[])
            if len(targets)==1 and targets[0] in col_ids and np.isfinite(costs[ri,targets[0]]):
                component.append((ri,targets[0]))
        if len(component)!=len(row_ids) or len({cj for ri,cj in component})!=len(col_ids):
            diagnostics['rejected'].append(dict(anchor_id=anchors[i].anchor_id,prediction_index=j,reasons=['incomplete_legal_identity_component']))
            continue
        external.append((i,j))
    # External edges are unique by ID; no score becomes an additive geometry cost.
    blocked_rows={i for i,j in external};blocked_cols={j for i,j in external}
    # Preserve conservative accepted pairs outside the affected components;
    # never silently accept a residual edge merely because one competitor left.
    result=external+[(i,j) for i,j in accepted if i not in blocked_rows and j not in blocked_cols]
    if len(set(i for i,j in result))!=len(result) or len(set(j for i,j in result))!=len(result):
        raise ValueError('External assignment must be one-to-one')
    changed=set(external)-set(accepted)
    diagnostics['proposals']=[dict(anchor_id=anchors[i].anchor_id,prediction_index=j,distance_m=float(costs[i,j])) for i,j in sorted(changed)]
    return result,changed,diagnostics
