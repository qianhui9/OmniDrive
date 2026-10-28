"""Empirically qualified identity/class continuity; never physical certification.

Online extraction consumes only current committed M1 observations. Ground truth
and qualification fitting live in the offline experiment tool, not this module.
No visibility, geometry, motion, absence, or full-scene count claim is licensed.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from worldstream.code_signature import stable_function_hash
from .stable_world import digest

QUALIFICATION_VERSION = 'eq_ssc_identity_class_v1'
SEMANTIC_CLAIM = 'empirical_associated_identity_and_observed_class_continuity'


@dataclass(frozen=True)
class SemanticContinuityEvidence:
    anchor_id: str
    anchor_type: str
    semantic_class: str
    source_frame: str
    current_frame: str
    source_timestamp: float
    timestamp: float
    consecutive_observations: int
    score: float
    residual_m: float
    margin_m: object
    no_competing_edge: bool
    observed_camera_count: int
    source_state_version: str
    current_state_version: str
    qualification_version: str = QUALIFICATION_VERSION


def protocol_signature(manager):
    from worldstream.m1.association import associate
    return digest(dict(config=manager.config.to_dict(),
        associate=stable_function_hash(associate),commit=stable_function_hash(type(manager).commit),
        extractor=stable_function_hash(committed_semantic_candidates),claim=SEMANTIC_CLAIM))


def committed_semantic_candidates(manager, context):
    store=manager.store
    if (store.scene_id!=context['scene_id'] or manager._previous_frame_id!=context['frame_id']
            or store.last_timestamp!=context['timestamp'] or store.generation!=context['state_epoch']
            or store.commit_count!=context['state_version']
            or store.execution_signature!=context['state_execution_signature']
            or manager.last_commit_diagnostics.get('transaction_status')=='rejected'):
        raise ValueError('Semantic extraction requires the matching current M1 commit')
    records=[]
    for a in store.anchors.values():
        if (a.anchor_type!='object' or a.evidence_origin!='actual_perception'
                or a.source_frame_id!=context['frame_id'] or a.source_timestamp!=context['timestamp']):continue
        history=a.association_history
        consecutive=1
        for row in reversed(history):
            if (not row.get('source_is_previous_frame') or row.get('semantic_class')!=a.semantic_class
                    or row.get('association_residual_m') is None):break
            consecutive+=1
        last=history[-1] if history else {}
        residual=last.get('association_residual_m')
        score=a.source_score
        eligible=(last.get('source_is_previous_frame') and consecutive>=2
            and residual is not None and score is not None and math.isfinite(score)
            and math.isfinite(residual) and bool(a.observation_support))
        evidence=None
        if eligible:
            evidence=asdict(SemanticContinuityEvidence(a.anchor_id,a.anchor_type,a.semantic_class,
                last['source_frame_id'],a.source_frame_id,last['source_timestamp'],a.source_timestamp,
                consecutive,float(score),float(residual),last.get('association_margin_m'),
                bool(last.get('no_competing_edge')),len(a.observation_support),
                last.get('source_state_version') or '',a.state_version))
        records.append(dict(anchor_id=a.anchor_id,anchor_type=a.anchor_type,semantic_class=a.semantic_class,
            score=float(score) if score is not None and math.isfinite(score) else None,
            has_history=bool(last.get('source_frame_id')),eligible_observation=bool(eligible),evidence=evidence))
    return dict(context=dict(context),protocol_signature=protocol_signature(manager),records=records,
        claim=SEMANTIC_CLAIM,physical_certification=False,requires_current_observation=True)


class SemanticQualifier:
    def __init__(self,path):
        raw=Path(path).read_bytes()
        self.policy=json.loads(raw,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
        self.sha256=hashlib.sha256(raw).hexdigest()
        if (self.policy.get('version')!=QUALIFICATION_VERSION or self.policy.get('claim')!=SEMANTIC_CLAIM
                or not self.policy.get('independent_validation_completed')
                or not self.policy.get('calibration_scene_ids') or not self.policy.get('validation_scene_ids')
                or set(self.policy['calibration_scene_ids']) & set(self.policy['validation_scene_ids'])
                or set(self.policy['calibration_log_ids']) & set(self.policy['validation_log_ids'])):
            raise ValueError('A scene/log-disjoint independently validated semantic policy is required')
        self.rules={k:v for k,v in self.policy.get('rules',{}).items() if v.get('admitted') is True}
        if self.policy.get('selector_implementation')!=self.implementation_signature():
            raise ValueError('Semantic qualification selector implementation changed')
        limits=self.policy['qualification_limits']
        for rule in self.rules.values():
            if (not math.isfinite(rule['min_score']) or not math.isfinite(rule['max_residual_m'])
                    or rule['max_residual_m']<0 or type(rule['min_observations']) is not int or rule['min_observations']<2):
                raise ValueError('Invalid calibrated semantic rule')
            for partition in ('calibration','validation'):
                metrics=rule[partition]
                if (metrics['scored']<limits['min_scored_edges'] or metrics['scenes']<limits['min_scenes']
                        or metrics['wrong']>limits['max_observed_wrong_edges']
                        or metrics['unknown_fraction']>limits['max_unknown_fraction']):
                    raise ValueError('Semantic rule lacks independent empirical support')

    @classmethod
    def implementation_signature(cls):
        return digest((stable_function_hash(cls.select),stable_function_hash(cls.rejection)))

    def select(self,candidates):
        if candidates['protocol_signature']!=self.policy['protocol_signature']:
            raise ValueError('Semantic qualifier M1/extractor protocol mismatch')
        facts=[];evidence=[];rejected={}
        now=candidates['context']['timestamp']
        for record in candidates['records']:
            rule=self.rules.get(record['semantic_class'])
            reason=self.rejection(record,rule,now)
            if reason:rejected[record['anchor_id']]=reason
            else:
                facts.append((record['anchor_id'],record['anchor_type'],record['semantic_class']))
                evidence.append(record['evidence'])
        # Prefer an established high-confidence observed entity when creating
        # the one immutable core. Existing core members are maintained by ID.
        priorities={r['anchor_id']:(-(r['score'] or 0),-r['evidence']['consecutive_observations'],r['anchor_id'])
            for r in candidates['records'] if r['anchor_id'] not in rejected}
        return dict(facts=tuple(sorted(facts)),evidence=evidence,rejected=rejected,
            priority_ids=sorted(priorities,key=priorities.get),qualification_sha256=self.sha256,
            claim=SEMANTIC_CLAIM,physical_certification=False)

    @staticmethod
    def rejection(record,rule,now):
        if not rule:return 'semantic_type_not_independently_qualified'
        e=record.get('evidence')
        if not e:return 'no_current_consecutive_observation'
        if (e['current_frame']==e['source_frame'] or e['timestamp']!=now
                or not 0<now-e['source_timestamp']<=1.):return 'stale_or_invalid_source'
        if e['consecutive_observations']<rule['min_observations']:return 'insufficient_consecutive_observations'
        if e['score']<rule['min_score']:return 'class_score_below_calibrated_range'
        if e['residual_m']>rule['max_residual_m']:return 'association_residual_above_calibrated_range'
        margin=e['margin_m']
        if not e['no_competing_edge'] and (margin is None or margin<=0):return 'association_ambiguous'
        if not e['observed_camera_count']:return 'no_current_observation_support'
        return None
