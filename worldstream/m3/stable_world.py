"""Causal, claim-scoped stable facts from M1's committed state.

Geometric associations and RGB support are candidates, not stable identities.
Only M1's authenticated, independently qualified continuity certificates can
publish identity/class facts. Metric geometry, pose, velocity and signal state
are deliberately outside that certificate's claim and remain current vision.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StableWorldSnapshot:
    scene_id: str
    frame_id: str
    timestamp: float
    state_epoch: int
    state_version: int
    execution_signature: str
    facts: tuple
    provenance_json: str
    content_sha256: str = ''

    def __post_init__(self):
        if (not isinstance(self.facts, tuple) or any(not isinstance(f, tuple) or len(f)!=3
                or any(not isinstance(v,str) or not v for v in f) for f in self.facts)
                or self.facts != tuple(sorted(set(self.facts)))):
            raise ValueError('Stable facts must be canonical immutable identity/type/class triples')
        payload = asdict(self); payload.pop('content_sha256')
        object.__setattr__(self, 'content_sha256', digest(payload))

    def validate(self, context):
        payload = asdict(self); payload.pop('content_sha256')
        if digest(payload) != self.content_sha256:
            raise ValueError('Stable snapshot was modified')
        for name in ('scene_id', 'frame_id', 'timestamp', 'state_epoch', 'state_version'):
            if getattr(self, name) != context.get(name):
                raise ValueError('Stable snapshot context mismatch: '+name)
        if self.execution_signature != context.get('state_execution_signature'):
            raise ValueError('Stable snapshot execution mismatch')


def freeze_stable_world(manager, context):
    """Read only the current commit; never use pending or future perception."""
    store = manager.store
    certifications = manager.get_current_certifications()
    if (certifications.scene_id != context['scene_id']
            or certifications.frame_id != context['frame_id']
            or certifications.timestamp != context['timestamp']
            or store.generation != context['state_epoch']
            or store.commit_count != context['state_version']
            or store.execution_signature != context['state_execution_signature']):
        raise ValueError('Stable world requires a matching completed M1 commit')
    facts, sources, scores = [], [], []
    for certificate in certifications.certifications:
        anchor = store.anchors[certificate.anchor_id]
        # get_current_certifications verifies receipt, frame, epoch and geometry
        # versions. Its limited claim licenses identity only, never map geometry.
        if not certificate.evidence_provenance.get('risk_qualification_sha256'):
            continue
        facts.append((anchor.anchor_id, anchor.anchor_type, anchor.semantic_class))
        sources.append(asdict(certificate))
        scores.append(dict(anchor_id=anchor.anchor_id,score=anchor.source_score,
            status='uncalibrated_detector_class_score_not_a_stability_probability'))
    rejected = Counter(reason for r in certifications.rejected for reason in r['rejection_reasons'])
    provenance = dict(source='M1.get_current_certifications', candidates=len(store.anchors),
        unqualified_geometric_tracks=sum(not a.identity_link_qualified for a in store.anchors.values()),
        multi_observation_candidates=sum(a.history_observations>1 for a in store.anchors.values()),
        accepted=len(facts), certificates=sources, rejection_reasons=dict(rejected),
        class_prediction_scores=scores,
        claim_scope='identity and currently observed class only; no geometry or motion',
        producer_qualification_sha256=manager._physical_issuer.qualification_sha256)
    return StableWorldSnapshot(str(store.scene_id), str(context['frame_id']), context['timestamp'],
        store.generation, store.commit_count, store.execution_signature, tuple(sorted(facts)),
        json.dumps(provenance, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class StableWorldAnchor:
    anchor_id: str
    scene_id: str
    stable_state_digest: str
    source_frame: str
    anchor_age: int
    stable_state_valid: bool
    facts: tuple
    source_snapshot_sha256: str
    update_reason: str


class StableAnchorManager:
    """One bounded anchor, revalidated from each real current M1 snapshot."""
    def __init__(self, max_age_frames=8, max_gap_s=1.):
        if type(max_age_frames) is not int or max_age_frames < 1 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('Invalid stable anchor horizon')
        self.max_age_frames, self.max_gap_s = max_age_frames, max_gap_s
        self.reset()

    def reset(self):
        self.anchor = self.previous = None
        self.generation = 0

    def update(self, snapshot, context):
        snapshot.validate(context)
        if not math.isfinite(snapshot.timestamp):
            raise ValueError('Invalid stable world time')
        prior = self.previous
        if prior is not None and snapshot.frame_id == prior.frame_id and snapshot.scene_id == prior.scene_id:
            if snapshot.content_sha256 != prior.content_sha256:
                raise ValueError('Same-frame stable snapshot changed')
            return self.anchor, self.anchor.stable_state_valid
        semantic_digest = digest(snapshot.facts)
        reason = 'cold_anchor'
        if prior is not None:
            if snapshot.scene_id != prior.scene_id: reason = 'scene_changed'
            elif snapshot.state_epoch != prior.state_epoch: reason = 'state_epoch_changed'
            elif snapshot.execution_signature != prior.execution_signature: reason = 'state_execution_changed'
            elif snapshot.state_version < prior.state_version: reason = 'state_version_rollback'
            elif not 0 < snapshot.timestamp-prior.timestamp <= self.max_gap_s: reason = 'time_discontinuity'
            elif not snapshot.facts: reason = 'no_qualified_stable_facts'
            elif semantic_digest != self.anchor.stable_state_digest: reason = 'stable_content_changed'
            elif self.anchor.anchor_age+1 >= self.max_age_frames: reason = 'anchor_horizon_expired'
            elif self.anchor.stable_state_valid:
                self.anchor = replace(self.anchor, anchor_age=self.anchor.anchor_age+1)
                self.previous = snapshot
                return self.anchor, True
        if not snapshot.facts: reason = 'no_qualified_stable_facts'
        self.generation += 1
        identity = digest((snapshot.scene_id, snapshot.frame_id, snapshot.content_sha256, self.generation))
        self.anchor = StableWorldAnchor(identity, snapshot.scene_id, semantic_digest, snapshot.frame_id,
            0, bool(snapshot.facts), snapshot.facts, snapshot.content_sha256, reason)
        self.previous = snapshot
        return self.anchor, False
