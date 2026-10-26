"""Quality-bounded continuous prefixes in the native object/ego/map order."""
from dataclasses import dataclass, fields
from time import perf_counter
import math
from .task_conditioner import task_profile
from .prefix_matcher import PrefixProof, prefix_statistics, input_revision
from .scheduler import POLICY_VERSION

WSDR_VERSION = POLICY_VERSION
GROUPS = ('object', 'ego', 'map')
WEIGHTS = {'object_attributes': (1.25, 1., .75), 'map_relations': (.75, 1., 1.25),
           'object_ego_motion': (1.25, 1.25, .8), 'full_visual': (1.5, 1.5, 1.5)}


@dataclass(frozen=True)
class ReuseDecision:
    reuse_length: int = 0
    reuse_type: str = 'none'
    estimated_quality_risk: float = 0.
    estimated_saved_ms: float = 0.
    cache_source_frame: str = ''
    cache_age_frames: int = 0
    approx_reuse_depth: int = 0
    decision_reason: str = 'cold_cache'
    frame_index: int = 0
    root_frame_index: int = 0
    proof: object = None
    risk_profile: str = 'full_visual'

    def diagnostics(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'proof'}


def risk_signals(history, context, m2):
    """M1 continuity and M2 refresh are risk covariates, never token mappings."""
    history = history or {}; m2 = m2 or {}
    bound = m2.get('cache_context', {})
    valid_m2 = bool(m2.get('execution_completed')) and all(
        bound.get(k) == context.get(k) for k in ('scene_id', 'frame_id', 'timestamp'))
    stages = m2.get('stages', [])
    total = sum(s.get('total_windows', 0) for s in stages)
    executed = sum(s.get('executed_windows', 0) for s in stages)
    refresh = min(1., executed / total) if valid_m2 and total else 1.
    age = max([s.get('cache_age_frames', 0) or 0 for s in stages] or [0])
    reasons = tuple(sorted({str(s.get('full_refresh_reason')) for s in stages if s.get('full_refresh_reason')}))
    return dict(m1_state_valid=bool(history.get('state_valid')), m2_context_valid=valid_m2,
        m2_refresh_ratio=refresh, m2_cache_age_frames=age, m2_full_refresh_reasons=reasons,
        m2_mapping_available=False, m2_mapping_reason='m2_region_mapping_unavailable')


class ReusePlanner:
    def __init__(self, scheduler, calibration_threshold=None, calibration_max_visual_tokens=None):
        self.scheduler = scheduler
        # Only the offline calibrator supplies a trial threshold. It still uses
        # identical statistics, source limits, proofs, executor and decode.
        self.calibration_threshold = calibration_threshold
        self.calibration_max_visual_tokens = calibration_max_visual_tokens

    def plan(self, question, expanded, cache, history, context, m2_diagnostics=None, overhead_ms=0.):
        tick = perf_counter()
        task, profile, relevant = task_profile(question)
        signals = risk_signals(history, context, m2_diagnostics)
        details = dict(task_type=task, task_profile=profile, relevant_groups=relevant,
            layout_signature=expanded.layout_signature, layout_permutation=GROUPS,
            strategy_version=WSDR_VERSION, **signals)
        entry = cache.validate_storage()
        frame_index = context.get('frame_index', 0)
        def none(reason):
            return ReuseDecision(decision_reason=reason, frame_index=frame_index,
                root_frame_index=frame_index), dict(details, planning_ms=(perf_counter()-tick)*1000)
        if entry is None:
            details['matched_prefix_tokens'] = 0
            return none('cold_or_invalid_cache')
        exact, legal, blocks = prefix_statistics(entry.expanded, expanded, self.scheduler.config.block_size)
        details.update(matched_prefix_tokens=exact, structural_prefix_tokens=legal, block_changes=blocks)
        same_frame = entry.expanded.source[:2] == expanded.source[:2]
        age = frame_index - entry.root_frame_index
        if age < 0:
            raise ValueError('Cache frame order is invalid')
        inherited = entry.approx_reuse_depth > 0
        # Bound a change of question dependencies without pretending the
        # inherited approximate KV became exact when current embeddings match.
        old_weights=WEIGHTS[entry.risk_profile]
        inherited_risk=entry.quality_risk*max(a/b for a,b in zip(WEIGHTS[profile],old_weights))
        depth = entry.approx_reuse_depth + int(not same_frame)
        if not inherited:
            depth = int(not same_frame)
        candidates = []
        if exact:
            candidates.append(dict(length=exact, reuse_type='bounded_approx' if inherited else 'exact',
                risk=inherited_risk, rms=0., max_token=0., age=age,
                depth=entry.approx_reuse_depth if same_frame else depth))
        # Same-frame requests ALWAYS require exact current input. Inherited
        # approximate KV retains its label/risk even with equal embeddings.
        allow_approx = (not same_frame and signals['m1_state_valid'] and bool(blocks)
                        and self.scheduler.config.reuse_strategy != 'exact_prefix')
        if allow_approx:
            weights = dict(zip(GROUPS, WEIGHTS[profile]))
            factor = 1. + .25*signals['m2_refresh_ratio'] + .05*min(4, signals['m2_cache_age_frames'])
            if not signals['m2_context_valid']:factor += .25
            rms = max_token = 0.
            risk = inherited_risk
            rule=self.scheduler.policy.get('profiles',{}).get(profile,{}).get('approx',{})
            visual_cap=self.calibration_max_visual_tokens if self.calibration_threshold is not None else rule.get('max_visual_tokens')
            first_visual=expanded.visual_group_spans[0][1]
            for b in blocks:
                if visual_cap is not None and b['end']>first_visual+visual_cap:break
                if b['start'] > legal or b['end'] > legal:
                    break
                # No block after a text mismatch, hole, or layout boundary can
                # be stitched into a causal prefix. legal certifies the text.
                rms = max(rms, b['rms']); max_token = max(max_token, b['max_token'])
                risk = max(risk, inherited_risk + weights[b['group']]*factor*(b['rms']+.25*b['max_token']))
                if not math.isfinite(risk):break
                candidates.append(dict(length=b['end'], reuse_type='bounded_approx', risk=risk,
                    rms=rms, max_token=max_token, age=age, depth=depth))
        # Only an exact full cached request owns final-token logits. An
        # approximate full match must recompute the final token's logits.
        for c in candidates:
            if c['length'] == expanded.length and (c['reuse_type'] != 'exact' or entry.expanded.length != expanded.length):
                c['length'] -= 1
        context = dict(context, task_profile=profile, input_tokens=expanded.length,
            layout_signature=expanded.layout_signature, dtype=str(expanded.embeddings.dtype))
        policy = self.scheduler.policy
        trial = self.calibration_threshold
        if trial is not None:
            usable = [c for c in candidates if c['length'] >= self.scheduler.config.min_prefix_tokens
                and c['risk'] <= trial and c['age'] <= 2 and c['depth'] <= 1]
            chosen = max(usable, key=lambda c:c['length'], default=None)
            reason = 'offline_calibration_trial'
            limits = dict(risk_limit=trial, max_rms=2., max_token_change=2.)
            policy_id = 'offline_trial:' + str(trial)
        else:
            chosen, reason = self.scheduler.choose(candidates, context, overhead_ms+(perf_counter()-tick)*1000)
            limits = policy.get('profiles', {}).get(profile, {}).get('approx', {})
            policy_id = self.scheduler.policy_id
        details.update(planning_ms=(perf_counter()-tick)*1000, candidate_prefix_tokens=[c['length'] for c in candidates])
        if chosen is None:
            return ReuseDecision(decision_reason=reason, frame_index=frame_index, root_frame_index=frame_index), details
        c = chosen
        details['estimated_quality_loss']=c.get('quality_loss')
        details['quality_budget']=self.scheduler.config.quality_budget
        proof = PrefixProof(input_revision(expanded), cache.revision, c['length'], policy_id,
            c['risk'], limits.get('risk_limit', c['risk']), c['rms'], limits.get('max_rms', c['rms']),
            c['max_token'], limits.get('max_token_change', c['max_token']),
            2 if trial is not None else policy.get('max_cache_age_frames', 0),
            1 if trial is not None else policy.get('max_approx_depth', 0))
        root = entry.root_frame_index if c['reuse_type']=='bounded_approx' else frame_index
        return ReuseDecision(c['length'], c['reuse_type'], c['risk'], c.get('saved_ms', 0.),
            str(entry.computed_source[1]) if len(entry.computed_source)>1 else '', c['age'],
            c['depth'] if c['reuse_type']=='bounded_approx' else 0, reason, frame_index, root, proof, profile), details
