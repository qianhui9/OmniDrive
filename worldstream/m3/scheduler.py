"""Read one empirically fitted quality/latency policy, with no approval flags."""
import hashlib
import json
import math
from pathlib import Path

POLICY_VERSION = 'quality_bounded_wsdr_v1'


def reject_nonfinite(value):
    raise ValueError('Nonfinite policy value: '+value)


class Scheduler:
    def __init__(self, config):
        self.config = config
        self.policy = {}
        self.reason = 'reuse_policy_missing'
        if config.reuse_policy:
            try:
                raw = Path(config.reuse_policy).read_bytes()
                value = json.loads(raw,parse_constant=reject_nonfinite)
                if (not isinstance(value, dict) or value.get('version') != POLICY_VERSION
                        or not isinstance(value.get('profiles'), dict)):
                    raise ValueError('Invalid policy schema')
                self.policy = value
                self.policy_id = hashlib.sha256(raw).hexdigest()
                self.reason = 'policy_loaded'
            except (OSError, ValueError, TypeError):
                self.reason = 'reuse_policy_missing_or_invalid'
        if not self.policy:
            self.policy_id = ''

    def compatible(self, context):
        binding = self.policy.get('binding', {})
        keys = ('model_version', 'weights_sha256', 'execution_key', 'generation_key',
                'layout_signature', 'dtype')
        return (self.policy.get('block_size') == self.config.block_size
                and all(context.get(k) and context[k] == binding.get(k) for k in keys))

    def quality_rejection(self,evidence):
        """The same evidence checks serve selection and offline explanations."""
        if not evidence:return 'quality_evidence_missing'
        if evidence.get('calibration_samples',0)<=0:return 'calibration_samples_missing'
        if evidence.get('validation_samples',0)<=0:return 'validation_samples_missing'
        loss=evidence.get('quality_loss_upper')
        if type(loss) not in (int,float) or not math.isfinite(loss):return 'quality_loss_unavailable'
        if loss>self.config.quality_budget:return 'quality_loss_exceeds_budget'
        if evidence.get('critical_regressions',0)>0:return 'critical_regressions'
        return None

    def needs_approximate_statistics(self,profile):
        # An unavailable or quality-rejected approximate rule cannot consume
        # visual RMS blocks. Exact LCP and all lifecycle checks remain active.
        evidence=self.policy.get('profiles',{}).get(profile,{}).get('approx',{})
        return self.quality_rejection(evidence) is None

    def evaluate_candidate(self,c,context,overhead_ms):
        """First failed predicate; no policy mutation or alternative admission."""
        if c['length']<self.config.min_prefix_tokens:return 'prefix_below_minimum',None,None
        approximate=c['reuse_type']=='bounded_approx'
        rule=self.policy.get('profiles',{}).get(context['task_profile'],{})
        evidence=rule.get('approx' if approximate else 'exact',{})
        reason=self.quality_rejection(evidence)
        if reason:return reason,None,evidence.get('quality_loss_upper')
        if approximate:
            for value,limit,reason in ((c['risk'],evidence.get('risk_limit',-1),'risk_limit'),
                    (c['rms'],evidence.get('max_rms',-1),'rms_limit'),
                    (c['max_token'],evidence.get('max_token_change',-1),'token_change_limit'),
                    (c['age'],self.policy.get('max_cache_age_frames',0),'cache_age_limit'),
                    (c['depth'],self.policy.get('max_approx_depth',0),'approx_depth_limit')):
                if value>limit:return reason,None,evidence['quality_loss_upper']
        cost=evidence.get('cost',{});slope=cost.get('net_saved_ms_per_token')
        if type(slope) not in (float,int) or not math.isfinite(slope):return 'cost_slope_invalid',None,evidence['quality_loss_upper']
        if not cost.get('samples'):return 'cost_samples_missing',None,evidence['quality_loss_upper']
        if not cost.get('min_input_tokens',0)<=context['input_tokens']<=cost.get('max_input_tokens',-1):
            return 'input_tokens_out_of_range',None,evidence['quality_loss_upper']
        if c['length']<cost.get('min_reuse_tokens',0):return 'reuse_tokens_below_range',None,evidence['quality_loss_upper']
        if c['length']>cost.get('max_reuse_tokens',0):return 'reuse_tokens_above_range',None,evidence['quality_loss_upper']
        saved=slope*c['length']-cost.get('fixed_overhead_ms',0.)-overhead_ms
        return ('eligible' if saved>0 else 'nonpositive_net_saving'),saved,evidence['quality_loss_upper']

    def choose(self,candidates,context,overhead_ms):
        if not self.policy:return None,self.reason
        if not self.compatible(context):return None,'reuse_policy_binding_mismatch'
        if not self.policy['profiles'].get(context['task_profile']):return None,'task_profile_not_calibrated'
        best=None
        for c in candidates:
            reason,saved,loss=self.evaluate_candidate(c,context,overhead_ms)
            if reason=='eligible' and (best is None or saved>best['saved_ms']):
                best=dict(c,saved_ms=saved,quality_loss=loss)
        return best,'quality_latency_minimum' if best else 'no_quality_bounded_net_gain'

    def explain(self,candidates,context,overhead_ms):
        """Offline observer only. Formal selection does not allocate these rows."""
        rows=[]
        for c in candidates:
            if not self.policy:reason,saved,loss=self.reason,None,None
            elif not self.compatible(context):reason,saved,loss='reuse_policy_binding_mismatch',None,None
            elif not self.policy['profiles'].get(context['task_profile']):reason,saved,loss='task_profile_not_calibrated',None,None
            else:reason,saved,loss=self.evaluate_candidate(c,context,overhead_ms)
            evidence=self.policy.get('profiles',{}).get(context['task_profile'],{}).get('approx' if c['reuse_type']=='bounded_approx' else 'exact',{})
            cost=evidence.get('cost',{})
            predicted=(cost['net_saved_ms_per_token']*c['length']-cost.get('fixed_overhead_ms',0.)-overhead_ms
                if type(cost.get('net_saved_ms_per_token')) in (float,int) else None)
            rows.append(dict(candidate=dict(c),first_rejection=None if reason=='eligible' else reason,
                estimated_saved_ms=saved,unapproved_cost_prediction_ms=predicted,
                quality_loss_upper=loss,quality_budget=self.config.quality_budget,
                calibrated_profile=bool(evidence),rule=evidence,overhead_ms=overhead_ms))
        return rows

    def history_enabled(self, context):
        return self.compatible(context) and self.policy.get('history_text') is True
