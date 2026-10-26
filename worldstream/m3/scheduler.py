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

    def choose(self, candidates, context, overhead_ms):
        if not self.policy:
            return None, self.reason
        if not self.compatible(context):
            return None, 'reuse_policy_binding_mismatch'
        rule = self.policy['profiles'].get(context['task_profile'])
        if not rule:
            return None, 'task_profile_not_calibrated'
        best = None
        for c in candidates:
            if c['length'] < self.config.min_prefix_tokens:
                continue
            approximate = c['reuse_type'] == 'bounded_approx'
            evidence = rule.get('approx' if approximate else 'exact', {})
            loss = evidence.get('quality_loss_upper')
            if (evidence.get('calibration_samples', 0) <= 0 or evidence.get('validation_samples', 0) <= 0
                    or type(loss) not in (int, float) or not math.isfinite(loss)
                    or loss > self.config.quality_budget or evidence.get('critical_regressions', 0) > 0):
                continue
            if approximate and (c['risk'] > evidence.get('risk_limit', -1)
                    or c['rms'] > evidence.get('max_rms', -1)
                    or c['max_token'] > evidence.get('max_token_change', -1)
                    or c['age'] > self.policy.get('max_cache_age_frames', 0)
                    or c['depth'] > self.policy.get('max_approx_depth', 0)):
                continue
            # Complete paired runtime wall measurements include branch, suffix,
            # cache update and affected decode; substitute current decision time.
            cost = evidence.get('cost', {})
            slope = cost.get('net_saved_ms_per_token')
            if (type(slope) not in (float, int) or not math.isfinite(slope)
                    or not cost.get('samples')
                    or not cost.get('min_input_tokens', 0) <= context['input_tokens'] <= cost.get('max_input_tokens', -1)
                    or c['length'] < cost.get('min_reuse_tokens', 0)
                    or c['length'] > cost.get('max_reuse_tokens', 0)):
                continue
            saved = slope * c['length'] - cost.get('fixed_overhead_ms', 0.) - overhead_ms
            if saved > 0 and (best is None or saved > best['saved_ms']):
                best = dict(c, saved_ms=saved, quality_loss=loss)
        return best, 'quality_latency_minimum' if best else 'no_quality_bounded_net_gain'

    def history_enabled(self, context):
        return self.compatible(context) and self.policy.get('history_text') is True
