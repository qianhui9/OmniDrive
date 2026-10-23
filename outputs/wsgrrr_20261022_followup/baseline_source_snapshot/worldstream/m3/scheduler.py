"""Independent exact-prefill and historical-text evidence; no approximate policy."""
import json
import math
from pathlib import Path


def read_artifact(path):
    if not path:return {}
    try:
        value=json.loads(Path(path).read_text())
        if not isinstance(value,dict) or not isinstance(value.get('approved',[]),list):return {}
        return value if all(isinstance(r,dict) for r in value.get('approved',[])) else {}
    except (OSError,ValueError,TypeError):return {}


class Scheduler:
    def __init__(self,config):
        self.config=config
        self.costs=read_artifact(config.cost_table)
        self.quality=read_artifact(config.quality_certificate)
        self.state_quality=read_artifact(config.state_quality_certificate)

    @staticmethod
    def compatible(row,context,strategy):
        return (row.get('strategy')==strategy and context.get('model_version')
            and row.get('model_version')==context['model_version'] and context.get('execution_key')
            and row.get('execution_key')==context['execution_key'] and row.get('evidence_sha256')
            and row.get('protocol_sha256'))

    def prefix_allowed(self,expanded,prefix,context,overhead_ms):
        if prefix<self.config.min_prefix_tokens:return False,'short_or_missing_prefix'
        def same(row):
            return (self.compatible(row,context,'exact_prefill') and row.get('input_tokens')==expanded.length
                and row.get('prefix_tokens')==prefix and row.get('dtype')==str(expanded.embeddings.dtype)
                and row.get('generation_key')==context.get('generation_key'))
        quality=any(same(r) and r.get('tokens_equal') is True and r.get('rng_equal') is True
            and r.get('scores_close') is True and r.get('heldout_samples',0)>0 for r in self.quality.get('approved',[]))
        cost=any(same(r) and type(r.get('baseline_ms')) in (int,float) and math.isfinite(r['baseline_ms'])
            and type(r.get('optimized_ms')) in (int,float) and math.isfinite(r['optimized_ms'])
            and r['baseline_ms']>r['optimized_ms']+overhead_ms for r in self.costs.get('approved',[]))
        return (True,'exact_prefill_quality_and_cost_verified') if quality and cost else (False,'prefix_evidence_unavailable')

    def state_allowed(self,retrieval,context):
        for r in self.state_quality.get('approved',[]):
            if (self.compatible(r,context,'verified_history_augment') and r.get('task')==retrieval['task']
                    and retrieval.get('m1_protocol_sha256')
                    and r.get('m1_protocol_sha256')==retrieval['m1_protocol_sha256']
                    and r.get('current_visual_complete') is True and r.get('frozen_targets_pass') is True
                    and r.get('heldout_samples',0)>0 and r.get('heldout_scenes') and r.get('calibration_scenes')
                    and not set(r['heldout_scenes'])&set(r['calibration_scenes'])):
                return True,'independent_historical_context_quality_verified'
        return False,'historical_context_quality_unverified'
