"""Independent exact-prefill and historical-text evidence; no approximate policy."""
import json
import math
import hashlib
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
        if self.config.reuse_strategy=='wsdr_kv':
            return self.wsdr_allowed(expanded,prefix,context,overhead_ms)
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

    @staticmethod
    def _hashed_evidence(row):
        documents={}
        for kind in ('evidence','protocol'):
            path=row.get(kind+'_path');expected=row.get(kind+'_sha256')
            if not isinstance(path,str) or not isinstance(expected,str):return False
            try:
                digest=hashlib.sha256()
                with Path(path).open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1024**2),b''):digest.update(chunk)
                if digest.hexdigest()!=expected:return False
                documents[kind]=json.loads(Path(path).read_text())
            except (OSError,ValueError):return False
        evidence=documents['evidence']
        if not isinstance(evidence,dict) or evidence.get('schema')!='wsdr_kv_evidence_v1':return False
        binding=evidence.get('binding',{})
        if not isinstance(binding,dict):return False
        for key in ('strategy','strategy_version','model_version','execution_key','weights_sha256',
            'task_type','task_profile','layout_signature','input_tokens','prefix_tokens','reused_tokens','dtype','generation_key'):
            if key not in binding or binding[key]!=row.get(key):return False
        if evidence.get('measurements')!=row.get('measurements'):return False
        numerical=evidence.get('numerical',{})
        if not isinstance(numerical,dict) or any(numerical.get(k) is not True for k in ('tokens_equal','rng_equal','scores_close')):return False
        if not row.get('approved_by'):return False
        return True

    def wsdr_allowed(self,expanded,prefix,context,overhead_ms):
        if not context.get('m1_state_valid'):return False,'wsdr_m1_state_unavailable'
        if prefix<self.config.min_prefix_tokens:return False,'short_or_missing_prefix'
        fields=('strategy_version','task_type','task_profile','layout_signature','weights_sha256')
        def same(row):
            return (self.compatible(row,context,'wsdr_kv') and self._hashed_evidence(row)
                and all(context.get(k) and row.get(k)==context[k] for k in fields)
                and row.get('input_tokens')==expanded.length and row.get('prefix_tokens')==prefix
                and row.get('reused_tokens')==context.get('effective_reuse_length')
                and row.get('dtype')==str(expanded.embeddings.dtype)
                and row.get('generation_key')==context.get('generation_key'))
        quality=any(same(r) and r.get('tokens_equal') is True and r.get('rng_equal') is True
            and r.get('scores_close') is True and r.get('independent_task_quality') is True
            and r.get('task_quality_status')=='passed' and r.get('heldout_samples',0)>0 and r.get('task_metrics')
            for r in self.quality.get('approved',[]))
        if not quality:return False,'wsdr_quality_evidence_unavailable'
        def measured(row):
            measurements=row.get('measurements')
            return (isinstance(measurements,list) and len(measurements)>0
                and all(isinstance(m,dict) and m.get('sample_id')
                    and all(type(m.get(k)) in (int,float) and math.isfinite(m[k]) and m[k]>0
                        for k in ('baseline_ms','optimized_ms'))
                    and m['baseline_ms']>m['optimized_ms']+overhead_ms for m in measurements))
        cost=any(same(r) and measured(r) for r in self.costs.get('approved',[]))
        return (True,'wsdr_quality_and_cost_verified') if cost else (False,'wsdr_cost_evidence_unavailable')

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
