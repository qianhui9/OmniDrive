"""Measured lookup tables and frozen task certificates, no weighted scoring."""
import json
from pathlib import Path


def read_artifact(path):
    if not path:
        return {}
    with Path(path).open() as handle:
        return json.load(handle)


class Scheduler:
    def __init__(self, config):
        self.config = config
        self.costs = read_artifact(config.cost_table)
        self.quality = read_artifact(config.quality_certificate)

    def compact_allowed(self, selection, context):
        if not selection['approximate']:
            return False, selection['reason']
        if self.config.offline_experiment:
            return True, 'offline_approximation_experiment'
        if selection['task'] in ('unknown', 'global_safety'):
            return False, 'uncertified_or_safety_task'
        certificate = self.quality
        key = (context.get('model_version'), selection['method'], self.config.keep_ratio, selection['task'])
        for row in certificate.get('approved', []):
            if (row.get('model_version'), row.get('selector'), row.get('keep_ratio'), row.get('task')) == key:
                if (key[0] and row.get('frozen_targets_pass') is True and row.get('heldout_samples', 0) > 0
                        and row.get('execution_key') == context.get('execution_key') and context.get('execution_key')
                        and row.get('protocol_sha256') and row.get('evidence_sha256')):
                    return True, 'heldout_quality_certificate'
        return False, 'no_frozen_task_quality_certificate'

    def capture_allowed(self, expanded, context, questions):
        """Auto capture only when measured amortization includes the cold store."""
        for row in self.costs.get('approved', []):
            if (context.get('model_version') and row.get('model_version') == context['model_version']
                    and row.get('execution_key') == context.get('execution_key') and context.get('execution_key')
                    and row.get('input_tokens') == expanded.length
                    and row.get('cold_capture_profitable') is True
                    and row.get('min_questions', float('inf')) <= questions
                    and row.get('exact_greedy_pass') is True and row.get('evidence_sha256')):
                return True
        return False

    def compact_cost_allowed(self, selection, context):
        for row in self.costs.get('approved', []):
            if (row.get('strategy') == 'compact' and context.get('model_version')
                    and row.get('model_version') == context['model_version']
                    and row.get('execution_key') == context.get('execution_key') and context.get('execution_key')
                    and row.get('selector') == selection['method'] and row.get('task') == selection['task']
                    and row.get('original_visual_tokens') == selection['original_tokens']
                    and row.get('selected_visual_tokens') == selection['selected_tokens']
                    and row.get('baseline_ms', 0) > row.get('optimized_including_overhead_ms', float('inf'))
                    and row.get('evidence_sha256')):
                return True
        return False

    def prefix_allowed(self, expanded, prefix, context, overhead_ms):
        if prefix < self.config.min_prefix_tokens:
            return False, 'short_or_missing_prefix'
        if self.config.mode in ('exact', 'combined') and self.config.offline_experiment:
            return True, 'explicit_exact_experiment'
        for row in self.costs.get('approved', []):
            if (row.get('model_version') == context.get('model_version') and context.get('model_version')
                    and row.get('execution_key') == context.get('execution_key') and context.get('execution_key')
                    and row.get('device') == str(expanded.embeddings.device)
                    and row.get('dtype') == str(expanded.embeddings.dtype)
                    and row.get('input_tokens') == expanded.length
                    and row.get('prefix_tokens') == prefix
                    and row.get('exact_greedy_pass') is True
                    and row.get('baseline_ms', 0) > row.get('optimized_ms', float('inf')) + overhead_ms
                    and row.get('evidence_sha256')):
                return True, 'measured_net_benefit'
        return False, 'no_compatible_profitable_cost_measurement'
