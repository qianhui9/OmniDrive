"""Explain observed scheduler rejections and GPU events without changing policy."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import math
from tools.worldstream_m2.global_residual_benchmark import read, rows
from tools.worldstream_m2.global_residual_analysis import write


def rejection(c, trace, policy):
    cfg=trace['config'];context=trace['scheduler_context']
    if not trace['policy_compatible']:return 'policy_binding_mismatch'
    rule=policy['profiles'].get(context['task_profile'])
    if not rule:return 'task_profile_not_calibrated'
    if c['length']<cfg['min_prefix_tokens']:return 'below_min_prefix'
    approximate=c['reuse_type']=='bounded_approx';e=rule.get('approx' if approximate else 'exact',{})
    loss=e.get('quality_loss_upper')
    if e.get('calibration_samples',0)<=0 or e.get('validation_samples',0)<=0:return 'missing_calibration_or_validation_evidence'
    if type(loss) not in (int,float) or not math.isfinite(loss):return 'invalid_quality_loss_evidence'
    if loss>cfg['quality_budget']:return 'quality_loss_above_internal_budget'
    if e.get('critical_regressions',0)>0:return 'critical_regressions_in_policy_evidence'
    if approximate:
        for key,bound in [('risk','risk_limit'),('rms','max_rms'),('max_token','max_token_change')]:
            if c[key]>e.get(bound,-1):return key+'_above_policy'
        if c['age']>policy.get('max_cache_age_frames',0):return 'cache_age'
        if c['depth']>policy.get('max_approx_depth',0):return 'approx_depth'
    cost=e.get('cost',{});slope=cost.get('net_saved_ms_per_token')
    if type(slope) not in (int,float) or not math.isfinite(slope) or not cost.get('samples'):return 'cost_evidence_missing'
    if not cost.get('min_input_tokens',0)<=context['input_tokens']<=cost.get('max_input_tokens',-1):return 'input_length_outside_measured_cost_range'
    if not cost.get('min_reuse_tokens',0)<=c['length']<=cost.get('max_reuse_tokens',0):return 'reuse_length_outside_measured_cost_range'
    if slope*c['length']-cost.get('fixed_overhead_ms',0)-trace['overhead_ms']<=0:return 'nonpositive_net_saved_ms'
    return 'eligible'


def kernels(path):
    if not path.exists():return None
    events=read(path)['traceEvents'];totals=Counter();categories=Counter();count=Counter()
    for e in events:
        if e.get('ph')!='X' or e.get('cat')!='kernel':continue
        name=e['name'];dur=e['dur']/1000;totals[name]+=dur;count[name]+=1
        low=name.lower()
        kind=('matrix_multiply' if any(x in low for x in ('gemm','gemv','cutlass')) else
              'attention' if any(x in low for x in ('attention','softmax','fmha')) else
              'normalization' if any(x in low for x in ('norm','rms')) else
              'copy_concat_index' if any(x in low for x in ('copy','catarray','index','gather')) else 'other_elementwise')
        categories[kind]+=dur
    return dict(top_kernels=[dict(name=k,total_ms=v,calls=count[k]) for k,v in totals.most_common(15)],
        category_kernel_ms=dict(categories),scope='Sum of kernel durations, not additive with host wall or overlapping stages')


def analyze(root):
    policy=read('outputs/final_route/m3_reuse_policy.json');traces=rows(root/'after_profile/decision_traces.jsonl');details=[];counts=Counter()
    for t in traces:
        rejected=[dict(candidate=c,first_rejection=rejection(c,t,policy)) for c in t.get('candidates',[])]
        counts.update(c['first_rejection'] for c in rejected)
        details.append(dict(frame_id=t['context']['frame_id'],scene_id=t['context']['scene_id'],question=t['question'],
            decision=t['decision'],cache=t['cache'],cache_validated=t['cache_validated'],policy_compatible=t.get('policy_compatible'),
            exact_prefix=t['details'].get('matched_prefix_tokens',0),structural_prefix=t['details'].get('structural_prefix_tokens',0),
            block_changes=t['details'].get('block_changes',[]),candidates=rejected))
    qq=rows(root/'after_profile/per_question_metrics.jsonl')
    result=dict(scope='Four diagnostic development frames; profiler timings excluded from service comparison',
        requests=len(traces),candidate_decisions=sum(bool(t.get('candidates')) for t in traces),candidate_count=sum(len(t.get('candidates',[])) for t in traces),
        decision_reasons=dict(Counter(t['decision']['decision_reason'] for t in traces)),first_rejection_counts=dict(counts),
        actual_reused_tokens=sum(q['reused_prefix_length'] for q in qq),effective_prefill_lengths=[q['effective_prefill_tokens'] for q in qq],
        internal_quality_budget=0.02,policy_sha256=__import__('hashlib').sha256(Path('outputs/final_route/m3_reuse_policy.json').read_bytes()).hexdigest(),
        records=details,decode_kernel_profile=kernels(root/'decode_control/decode_cuda.json'))
    natural=[]
    for i in range(3):
        path=root/f'r{i}_formal/per_question_metrics.jsonl'
        if not path.exists():continue
        for q in rows(path):
            d=q['reuse_diagnostics']
            natural.append(dict(repeat=i,scene_id=q['scene_id'],frame_id=q['frame_id'],question=q['question'],
                reason=d.get('decision_reason'),candidate_lengths=d.get('candidate_prefix_tokens',[]),
                exact_prefix_tokens=d.get('matched_prefix_tokens',0),structural_prefix_tokens=d.get('structural_prefix_tokens',0),
                accepted_reuse_tokens=q['reused_prefix_length'],effective_prefill_tokens=d['effective_prefill_tokens'],
                input_tokens=d['input_tokens'],risk_profile=d.get('risk_profile'),
                reuse_type=d.get('reuse_type'),estimated_quality_risk=d.get('estimated_quality_risk'),
                estimated_saved_ms=d.get('estimated_saved_ms'),cache_source_frame=d.get('cache_source_frame'),
                forward_prefill_lengths=d.get('forward_prefill_lengths'),
                frame_index=d.get('frame_index'),root_frame_index=d.get('root_frame_index'),
                cache_age_frames=d.get('cache_age_frames'),approx_reuse_depth=d.get('approx_reuse_depth'),
                cache_stored=d['cache_stored'],cache_invalidation_reason=d.get('cache_invalidation_reason'),
                current_prefill_source=d.get('computed_source'),m2_context_valid=d.get('m2_context_valid')))
    result['heldout_natural']=dict(requests=len(natural),candidate_requests=sum(bool(r['candidate_lengths']) for r in natural),
        candidate_count=sum(len(r['candidate_lengths']) for r in natural),decision_reasons=dict(Counter(r['reason'] for r in natural)),
        actual_reused_tokens=sum(r['accepted_reuse_tokens'] for r in natural),
        executed_reuse_requests=sum(r['accepted_reuse_tokens']>0 for r in natural),records=natural,
        scope='All natural heldout questions in the three timing repeats; detailed first-stage candidate rejection classification above is from diagnostic profile only')
    write(root/'m3_rejection_diagnostics.json',result)
    profiles={}
    for name in ('before_profile','after_profile'):
        data=read(root/name/'profile_output/cpu_summary.json')
        import pstats
        stats=pstats.Stats(str(root/name/'profile_output/cpu.prof')).stats
        # Wrapper and implementation share a name; do not add nested cumulative
        # times or call counts. Report each concrete function separately.
        concrete=[dict(file=k[0],line=k[1],name=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
            for k,v in stats.items() if k[2] in data]
        profiles[name]=dict(original_name_group_summary=data,concrete_functions=concrete,
            generation_and_sync_functions=[dict(file=k[0],line=k[1],name=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
                for k,v in stats.items() if any(s in k[2].lower() for s in ('sample','multinomial','synchronize','cumsum','masked_fill','item>'))],
            warning='Name-group sums may double count nested wrapper/implementation calls; use concrete_functions')
    write(root/'profile_comparison.json',profiles)
    bf,af=[rows(root/n/'per_frame_metrics.jsonl') for n in ('before_profile','after_profile')]
    bq,aq=[rows(root/n/'per_question_metrics.jsonl') for n in ('before_profile','after_profile')]
    checks=dict(frames_equal=[r['frame_id'] for r in bf]==[r['frame_id'] for r in af],
        output_digests_equal=[r['output_digest'] for r in bf]==[r['output_digest'] for r in af],
        answers_equal=[r['answer'] for r in bq]==[r['answer'] for r in aq],
        tokens_equal=[r['token_ids'] for r in bq]==[r['token_ids'] for r in aq])
    write(root/'real_request_equivalence.json',dict(checks=checks,frames=len(bf),
        scope='Same real four-frame inputs, weights, AR4 parameters and per-frame seeds; before/after optimization; profiled timings excluded'))
    if not all(checks.values()):raise ValueError('Real before/after optimization request outputs changed')
    print('KV diagnostics',result['decision_reasons'],result['first_rejection_counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();analyze(Path(a.root))
