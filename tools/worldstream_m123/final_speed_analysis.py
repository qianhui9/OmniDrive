"""Scene-clustered analysis of frozen free/fixed full-request measurements."""
import argparse
from pathlib import Path
import numpy as np
from tools.worldstream_m2.global_residual_benchmark import read,rows
from tools.worldstream_m2.global_residual_analysis import compare_quality,service_summary,write
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.final_speed_experiment import validate


KINDS=('native','rms','risk0')


def name(mode,repeat,kind):return f'{mode}_r{repeat}_{kind}'


def cluster_bootstrap(pairs,plan):
    scenes=sorted({p['scene_id'] for p in pairs})
    means=np.array([np.mean([p['saved_ms'] for p in pairs if p['scene_id']==s]) for s in scenes])
    rng=np.random.RandomState(plan['bootstrap']['seed'])
    sample=rng.choice(means,(plan['bootstrap']['repetitions'],len(means)),replace=True).mean(1)
    low,high=np.quantile(sample,[.025,.975])
    return dict(mean_saved_ms=float(means.mean()),lower=float(low),upper=float(high),independent_clusters=len(means),
        scene_means={s:float(v) for s,v in zip(scenes,means)},scope='Resample scenes after averaging all repeated frame pairs within each scene; repetitions are not independent scenes')


def summarize_run(root,run,formal):
    ff=rows(root/run/'per_frame_metrics.jsonl');qq=rows(root/run/'per_question_metrics.jsonl')
    result=service_summary(root/run,formal)
    result.update(**{key:distribution(r[key] for r in ff) for key in ('model_ms','preprocessing_ms','transfer_and_collate_ms')})
    result['stage_spans']={k:distribution(f['stages'].get(k,{}).get('gpu_span_ms',0.) for f in ff) for k in ('detection','map','prefill','decode')}
    result['actual_decode_forwards']=sum(f['stages'].get('decode',{}).get('calls',0) for f in ff)
    result['per_scene']={s:dict(request_ms=distribution(f['request_ms'] for f in ff if f['scene_id']==s),model_ms=distribution(f['model_ms'] for f in ff if f['scene_id']==s)) for s in sorted({f['scene_id'] for f in ff})}
    if formal:result['risk_decision_wall_ms']=distribution(f['m2']['global_decision_ms'] for f in ff)
    if ff[0].get('fixed_tokens'):
        result['fixed_selection_overhead']={key:distribution(f['fixed_tokens'][key] for f in ff) for key in ('selector_enqueue_ms','selector_cuda_span_ms')}
    return result


def paired(root,mode,candidate,reference,plan):
    pairs=[];rounds=[]
    for repeat in range(3):
        nn=rows(root/name(mode,repeat,reference)/'per_frame_metrics.jsonl');ff=rows(root/name(mode,repeat,candidate)/'per_frame_metrics.jsonl')
        if [(v['scene_id'],v['frame_id']) for v in nn]!=[(v['scene_id'],v['frame_id']) for v in ff]:raise ValueError('Unpaired requests')
        group=[dict(repeat=repeat,scene_id=a['scene_id'],frame_id=a['frame_id'],reference_ms=a['request_ms'],candidate_ms=b['request_ms'],
            saved_ms=a['request_ms']-b['request_ms'],reference_model_ms=a['model_ms'],candidate_model_ms=b['model_ms'],model_saved_ms=a['model_ms']-b['model_ms']) for a,b in zip(nn,ff)]
        pairs+=group
        a=distribution(p['reference_ms'] for p in group);b=distribution(p['candidate_ms'] for p in group)
        rounds.append(dict(reference=a,candidate=b,speedup=a['mean']/b['mean'],saved_ms=a['mean']-b['mean']))
    ci=cluster_bootstrap(pairs,plan)
    model_ci=cluster_bootstrap([dict(p,saved_ms=p['model_saved_ms']) for p in pairs],plan)
    aggregate=dict(reference=distribution(p['reference_ms'] for p in pairs),candidate=distribution(p['candidate_ms'] for p in pairs),saved_ms=distribution(p['saved_ms'] for p in pairs),model_saved_ms=distribution(p['model_saved_ms'] for p in pairs))
    return dict(rounds=rounds,aggregate=aggregate,speedup=aggregate['reference']['mean']/aggregate['candidate']['mean'],
        positive_request_fraction=sum(p['saved_ms']>0 for p in pairs)/len(pairs),negative_requests=sum(p['saved_ms']<0 for p in pairs),
        cluster_bootstrap=ci,model_cluster_bootstrap=model_ci,
        model_speedup=float(np.mean([p['reference_model_ms'] for p in pairs])/np.mean([p['candidate_model_ms'] for p in pairs])),
        per_scene_saved={s:distribution(p['saved_ms'] for p in pairs if p['scene_id']==s) for s in plan['scenes']},
        stable_net_speedup=ci['lower']>0 and all(v['saved_ms']>0 for v in rounds) and all(v>0 for v in ci['scene_means'].values()),pairs=pairs)


def analyze(root):
    from tools.worldstream_m2.production_acceptance import validate_rows
    plan=validate(root);expected=read(root/'sequence.json')['tokens'];baseline=rows(root/'free_r0_native/per_frame_metrics.jsonl')
    inputs=[f['input_tensor_digest'] for f in baseline]
    fixed_reference=read(root/'fixed_reference.json');audits={};runs={mode:{} for mode in ('free','fixed')}
    for mode in ('free','fixed'):
        for repeat in range(3):
            for kind in KINDS:
                run=name(mode,repeat,kind);ff=rows(root/run/'per_frame_metrics.jsonl');qq=rows(root/run/'per_question_metrics.jsonl')
                if read(root/run/'status.json').get('integrity')!='PASS' or read(root/run/'checkpoint_validation.json')['status']!='PASS':
                    raise ValueError('Checkpoint or scene ledger integrity failed')
                if [f['frame_id'] for f in ff]!=expected or [f['input_tensor_digest'] for f in ff]!=inputs:raise ValueError('Input tensors or declared sample order differ')
                if any(f.get('shadow_suffix_labels') or f.get('risk_equivalence_diagnostic') for f in ff):raise ValueError('Diagnostic work in headline measurement')
                if any(f['generation_workload']!=('free_generation' if mode=='free' else 'fixed_reference_tokens') for f in ff):raise ValueError('Generation boundary changed')
                if kind!='native':
                    from worldstream.m2 import M2Config
                    policy=M2Config(**read(root/(kind+'.json')))
                    for f in ff:
                        if (list(f['m2']['global_risk_weights'])!=list(policy.global_risk_weights)
                                or list(f['m2']['global_risk_scales'])!=list(policy.global_risk_scales)
                                or f['m2']['global_risk_threshold']!=policy.risk_threshold
                                or f['m2']['global_anchor_interval']!=3):raise ValueError('Executed policy differs from frozen candidate')
                    audit=validate_rows(ff,require_reuse=False)
                    if audit['status']!='PASS':raise ValueError('Formal execution integrity failed')
                    audits[run]=audit
                if mode=='fixed':
                    if any(q['quality'] is not None or q['reference'] is not None for q in qq):raise ValueError('Forced answers were quality-scored')
                    for f in ff:
                        reference=fixed_reference['frames'][f['frame_id']]
                        t=sum(len(q['tokens']) for q in reference);n=len(reference)
                        if f['fixed_tokens']['generated_steps']!=t or f['fixed_tokens']['actual_decode_forward_calls']!=t-n:raise ValueError('Unequal real decode workload')
                        if f['stages']['decode']['calls']!=t-n:raise ValueError('Actual model hooks disagree with fixed history')
                    for q in qq:
                        ref=fixed_reference['frames'][q['frame_id']][q['question_index']]
                        if q['question']!=ref['question'] or q['token_ids'][0][1:]!=ref['tokens']:raise ValueError('Generated fixed sequence mismatch')
                runs[mode][run]=summarize_run(root,run,kind!='native')
    quality={kind:compare_quality(root,'free_r0_native','free_r0_'+kind) for kind in ('rms','risk0')}
    comparisons={mode:{kind:paired(root,mode,kind,'native',plan) for kind in ('rms','risk0')} for mode in ('free','fixed')}
    comparisons['risk0_vs_rms']={mode:paired(root,mode,'risk0','rms',plan) for mode in ('free','fixed')}
    aggregate={mode:{} for mode in ('free','fixed')};repro={}
    for mode in ('free','fixed'):
        for kind in KINDS:
            ff=[f for i in range(3) for f in rows(root/name(mode,i,kind)/'per_frame_metrics.jsonl')]
            qq=[q for i in range(3) for q in rows(root/name(mode,i,kind)/'per_question_metrics.jsonl')]
            aggregate[mode][kind]=dict(**{key:distribution(f[key] for f in ff) for key in ('request_ms','model_ms','preprocessing_ms')},
                generated_tokens=sum(q['generated_tokens'] for q in qq),actual_decode_forwards=sum(f['stages']['decode']['calls'] for f in ff),
                input_tokens=sum(q['input_tokens'] for q in qq),
                effective_prefill_tokens=sum((q.get('reuse_diagnostics') or {}).get('effective_prefill_tokens',q['input_tokens']) for q in qq),
                kv_reused_tokens=sum(q['reused_prefix_length'] for q in qq),unique_reused_requests=len({(q['frame_id'],q['question']) for q in qq if q['reused_prefix_length']}),
                peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff),
                throughput_fps=1000*len(ff)/sum(f['request_ms'] for f in ff),
                visual_cuda_span_ms=distribution(f['stages']['eva_native' if kind=='native' else 'm2_total']['gpu_span_ms'] for f in ff),
                stages={stage:distribution(f['stages'].get(stage,{}).get('gpu_span_ms',0.) for f in ff) for stage in ('m1_precheck','m1_commit','detection','map','prefill','decode')},
                m1_ms=distribution(f['m1_commit_timings_ms']['m1_total_including_adapters_ms'] for f in ff) if kind!='native' else None,
                per_scene={s:distribution(f['request_ms'] for f in ff if f['scene_id']==s) for s in plan['scenes']})
            initial=rows(root/name('free',0,kind)/'per_frame_metrics.jsonl')
            repro[mode+'_'+kind]=all([f['output_digest'] for f in rows(root/name(mode,i,kind)/'per_frame_metrics.jsonl')]==[f['output_digest'] for f in initial] for i in range(3))
    if not all(repro.values()):raise ValueError('Perception outputs changed across repeats or fixed/free generation')
    eq=rows(root/'equivalence_development/risk_equivalence.jsonl')
    if len(eq)!=32 or not all(all(r['checks'].values()) for r in eq):raise ValueError('Pruning equivalence failed')
    teacher=rows(root/'residual_labels_new/per_frame_metrics.jsonl');actual=rows(root/'free_r0_risk0/per_frame_metrics.jsonl')
    if [f['output_digest'] for f in teacher]!=[f['output_digest'] for f in actual]:raise ValueError('Offline label run has a different real trajectory')
    labels={(q['frame_id'],q['anchor_frame_id']):q for q in rows(root/'residual_labels_new/residual_labels.jsonl')}
    feature_errors=[dict(labels[(f['frame_id'],f['m2']['global_anchor_frame_id'])]) for f in actual if f['m2']['approx_global_residual_reuse']]
    losses={k:[quality[k]['metrics'][m]['loss'] for m in ('mAP','NDS','DET_l','Token_F1')] for k in ('rms','risk0')}
    cost={k:aggregate['fixed'][k]['model_ms']['mean'] for k in ('rms','risk0')}
    def dominates(a,b):return bool(all(x<=y for x,y in zip(losses[a],losses[b])) and cost[a]<=cost[b] and (any(x<y for x,y in zip(losses[a],losses[b])) or cost[a]<cost[b]))
    pareto=dict(rms_dominates_risk0=dominates('rms','risk0'),risk0_dominates_rms=dominates('risk0','rms'),
        scope='Four natural task losses vs measured fixed-token model mean; no matched-compute retuning or inference from different scene sets')
    accepted=bool(quality['risk0']['within_quality_budgets'] and comparisons['fixed']['risk0']['stable_net_speedup']
        and comparisons['fixed']['risk0']['model_cluster_bootstrap']['lower']>0 and not pareto['rms_dominates_risk0'])
    result=dict(plan=plan,quality=quality,runs=runs,aggregate=aggregate,comparisons=comparisons,pareto=pareto,
        risk0_upgrade_criteria_pass=accepted,perception_outputs_equal_across_repeats_and_generation_modes=repro,
        pruning_equivalence_all_checks_pass=True,all_input_tensors_equal=True,
        exact_feature_errors=feature_errors,feature_error_summary=distribution(e['final_relative_rms'] for e in feature_errors),
        independent_frame_count=48,independent_scene_count=6,replicates=3)
    write(root/'execution_audits.json',audits);write(root/'summary.json',result)
    write(root/'status.json',dict(status='UPGRADE_ELIGIBLE' if accepted else 'GOAL_NOT_MET',implementation_complete=True,
        quality_pass=quality['risk0']['within_quality_budgets'],fixed_tokens_stable_net_speedup=comparisons['fixed']['risk0']['stable_net_speedup'],
        upgrade_criteria_pass=accepted,formal_default_update_pending=accepted))
    from tools.worldstream_m123.final_speed_report import report
    report(root)
    print('FINAL fixed speedup',comparisons['fixed']['risk0']['speedup'],'upgrade eligible',accepted,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();analyze(Path(a.root))
