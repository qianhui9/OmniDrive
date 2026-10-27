"""Trace first rejection predicates and separate Prefill/M3/E2E evidence."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import hashlib
import json
import sys
import numpy as np
import torch
from tools.worldstream_m2.global_residual_benchmark import read,rows
from tools.worldstream_m2.global_residual_analysis import compare_quality,service_summary
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality,quality_upper
from tools.worldstream_m123.final_speed_analysis import cluster_bootstrap
from tools.worldstream_m123.latency_controls import before_package
from worldstream.m3.config import M3Config
from worldstream.m3.scheduler import Scheduler

KINDS=('native','zero','reuse')


def diagnose(root,runs=('historical_current','single_before','single_after_diagnostic','multi_after_diagnostic')):
    scheduler=Scheduler(M3Config(reuse_policy=str(root/'frozen_m3_policy.json')))
    reports={};semantics=[]
    package=before_package(root,'m3');legacy_scheduler=sys.modules[package.__name__+'.scheduler'].Scheduler
    for run in runs:
        traced=rows(root/run/'decision_traces.jsonl');output=[];first=Counter();requests=Counter();candidates=Counter()
        actual={(q['frame_id'],q['question']):q for q in rows(root/run/'per_question_metrics.jsonl')}
        for t in traced:
            q=actual[(t['context']['frame_id'],t['question'])];d=q['reuse_diagnostics']
            evaluations=scheduler.explain(t.get('candidates',[]),t.get('scheduler_context',{}),t.get('overhead_ms',0.))
            if 'candidates' in t:
                a=legacy_scheduler(scheduler.config).choose(t['candidates'],t['scheduler_context'],t['overhead_ms'])
                b=scheduler.choose(t['candidates'],t['scheduler_context'],t['overhead_ms'])
                if a!=b:raise ValueError('Old/new Scheduler semantics differ')
                semantics.append(dict(run=run,frame_id=t['context']['frame_id'],question=t['question'],candidate_count=len(t['candidates']),old_and_new_selection_identical=True))
            reason='cold_cache' if t['cache'] is None else evaluations[0]['first_rejection'] if evaluations else t['decision']['decision_reason']
            first[str(reason)]+=1;requests.update(set(str(x['first_rejection']) for x in evaluations));candidates.update(str(x['first_rejection']) for x in evaluations)
            output.append(dict(scene_id=t['context']['scene_id'],frame_id=t['context']['frame_id'],question=t['question'],
                task_type=t['details']['task_type'],profile=t['details']['task_profile'],cache=t['cache'],cache_validated=t['cache_validated'],
                cache_invalidation=d['cache_invalidation_reason'],exact_prefix=t['details'].get('matched_prefix_tokens',0),
                legal_prefix=t['details'].get('structural_prefix_tokens',0),
                blocks=t.get('offline_full_statistics',{}).get('blocks',t['details'].get('block_changes',[])),
                candidate_evaluations=evaluations,first_candidate_reason=reason,actual_reuse_tokens=d['reused_tokens'],
                actual_prefill_forward_lengths=d['forward_prefill_lengths'],input_tokens=d['input_tokens'],
                layout_preparation_ms=d['layout_preparation_ms'],prefix_matching_ms=d['prefix_matching_ms'],
                cache_branch_enqueue_ms=d['cache_branch_cpu_enqueue_ms'],cache_update_enqueue_ms=d.get('cache_update_cpu_enqueue_ms',0.),
                m3_question_wall_ms=d['end_to_end_ms']))
        path=root/(run+'_rejection_details.jsonl')
        with path.open('w') as handle:
            for r in output:handle.write(json.dumps(r,allow_nan=False)+'\n')
        reports[run]=dict(requests=len(output),profiles=dict(Counter(x['profile'] for x in output)),
            first_candidate_reasons=dict(first),requests_with_each_candidate_rejection=dict(requests),candidate_rejection_counts=dict(candidates),
            requests_with_cache=sum(x['cache'] is not None for x in output),
            requests_with_exact_prefix_at_least32=sum(x['exact_prefix']>=32 for x in output),
            reused_tokens=sum(x['actual_reuse_tokens'] for x in output))
    # Replay the PRECISE old Risk0 statistics, without injecting new H2/M1 or
    # running a model. The old observed generic reason proves binding passed;
    # matching reproduction supplies the identical portable binding fields.
    oldroot=Path('outputs/m123_final_speed_validation_20261027_run01/free_r0_risk0')
    oldq=rows(oldroot/'per_question_metrics.jsonl');oldf={x['frame_id']:x for x in rows(oldroot/'per_frame_metrics.jsonl')}
    traces={x['context']['frame_id']:x for x in rows(root/'single_before/decision_traces.jsonl')}
    oldpackage=before_package(root,'m3');module=sys.modules[oldpackage.__name__+'.reuse_planner'];output=[];previous=None
    for q in oldq:
        d=q['reuse_diagnostics'];f=oldf[q['frame_id']];t=traces[q['frame_id']]
        context=dict(t['scheduler_context'] if 'scheduler_context' in t else t['context'])
        context.update(f['m2']['cache_context']);context['frame_index']=d['frame_index'];context.update(scheduler.policy['binding'])
        expanded=NS(length=d['input_tokens'],source=tuple(d['suffix_computed_source']),
            embeddings=NS(dtype=torch.float16),layout_signature=d['layout_signature'],
            visual_group_spans=(('object',35,291),('ego',291,292),('map',292,548)))
        entry=None
        if previous is not None and previous['scene_id']==q['scene_id']:
            p=previous['reuse_diagnostics']
            entry=NS(expanded=NS(source=tuple(p['suffix_computed_source']),length=p['input_tokens']),
                root_frame_index=p['root_frame_index'],approx_reuse_depth=0,quality_risk=0.,risk_profile=p['risk_profile'],computed_source=tuple(p['suffix_computed_source']))
        cache=NS(validate_storage=lambda:entry,revision='offline_not_executable');evaluations=[]
        original=scheduler.choose
        def choose(candidates,ctx,overhead):
            evaluations.extend(scheduler.explain(candidates,ctx,t.get('overhead_ms',overhead)))
            return original(candidates,ctx,t.get('overhead_ms',overhead))
        scheduler.choose=choose
        with patch.object(module,'prefix_statistics',return_value=(d['matched_prefix_tokens'],d.get('structural_prefix_tokens',0),d.get('block_changes',[]))),patch.object(module,'input_revision',return_value='offline_not_executable'):
            decision,_=module.ReusePlanner(scheduler).plan(q['question'],expanded,cache,d['state_retrieval'],context,f['m2'])
        scheduler.choose=original
        if decision.reuse_length!=d['reused_tokens'] or decision.decision_reason!=d['decision_reason']:raise ValueError('Original48 diagnostic does not reproduce')
        output.append(dict(scene_id=q['scene_id'],frame_id=q['frame_id'],question=q['question'],profile=d['task_profile'],
            original_observed_diagnostics=d,candidate_evaluations=evaluations,
            first_candidate_reason='cold_cache' if not evaluations else evaluations[0]['first_rejection'],
            scope='CPU statistics-only replay of original recorded Risk0 H2 block changes/M1 covariates; identical frozen policy; reproduced zero decision; no KV execution or performance claims'))
        previous=q
    with (root/'original48_rejection_details.jsonl').open('w') as h:
        for x in output:h.write(json.dumps(x,allow_nan=False)+'\n')
    reports['original48']=dict(requests=len(output),profiles=dict(Counter(x['profile'] for x in output)),
        first_candidate_reasons=dict(Counter(str(x['first_candidate_reason']) for x in output)),
        requests_with_each_candidate_rejection=dict(Counter(reason for x in output for reason in set(e['first_rejection'] for e in x['candidate_evaluations']))),
        all_recorded_zero_decisions_reproduced=True)
    write(root/'scheduler_semantics_equivalence.json',dict(status='PASS',actual_observed_candidate_sets=len(semantics),checks=semantics,scope='Old/new canonical selection under observed actual context and overhead; no policy modification'))
    changes=[]
    old_task=sys.modules[package.__name__+'.task_conditioner'].task_profile
    from worldstream.m3.task_conditioner import task_profile
    for name in ('historical_sequence24.json','single_sequence48.json','multi_sequence24.json'):
        questions=read(root/name).get('questions')
        if questions is None:
            questions=defaultdict(list)
            for q in rows(root/'single_before/per_question_metrics.jsonl'):questions[q['frame_id']].append(q['question'])
        for frame,qq in questions.items():
            for q in qq:
                a,b=old_task(q),task_profile(q)
                if a!=b:
                    if b[1]!='full_visual':raise ValueError('Task change creates new admission')
                    changes.append(dict(manifest=name,frame_id=frame,question=q,before=a,after=b))
    write(root/'task_classification_changes.json',dict(all_changes_more_conservative=True,changes=changes))
    write(root/'rejection_summary.json',reports);return reports


def reuse_counts(questions):
    groups=defaultdict(lambda:dict(requests=0,tokens=0));input_tokens=sum(q['input_tokens'] for q in questions);effective=0
    for q in questions:
        d=q.get('reuse_diagnostics') or {};n=q['reused_prefix_length'];effective+=q['input_tokens']-n
        if not n:continue
        same=d['cache_source_frame']==q['frame_id']
        key=('same_frame_exact' if same else 'cross_frame_exact') if d['reuse_type']=='exact' else ('same_frame_inherited_approx' if same else 'cross_frame_bounded_approx')
        groups[key]['requests']+=1;groups[key]['tokens']+=n
        if sum(d['forward_prefill_lengths'])!=d['effective_prefill_tokens'] or d['effective_prefill_tokens']!=q['input_tokens']-n:
            raise ValueError('Recorded KV did not reduce real prefill forward')
    return dict(groups=dict(groups),input_tokens=input_tokens,effective_prefill_tokens=effective,
        skipped_tokens=input_tokens-effective,skipped_fraction=(input_tokens-effective)/input_tokens,
        profiles=dict(Counter((q.get('reuse_diagnostics') or {}).get('task_profile','native') for q in questions)))


def frame_metric(frame,key):
    if key in ('request_ms','model_ms','preprocessing_ms','transfer_and_collate_ms'):return frame[key]
    if key=='non_model_ms':return frame['request_ms']-frame['model_ms']
    stage='m3_auto_total' if 'm3_auto_total' in frame['stages'] else 'native_generation'
    return frame['stages'].get('prefill' if key=='prefill_ms' else stage,{}).get('gpu_span_ms' if key=='prefill_ms' else 'host_wall_ms',0.)


def compare(root,workload,candidate,reference):
    result={}
    for metric in ('prefill_ms','m3_ms','model_ms','request_ms','preprocessing_ms','non_model_ms'):
        pairs=[];rounds=[]
        for repeat in range(3):
            a=rows(root/f'{workload}_fixed_r{repeat}_{reference}/per_frame_metrics.jsonl');b=rows(root/f'{workload}_fixed_r{repeat}_{candidate}/per_frame_metrics.jsonl')
            if [(x['scene_id'],x['frame_id']) for x in a]!=[(x['scene_id'],x['frame_id']) for x in b]:raise ValueError('Unpaired frames')
            aa=[frame_metric(x,metric) for x in a];bb=[frame_metric(x,metric) for x in b]
            pairs.extend(dict(scene_id=x['scene_id'],frame_id=x['frame_id'],repeat=repeat,saved_ms=v-w) for x,v,w in zip(a,aa,bb))
            rounds.append(dict(reference=distribution(aa),candidate=distribution(bb),saved_ms=float(np.mean(aa)-np.mean(bb)),speedup=float(np.mean(aa)/np.mean(bb))))
        ci=cluster_bootstrap(pairs,dict(bootstrap=dict(seed=20261027,repetitions=10000)))
        result[metric]=dict(rounds=rounds,speedup=float(np.mean([x['reference']['mean'] for x in rounds])/np.mean([x['candidate']['mean'] for x in rounds])),
            saved_ms=distribution(x['saved_ms'] for x in pairs),scene_bootstrap=ci,
            stable_net_speedup=ci['lower']>0 and all(x['saved_ms']>0 for x in rounds) and all(x>0 for x in ci['scene_means'].values()),
            positive_frame_fraction=sum(x['saved_ms']>0 for x in pairs)/len(pairs))
    return result


def analyze(root):
    from tools.worldstream_m2.production_acceptance import validate_rows
    plan=read(root/'experiment_plan.json');audits={};workloads={}
    if hashlib.sha256((root/'frozen_m3_policy.json').read_bytes()).hexdigest()!=plan['manifest_sha256']['frozen_m3_policy.json']:raise ValueError('Frozen experiment policy changed')
    rejection=diagnose(root)
    for workload in ('multi','single'):
        basef=rows(root/f'{workload}_free_r0_native/per_frame_metrics.jsonl');native_inputs=[f['input_tensor_digest'] for f in basef]
        zero=rows(root/f'{workload}_free_r0_zero/per_frame_metrics.jsonl');reuse=rows(root/f'{workload}_free_r0_reuse/per_frame_metrics.jsonl')
        if [f['output_digest'] for f in zero]!=[f['output_digest'] for f in reuse]:raise ValueError('M3 changed perception trajectory')
        reference=read(root/(workload+'_fixed_reference.json'));summaries={}
        for mode in ('free','fixed'):
            for kind in KINDS:
                allframes=[];allquestions=[]
                for repeat in (range(3) if mode=='fixed' else range(1)):
                    name=f'{workload}_{mode}_r{repeat}_{kind}';run=root/name;ff=rows(run/'per_frame_metrics.jsonl');qq=rows(run/'per_question_metrics.jsonl')
                    if read(run/'status.json')['integrity']!='PASS' or read(run/'checkpoint_validation.json')['status']!='PASS':raise ValueError('Execution/checkpoint integrity failed')
                    if [f['frame_id'] for f in ff]!=[f['frame_id'] for f in basef] or [f['input_tensor_digest'] for f in ff]!=native_inputs:raise ValueError('Actual inputs differ')
                    if kind!='native':
                        audit=validate_rows(ff,require_reuse=False)
                        if audit['status']!='PASS':raise ValueError('M2 execution invalid')
                        audits[name]=audit
                        if [f['output_digest'] for f in ff]!=[f['output_digest'] for f in zero]:raise ValueError('Perception differs across generation controls')
                        if kind=='zero' and any(q['reused_prefix_length'] for q in qq):raise ValueError('Zero control reused KV')
                    if mode=='fixed':
                        for f in ff:
                            refs=reference['frames'][f['frame_id']];count=sum(len(q['tokens']) for q in refs)
                            if f['fixed_tokens']['generated_steps']!=count or f['fixed_tokens']['actual_decode_forward_calls']!=count-len(refs) or f['stages'].get('decode',{}).get('calls',0)!=count-len(refs):raise ValueError('Actual fixed-forward counts differ')
                        for q in qq:
                            ref=reference['frames'][q['frame_id']][q['question_index']]
                            if q['question']!=ref['question'] or q['token_ids'][0][1:]!=ref['tokens'] or q['quality'] is not None or q['reference'] is not None:raise ValueError('Fixed-token proof/quality isolation failed')
                    allframes+=ff;allquestions+=qq
                summaries[mode+'_'+kind]=dict(counts=reuse_counts(allquestions),frames=len(allframes),questions=len(allquestions),
                    **{key:distribution(frame_metric(f,key) for f in allframes) for key in ('request_ms','model_ms','m3_ms','prefill_ms')},
                    decode_ms=distribution(f['stages'].get('decode',{}).get('gpu_span_ms',0.) for f in allframes),
                    visual_cuda_ms=distribution(f['stages']['eva_native' if kind=='native' else 'm2_total']['gpu_span_ms'] for f in allframes),
                    generated_tokens=sum(q['generated_tokens'] for q in allquestions),
                    actual_decode_forwards=sum(f['stages'].get('decode',{}).get('calls',0) for f in allframes),
                    throughput_fps=1000*len(allframes)/sum(f['request_ms'] for f in allframes),
                    peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in allframes),
                    peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in allframes),
                    selector_overhead_ms=distribution(f['fixed_tokens']['selector_cuda_span_ms'] for f in allframes) if mode=='fixed' else None,
                    timing_scope='free_zero captures are diagnostic, excluded from performance; all headline comparisons use untraced uncaptured fixed runs')
        nq=rows(root/f'{workload}_free_r0_zero/per_question_metrics.jsonl');cq=rows(root/f'{workload}_free_r0_reuse/per_question_metrics.jsonl');pairs=[]
        for n,c in zip(nq,cq):
            if (n['frame_id'],n['question'],n['reference'])!=(c['frame_id'],c['question'],c['reference']):raise ValueError('Quality inputs differ')
            pairs.append(dict(scene_id=c['scene_id'],frame_id=c['frame_id'],question=c['question'],baseline_answer=n['answer'],answer=c['answer'],reference=c['reference'],
                actual_reuse_tokens=c['reused_prefix_length'],reuse_type=c['reuse_diagnostics']['reuse_type'],**paired_quality(c['answer'],n['answer'],c['reference'])))
        write(root/(workload+'_m3_natural_quality_pairs.json'),pairs)
        m3quality=dict(mean_f1_loss=float(np.mean([p['f1_loss'] for p in pairs])),max_request_f1_loss=max(p['f1_loss'] for p in pairs),
            quality_loss_upper=quality_upper([dict(quality=p) for p in pairs]),critical_regressions=sum(p['critical_regressions'] for p in pairs),answers_equal=sum(p['answer_equal'] for p in pairs),requests=len(pairs),
            actual_reused_requests=[p for p in pairs if p['actual_reuse_tokens']],lexical_proxy_only=True)
        reused=[p for p in pairs if p['actual_reuse_tokens']]
        m3quality.update(reused_requests=len(reused),reused_mean_f1_loss=float(np.mean([p['f1_loss'] for p in reused])) if reused else None,
            reused_quality_upper=quality_upper([dict(quality=p) for p in reused]) if reused else None,
            reused_max_request_f1_loss=max([p['f1_loss'] for p in reused],default=None),reused_critical_regressions=sum(p['critical_regressions'] for p in reused))
        workloads[workload]=dict(summaries=summaries,m3_quality=m3quality,system_quality=compare_quality(root,f'{workload}_free_r0_native',f'{workload}_free_r0_reuse'),
            fixed_comparisons=dict(reuse_vs_zero=compare(root,workload,'reuse','zero'),reuse_vs_native=compare(root,workload,'reuse','native')))
    write(root/'execution_audits.json',audits)
    control=read(root/'same_input_control_summary.json');historical=read(root/'historical_exact_inputs_summary.json')
    if not control['all_actual_fixed_histories_equal'] or not control['planner_equivalence_all_checks_pass']:raise ValueError('Same-state control failed')
    historical_current=rows(root/'historical_current/per_question_metrics.jsonl')
    original_history=rows(Path('outputs/m3_quality_bounded_20261026_run01/wsdr_kv_final/per_question_metrics.jsonl'))
    historical_pipeline_comparison=dict(counts=reuse_counts(historical_current),answers_equal_to_old_pipeline=sum(a['answer']==b['answer'] for a,b in zip(historical_current,original_history)),requests=len(historical_current),scope='Current M1/M2 differs from historical perception; exact-carrier replay isolates M3')
    attribution={}
    for workload,w in workloads.items():
        c=w['fixed_comparisons']['reuse_vs_zero']
        attribution[workload]=dict(request_wall_observed_stable=c['request_ms']['stable_net_speedup'],
            request_saved_ms=c['request_ms']['saved_ms']['mean'],model_saved_ms=c['model_ms']['saved_ms']['mean'],
            non_model_saved_ms=c['non_model_ms']['saved_ms']['mean'],preprocessing_saved_ms=c['preprocessing_ms']['saved_ms']['mean'],
            m3_compute_ci_positive=c['m3_ms']['scene_bootstrap']['lower']>0,model_compute_ci_positive=c['model_ms']['scene_bootstrap']['lower']>0,
            e2e_compute_gain_proven=c['request_ms']['stable_net_speedup'] and c['m3_ms']['scene_bootstrap']['lower']>0 and c['model_ms']['scene_bootstrap']['lower']>0,
            scope='Preserve observed request gain separately; attribute a stable E2E computation gain only when M3 and complete-model paired scene intervals are also positive. No timing is filtered or subtracted.')
    result=dict(plan=plan,attribution=attribution,historical_current_pipeline=historical_pipeline_comparison,rejection=rejection,workloads=workloads,historical_exact_inputs=historical,same_input_control=control,
        microbenchmark=read(root/'prefix_pruning_microbenchmark.json'),policy_unchanged=True,single_formal_route=True,
        system_quality_pass=all(w['system_quality']['within_quality_budgets'] for w in workloads.values()),
        m3_quality_pass=all(w['m3_quality']['quality_loss_upper']<=.02 and not w['m3_quality']['critical_regressions'] and (not w['m3_quality']['reused_requests'] or w['m3_quality']['reused_quality_upper']<=.02) for w in workloads.values()),
        cross_scene_e2e_gain_proven={k:v['e2e_compute_gain_proven'] for k,v in attribution.items()})
    result['automatic_proxy_quality_pass']=result['m3_quality_pass']
    review=root/'semantic_failure_review.json'
    result['manual_semantic_review']=read(review) if review.exists() else None
    if result['manual_semantic_review']:
        result['m3_quality_pass']=False
        result['frozen_policy_qualification']='REJECTED_NEW_UNSUPPORTED_ROAD_RULE_CLAIMS'
    write(root/'summary.json',result)
    from tools.worldstream_m123.m3_recovery_report import report
    report(root,result)
    write(root/'status.json',dict(status='COMPLETED',implementation_complete=True,policy_unchanged=True,
        system_quality_pass=result['system_quality_pass'],m3_quality_pass=result['m3_quality_pass'],cross_scene_e2e_gain_proven=result['cross_scene_e2e_gain_proven']))
    print('M3 RECOVERY', {w:r['summaries']['free_reuse']['counts'] for w,r in workloads.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();analyze(Path(a.root))
