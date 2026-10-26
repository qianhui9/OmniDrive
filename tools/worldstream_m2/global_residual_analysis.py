"""Actual task quality, executed suffix counts, and quality/latency curves."""
import argparse
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import numpy as np
from .global_residual_benchmark import read, rows
from .quality_budget import quality_budgets, assess_quality, METRICS, DEFAULT_QUALITY_BUDGET
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
from tools.worldstream_m123.benchmark.metrics import distribution


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def compare_quality(root, native, candidate):
    base=read(root/(native+'_official')/'summary.json');actual=read(root/(candidate+'_official')/'summary.json')
    nq=rows(root/native/'per_question_metrics.jsonl');cq=rows(root/candidate/'per_question_metrics.jsonl')
    if not nq or any(not q['reference'] for q in nq+cq):
        raise ValueError('Task quality requires nonempty labelled reference answers')
    if [(q['frame_id'],q['question']) for q in nq]!=[(q['frame_id'],q['question']) for q in cq]:
        raise ValueError('Questions/frames differ')
    if [q['reference'] for q in nq]!=[q['reference'] for q in cq]:
        raise ValueError('Ground-truth answers differ')
    pairs=[dict(frame_id=q['frame_id'],question=q['question'],native_answer=n['answer'],answer=q['answer'],
                reference=q['reference'],**paired_quality(q['answer'],n['answer'],q['reference'])) for n,q in zip(nq,cq)]
    native_f1=float(np.mean([p['baseline_f1'] for p in pairs]));f1=float(np.mean([p['f1'] for p in pairs]))
    metrics={key:dict(native=float(b),candidate=float(a),loss=float(b-a)) for key,b,a in (
        ('mAP',base['detection']['mean_ap'],actual['detection']['mean_ap']),
        ('NDS',base['detection']['nd_score'],actual['detection']['nd_score']),
        ('DET_l',base['map']['OpenLane-V2 Score']['DET_l'],actual['map']['OpenLane-V2 Score']['DET_l']),('Token_F1',native_f1,f1))}
    known_path=Path('outputs/m2_execution_optimization_20261026_run01/service/r0_formal/per_question_metrics.jsonl')
    known={(q['frame_id'],q['question']):q for q in rows(known_path)}
    for pair in pairs:
        old=known.get((pair['frame_id'],pair['question']))
        pair['versus_previous_wsgrr']=paired_quality(pair['answer'],old['answer'],pair['reference']) if old else None
    budgets=read(root/'quality_budget.json')['quality_budgets'] if (root/'quality_budget.json').exists() else quality_budgets()
    matched=sum(p['versus_previous_wsgrr'] is not None for p in pairs)
    result=dict(metrics=metrics,**assess_quality(metrics,budgets),historical_wsgrr_matched_questions=matched,critical_proxy_regressions_vs_native=sum(p['critical_regressions'] for p in pairs),
        new_critical_proxy_regressions_vs_wsgrr=sum((p['versus_previous_wsgrr'] or {}).get('critical_regressions',0) for p in pairs) if matched else None,
        retained_critical_cases=[p for p in pairs if p['frame_id'] in ('88a6b5e35eb44deba6b1065d74b0a3c4','fd612e8660e74196af7da7e6487e5d53')],
        scope=actual['scope'],limitations='Numeric/spatial/motion/safety parsers are lexical proxies, not human semantic judgements')
    frame_map={f['frame_id']:f for f in rows(root/candidate/'per_frame_metrics.jsonl')}
    groups=defaultdict(list)
    for p in pairs:
        m2=frame_map[p['frame_id']]['m2']
        groups[(m2['global_update_mode'],m2['global_anchor_age'])].append(p)
    result['quality_by_anchor_age']=[dict(mode=mode,age=age,questions=len(pp),
        mean_f1_loss=float(np.mean([p['f1_loss'] for p in pp])),
        critical_proxy_regressions=sum(p['critical_regressions'] for p in pp))
        for (mode,age),pp in sorted(groups.items())]
    result['native_questions_below_perfect_f1']=sum(p['baseline_f1']<1. for p in pairs)
    native_scenes=read(root/(native+'_official')/'scenes.json')['scenes']
    candidate_scenes=read(root/(candidate+'_official')/'scenes.json')['scenes']
    result['per_scene']={}
    for scene,a in candidate_scenes.items():
        b=native_scenes[scene];ff=[f for f in frame_map.values() if f['scene_id']==scene]
        result['per_scene'][scene]=dict(frames=len(ff),
            global_suffix_reuse_frames=sum(f['m2']['global_suffix_reuse_frames'] for f in ff),
            mAP_loss=b['detection']['mean_ap']-a['detection']['mean_ap'],
            NDS_loss=b['detection']['nd_score']-a['detection']['nd_score'],
            DET_l_loss=b['map']['OpenLane-V2 Score']['DET_l']-a['map']['OpenLane-V2 Score']['DET_l'])
    write(root/(candidate+'_quality_pairs.json'),pairs)
    return result


def service_summary(path,formal):
    frames=rows(path/'per_frame_metrics.jsonl');questions=rows(path/'per_question_metrics.jsonl')
    label='m2_total' if formal else 'eva_native'
    result=dict(frames=len(frames),request_ms=distribution(f['request_ms'] for f in frames),
        visual_cuda_span_ms=distribution(f['stages'][label]['gpu_span_ms'] for f in frames),
        prefill_cuda_ms=distribution(f['stages'].get('prefill',{}).get('gpu_span_ms',0.) for f in frames),
        decode_cuda_ms=distribution(f['stages'].get('decode',{}).get('gpu_span_ms',0.) for f in frames),
        peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in frames),
        peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in frames),
        throughput_fps=1000*len(frames)/sum(f['request_ms'] for f in frames),
        generated_tokens=sum(q['generated_tokens'] for q in questions),
        m3_reused_tokens=sum(q['reused_prefix_length'] for q in questions))
    if formal:
        policies={(f['m2']['global_anchor_interval'],f['m2']['global_drift_threshold']) for f in frames}
        if len(policies)!=1:raise ValueError('M2 policy changed within one measured sequence')
        interval,threshold=policies.pop()
        result.update(global_anchor_interval=interval,global_drift_threshold=threshold)
        for f in frames:
            if f['observed_global_suffix_blocks']!=f['m2']['global_suffix_executed_blocks']:
                raise ValueError('Suffix did not execute as recorded')
            if f['observed_global_suffix_skipped_blocks']!=f['m2']['global_suffix_skipped_blocks']:
                raise ValueError('Skipped suffix counts incorrect')
        result.update(global_suffix_skipped_blocks=sum(f['observed_global_suffix_skipped_blocks'] for f in frames),
            global_suffix_reuse_frames=sum(f['m2']['global_suffix_reuse_frames'] for f in frames),
            anchor_updates=sum(f['m2']['global_suffix_executions'] for f in frames),
            refresh_reasons=dict(Counter(f['m2']['global_anchor_refresh_reason'] for f in frames)),
            anchor_ages=dict(Counter(str(f['m2']['global_anchor_age']) for f in frames)),
            drift=distribution(f['m2']['global_feature_drift'] for f in frames if f['m2']['global_feature_drift'] is not None),
            snapshot_wall_ms=distribution(f['m2']['M1_M2_state_read_ms'] for f in frames),
            m1_wall_ms=distribution(f['m1_commit_timings_ms']['m1_total_including_adapters_ms'] for f in frames),
            m2_wall_ms=distribution(f['m2']['M2_total_ms'] for f in frames),
            m3_wall_ms=distribution(f['m3']['runtime_host_elapsed_ms'] for f in frames),
            cache_publication_host_ms=distribution(f['m2']['M2_cache_publication_ms'] for f in frames),
            cache_bytes=max(f['m2']['cache_bytes'] for f in frames),
            window_reused=sum(f['m2']['reused_windows'] for f in frames),
            window_total=sum(f['m2']['total_windows'] for f in frames))
        result['component_cuda_span_ms']={label:distribution(
            sum(v['gpu_span_ms'] for k,v in f['stages'].items() if k.startswith(label)) for f in frames)
            for label in ('m2_patch_embedding','m2_window_block_','m2_suffix_block_','m2_global_state_update')}
        result['global_update_cuda_span_by_kind']={kind:distribution(
            f['stages'].get('m2_global_state_update',{}).get('gpu_span_ms',0.) for f in frames
            if f['m2']['global_update_mode']==kind) for kind in ('anchor_update','approx_global_residual_reuse')}
    return result


def calibration(root):
    parameters=read(root/'calibration_parameters.json');curve=[]
    native=service_summary(root/'cal_native',False)
    for age in (2,3,4):
        name='cal_age'+str(age);quality=compare_quality(root,'cal_native',name)
        timing=service_summary(root/name,True)
        if (timing['global_anchor_interval'],timing['global_drift_threshold'])!=(age,parameters['threshold']):
            raise ValueError('Executed calibration policy differs from candidate label')
        curve.append(dict(interval=age,threshold=parameters['threshold'],quality=quality,timing=timing,
            service_speedup=native['request_ms']['mean']/timing['request_ms']['mean'],
            visual_speedup=native['visual_cuda_span_ms']['mean']/timing['visual_cuda_span_ms']['mean']))
    write(root/'calibration_curve.json',dict(native=native,candidates=curve))
    eligible=[p for p in curve if p['quality']['within_quality_budgets']]
    if eligible:
        best=min(eligible,key=lambda p:(p['timing']['request_ms']['mean'],p['interval']))
        write(root/'selection.json',dict(interval=best['interval'],threshold=best['threshold'],
            status='development_metric_budget_pass',reason='Lowest measured complete service mean among candidates within all four frozen absolute quality budgets; numeric/spatial/motion/safety regressions are reported separately and are not certified by aggregate budgets',
            quality_budgets=best['quality']['quality_budgets'],
            calibration_parameters=parameters,
            new_critical_proxy_regressions_vs_wsgrr=best['quality']['new_critical_proxy_regressions_vs_wsgrr'],validation_used=False))
    else:
        if (root/'selection.json').exists():(root/'selection.json').unlink()
        write(root/'calibration_rejection.json',dict(status='NO_ACCEPTABLE_CANDIDATE',
            reason='No candidate passes all frozen quality budgets; do not run a selected policy on heldout data'))
    quality_latency_plot(root,curve,native)
    if not eligible:report_rejected_calibration(root,curve)
    return curve


def quality_latency_plot(root,points,native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,key in zip(axes,METRICS):
        budget=points[0]['quality']['quality_budgets'][key]
        native_ms=native['request_ms']['mean']
        ax.scatter([0.],[native_ms],marker='*',color='black',s=70)
        ax.annotate('Native',(0.,native_ms),xytext=(5,5),textcoords='offset points',fontsize=7)
        xs=[0.,budget]+[p['quality']['metrics'][key]['loss'] for p in points]
        lo,hi=min(xs),max(xs)
        for p in points:
            x=p['quality']['metrics'][key]['loss'];y=p['timing']['request_ms']['mean']
            right=p['interval'] in (3,4) or x>lo+.7*(hi-lo)
            dy={2:8,3:-12,4:-16}[p['interval']]
            ax.scatter([x],[y]);ax.annotate(str(p['interval'])+' / '+format(p['threshold'],'.3f'),(x,y),
                xytext=(-6 if right else 6,dy),ha='right' if right else 'left',
                textcoords='offset points',fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1))
        ax.axvline(budget,color='red',linestyle='--');ax.set(xlabel=key+' absolute loss',ylabel='Complete request ms')
        ax.margins(x=.15,y=.18)
    fig.suptitle('Development only: interval / H2 drift threshold; red dashed = task loss budget')
    fig.tight_layout();fig.savefig(root/'quality_latency_curve.png',dpi=160);fig.savefig(root/'quality_latency_curve.svg');plt.close(fig)


def report_rejected_calibration(root, points):
    budgets=points[0]['quality']['quality_budgets']
    lines=['# 开发实验：新阈值候选超出精度预算','',
        '四项任务允许的绝对下降：`'+json.dumps(budgets)+'`。新阈值只来自开发集漂移数据；下列候选均为同一 M2 算法，非多条部署路径。','',
        '| 间隔 | 漂移阈值 | 后缀复用帧 | 请求 mean ms | mAP 下降 | NDS 下降 | DET_l 下降 | Token F1 下降 | 通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for p in points:
        q,t=p['quality'],p['timing']
        lines.append(f"| {p['interval']} | {p['threshold']:.9f} | {t['global_suffix_reuse_frames']} | {t['request_ms']['mean']:.2f} | "+' | '.join(f"{q['metrics'][k]['loss']:.6f}" for k in METRICS)+f" | {q['within_quality_budgets']} |")
    lines += ['',
        '新阈值下没有候选通过四项开发精度预算，因此不生成新策略 selection.json，也不使用预留独立场景做反复调参。提高复用率没有保证任务误差在记录的预算内。','',
        '[原配置预算重评](previous_budget_reassessment/reassessment.md) 单独列出旧预测在新标准下的精度结论与原有计时。保留原参数，不把预算修改解释为精度或速度改善；回顾性结果不是新阈值的独立证据。','',
        '数字/空间/运动/安全逐问代理退化保留在 cal_age*_quality_pairs.json。曲线包含不合格候选：[PNG](quality_latency_curve.png) / [SVG](quality_latency_curve.svg)。新测试日志见 tests_gpu_final.log；运行命令与退出码见 commands.json。','',
        'M1/M3 算法、前缀预算与模型权重未改；M3 内部 0.02 策略参数不等于离线任务验收预算。新预算及默认值见 quality_budget.json 与 tools/worldstream_m2/quality_budget.py。','',
        '复现：从仓库根目录执行 `taskset -c 0-3,6-31 bash '+str(root)+'/reproduce.sh <新输出目录>`；没有合格候选时，脚本在开发结果处结束。']
    if (root/'refinement_history.md').exists():lines += ['', '[先前开发阈值与本轮校准的完整记录](refinement_history.md)。']
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    write(root/'status.json',dict(status='DEVELOPMENT_CANDIDATES_REJECTED',implementation_complete=True,
        quality_budgets=points[0]['quality']['quality_budgets'],selection_created=False,
        independent_validation_performed=False,reason='Every new-threshold candidate exceeds at least one frozen development quality budget'))


def final(root):
    result=dict(selection=read(root/'selection.json'))
    from .production_acceptance import validate_rows
    audits={name:validate_rows(rows(root/name/'per_frame_metrics.jsonl'),require_reuse=False)
            for name in ['validation_formal']+['r'+str(i)+'_formal' for i in range(3)]}
    expected=(result['selection']['interval'],result['selection']['threshold'])
    for name in audits:
        frames=rows(root/name/'per_frame_metrics.jsonl')
        if any((f['m2']['global_anchor_interval'],f['m2']['global_drift_threshold'])!=expected for f in frames):
            raise ValueError('Frozen development policy changed during validation or service')
        audits[name]['frozen_policy_consistent']=True
    write(root/'formal_execution_audit.json',audits)
    if any(a['status']!='PASS' for a in audits.values()):raise ValueError('Actual window publication audit failed')
    for split in ('service','validation'):
        if split=='service':
            ns=[service_summary(root/('r'+str(i)+'_native'),False) for i in range(3)]
            fs=[service_summary(root/('r'+str(i)+'_formal'),True) for i in range(3)]
            result['rounds']=[dict(native=n,formal=f,speedup=n['request_ms']['mean']/f['request_ms']['mean']) for n,f in zip(ns,fs)]
            saved=np.array([n['request_ms']['mean']-f['request_ms']['mean'] for n,f in zip(ns,fs)])
            margin=4.302652729911275*float(saved.std(ddof=1))/np.sqrt(3)
            result['round_mean_saved_ms_ci95']=dict(mean=float(saved.mean()),
                lower=float(saved.mean()-margin),upper=float(saved.mean()+margin),
                method='Paired round means, Student t(df=2), only 3 repeats on the same 40 frames; not a multi-scene confidence guarantee')
            result['service_quality']=compare_quality(root,'r0_native','r0_formal')
            result['aggregate']={}
            result['repeat_output_consistency']={}
            for name,label in [('native','eva_native'),('formal','m2_total')]:
                ff=[f for i in range(3) for f in rows(root/('r'+str(i)+'_'+name)/'per_frame_metrics.jsonl')]
                qq=[q for i in range(3) for q in rows(root/('r'+str(i)+'_'+name)/'per_question_metrics.jsonl')]
                result['aggregate'][name]=dict(unique_frames=len({f['frame_id'] for f in ff}),
                    request_ms=distribution(f['request_ms'] for f in ff),
                    visual_cuda_span_ms=distribution(f['stages'][label]['gpu_span_ms'] for f in ff),
                    prefill_cuda_ms=distribution(f['stages'].get('prefill',{}).get('gpu_span_ms',0.) for f in ff),
                    decode_cuda_ms=distribution(f['stages'].get('decode',{}).get('gpu_span_ms',0.) for f in ff),
                    peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),
                    peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff),
                    throughput_fps=1000*len(ff)/sum(f['request_ms'] for f in ff),
                    m3_reused_tokens=sum(q['reused_prefix_length'] for q in qq),
                    generated_tokens=sum(q['generated_tokens'] for q in qq))
                result['repeat_output_consistency'][name]=dict(
                    task_digests_equal=all([f['output_digest'] for f in ff[i*40:(i+1)*40]]==[f['output_digest'] for f in ff[:40]] for i in (1,2)),
                    answers_equal=all([q['answer'] for q in qq[i*40:(i+1)*40]]==[q['answer'] for q in qq[:40]] for i in (1,2)))
        else:
            result['validation']=dict(native=service_summary(root/'validation_native',False),formal=service_summary(root/'validation_formal',True),
                quality=compare_quality(root,'validation_native','validation_formal'),threshold_retuned=False)
    write(root/'final_summary.json',result)
    visual=rows(root/'drift_calibration40/frames.jsonl')
    grouped=defaultdict(list)
    for r in visual:grouped[(r['strategy'],r['global_state']['global_update_mode'],r['global_state']['global_anchor_age'])].append(r)
    probe_errors=[dict(strategy=key[0],mode=key[1],age=key[2],frames=len(rr),
        mean_abs_error=float(np.mean([r['difference']['mean_abs'] for r in rr])),
        mean_cuda_span_ms=float(np.mean([r['incremental']['gpu_span_ms'] for r in rr])))
        for key,rr in sorted(grouped.items())]
    points=read(root/'calibration_curve.json')['candidates']
    quality_latency_plot(root,points,read(root/'calibration_curve.json')['native'])
    paired=rows(root/'paired40/frames.jsonl')
    grouped_pair=defaultdict(list)
    for r in paired:grouped_pair[(r['global_state']['global_update_mode'],r['global_state']['global_anchor_age'])].append(r)
    write(root/'visual_error_by_age.json',dict(initial_drift_probe_vs_native=probe_errors,
        same_prefix_pair_vs_wsgrr=[dict(mode=mode,age=age,frames=len(rr),
            mean_abs_error=float(np.mean([r['reference_pair']['features']['mean_abs'] for r in rr])),
            max_abs_error=max(r['reference_pair']['features']['max_abs'] for r in rr))
            for (mode,age),rr in sorted(grouped_pair.items())]))
    visual={name:distribution(r['reference_pair'][key]['wall_ms'] for r in paired)
            for name,key in [('before','before'),('after','optimized')]}
    visual['native']=distribution(r['native']['wall_ms'] for r in paired)
    visual['cuda_span_ms']={name:distribution(r['reference_pair'][key]['gpu_span_ms'] for r in paired)
        for name,key in [('before','before'),('after','optimized')]}
    visual['cuda_span_ms']['native']=distribution(r['native']['gpu_span_ms'] for r in paired)
    visual['isolated_commit']={label:read(root/'paired40'/name) for label,name in (
        ('before','reference_synchronized_actual_publication.json'),('after','synchronized_actual_publication.json'))}
    visual['snapshot_export']={name:distribution(t for r in paired for t in r['snapshot_export_pair'][key])
        for name,key in [('before','before_wall_ms'),('after','optimized_wall_ms')]}
    import pstats
    visual['snapshot_cprofile']={}
    for label in ('before','optimized'):
        stats=pstats.Stats(str(root/'paired40'/('snapshot_'+label+'.prof'))).stats
        visual['snapshot_cprofile'][label]={name:dict(calls=sum(v[1] for k,v in stats.items() if k[2]==name),
            cumulative_ms=sum(v[3]*1000 for k,v in stats.items() if k[2]==name))
            for name in ('export_m2_inputs','source_digest','deepcopy','freeze')}
    visual['all_prefix_plans_outputs_ages_and_sources_equal']=all(all(r['reference_pair'][k] for k in (
        'window_ids_equal','packed_outputs_equal','ages_equal','cache_versions_equal','provenance_equal')) for r in paired)
    from .execution_analysis import trace_summary
    visual['profiler_before']=trace_summary(root/'paired40/reference_profile_output')
    visual['profiler_after_reuse']=trace_summary(root/'paired40/profile_output')
    visual['scope']='Same exported M1 snapshots; prefix plan/output/source equality required. Median of three repeats per frame, averaged across 40 unique frames. Snapshot export separately measured with three alternating pairs per request.'
    write(root/'paired_visual_summary.json',visual)
    result['validation']['all_metric_budgets_pass']=result['validation']['quality']['within_quality_budgets']
    result['independent_quality_budget_pass']=result['validation']['quality']['within_quality_budgets']
    result['quality_budgets']=result['validation']['quality']['quality_budgets']
    result['complete_service_speedup']=result['aggregate']['native']['request_ms']['mean']/result['aggregate']['formal']['request_ms']['mean']
    result['complete_service_mean_acceleration']=result['complete_service_speedup']>1.
    result['quality_latency_goal_met']=result['independent_quality_budget_pass'] and result['complete_service_mean_acceleration']
    import runpy
    configured=runpy.run_path('projects/configs/OmniDrive/worldstream_m123_auto.py')['model']['m2_config']
    result['formal_default_parameters']={k:configured[k] for k in ('global_anchor_interval','global_drift_threshold')}
    result['experimental_parameters_installed_as_default']=(configured['global_anchor_interval'],configured['global_drift_threshold'])==(result['selection']['interval'],result['selection']['threshold'])
    result['formal_default_matches_rejected_experiment']=(not result['independent_quality_budget_pass']) and result['experimental_parameters_installed_as_default']
    previous_path=root/'previous_budget_reassessment/reassessment.json'
    result['formal_default_historical_subset_quality_pass']=None
    if previous_path.exists():
        previous=read(previous_path);old_policy=previous['selection_unchanged']
        if (configured['global_anchor_interval'],configured['global_drift_threshold'])==(old_policy['interval'],old_policy['threshold']):
            result['formal_default_historical_subset_quality_pass']=previous['validation']['within_quality_budgets']
    result['external_service_deployed']=False
    old_development=read('outputs/m2_execution_optimization_20261026_run01/service/r0_formal_official_subset/summary.json')
    new_development=read(root/'r0_formal_official/summary.json')
    result['quality_vs_historical_wsgrr']={'development':dict(
        mAP_loss=old_development['detection']['mean_ap']-new_development['detection']['mean_ap'],
        NDS_loss=old_development['detection']['nd_score']-new_development['detection']['nd_score'],
        DET_l_loss=old_development['map']['OpenLane-V2 Score']['DET_l']-new_development['map']['OpenLane-V2 Score']['DET_l'])}
    old_path=root/'historical_wsgrr_validation_official/summary.json'
    if old_path.exists():
        old=read(old_path);new=read(root/'validation_formal_official/summary.json')
        if old['sample_tokens']!=new['sample_tokens']:raise ValueError('Historical quality frame identity mismatch')
        result['quality_vs_historical_wsgrr']['validation']=dict(
            mAP_loss=old['detection']['mean_ap']-new['detection']['mean_ap'],
            NDS_loss=old['detection']['nd_score']-new['detection']['nd_score'],
            DET_l_loss=old['map']['OpenLane-V2 Score']['DET_l']-new['map']['OpenLane-V2 Score']['DET_l'])
    result['quality_vs_historical_wsgrr']['scope']='Historical full-loop prediction comparison, including M1/task-head feedback; not an identical-state causal task-loss isolation. Latency causality comes only from paired_visual_summary.'
    write(root/'final_summary.json',result)
    report(root,result,visual,points)
    write(root/'status.json',dict(status='COMPLETED',implementation_complete=True,
        quality_budgets=result['quality_budgets'],independent_quality_budget_pass=result['independent_quality_budget_pass'],
        complete_service_speedup=result['complete_service_speedup'],quality_latency_goal_met=result['quality_latency_goal_met'],
        independent_validation_performed=True,formal_default_parameters=result['formal_default_parameters'],
        experimental_parameters_installed_as_default=result['experimental_parameters_installed_as_default'],external_service_deployed=False))


def report(root,result,visual,points):
    n,f=result['aggregate']['native'],result['aggregate']['formal']
    val=result['validation'];quality=result['service_quality'];selection=result['selection']
    parameters=read(root/'calibration_parameters.json')
    test_log='tests_gpu_after_budget_guards.log' if (root/'tests_gpu_after_budget_guards.log').exists() else 'tests_gpu_final.log'
    budgets=result['quality_budgets']
    lines=['# OmniDrive WSGRR-AR：放宽精度预算后的实验','',
        '预算单位为原生分数减去正式分数的**绝对下降**，逐项独立判断，边界包含等号；负值表示改善。允许下降 0.05 相当于 5 个百分点，并非相对下降 5%。预算仅用于离线选择与验收，不参与在线 M2 判定。','',
        '四项预算：`'+json.dumps(budgets)+'`。旧结果未被改写；[原配置预算重评](previous_budget_reassessment/reassessment.md) 与本轮新参数实测分别记录。','',
        f"本轮实测固定参数：间隔 {selection['interval']}，最大相机 H2 相对 RMS 漂移阈值 {selection['threshold']:.9f}；前缀预算 3、最大复用年龄 8、checkpoint iter_10548.pth 不变。M1/M3 算法不变；M3 内部 quality_budget=0.02 是原有 KV 策略参数，不作为本次四项任务验收预算。",'',
        '## 开发集校准','',
        f"40 帧自然连续开发场景；漂移阈值取既有开发数据 age-1 最大相机 H2 漂移的 {parameters.get('drift_quantile',.25):.0%} 分位数。漂移来源及 SHA256 见 calibration_parameters.json。只比较间隔 2/3/4；不使用独立场景调参。完整闭环真实 Detection/Map/语言输出由官方子集评分器评估。",'',
        '| 间隔 | 后缀复用帧 | 请求 mean ms | mAP 下降 | NDS 下降 | DET_l 下降 | Token F1 下降 | 预算 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for p in points:
        q=p['quality'];t=p['timing'];m=q['metrics']
        lines.append(f"| {p['interval']} | {t['global_suffix_reuse_frames']} | {t['request_ms']['mean']:.2f} | "+' | '.join(f"{m[k]['loss']:.6f}" for k in METRICS)+f" | {'通过' if q['within_quality_budgets'] else '未通过'} |")
    lines += ['',selection['reason'],'',
        '仅按单轮开发完整请求均值选择，未宣称统计显著最优；独立验证和三轮性能测量均不回调参数。逐问数字/空间/运动/安全词法代理退化保留在 *_quality_pairs.json，不被平均分数预算覆盖。','',
        '## 同快照视觉与 Snapshot','',
        f"同一 M1 快照、当前帧和窗口选择，旧 WSGRR / 新 WSGRR-AR / 原生 EVA 同步 Wall 均值为 {visual['before']['mean']:.2f} / {visual['after']['mean']:.2f} / {visual['native']['mean']:.2f} ms。旧→新节省 {visual['before']['mean']-visual['after']['mean']:.2f} ms；相对原生纯视觉 {visual['native']['mean']/visual['after']['mean']:.4f}×。此边界排除 Snapshot、M1 Precheck/Commit 和任务头。",'',
        f"对应 CUDA Span 均值 {visual['cuda_span_ms']['before']['mean']:.2f}/{visual['cuda_span_ms']['after']['mean']:.2f}/{visual['cuda_span_ms']['native']['mean']:.2f} ms。Span 包含主机提交空隙；代表帧 kernel active 旧/新 {visual['profiler_before']['kernel_active_sum_ms']:.3f}/{visual['profiler_after_reuse']['kernel_active_sum_ms']:.3f} ms。两种边界不能混用，也不能把代表帧推为所有帧。",'',
        f"前缀计划、packed outputs、年龄、版本、来源全部相同：{visual['all_prefix_plans_outputs_ages_and_sources_equal']}。最终 embedding 是近似输出，非精确等价。",'',
        f"同一 pending 请求交替导出 Snapshot：旧/新 mean {visual['snapshot_export']['before']['mean']:.3f}/{visual['snapshot_export']['after']['mean']:.3f} ms，P50 {visual['snapshot_export']['before']['p50']:.3f}/{visual['snapshot_export']['after']['p50']:.3f}，P95 {visual['snapshot_export']['before']['p95']:.3f}/{visual['snapshot_export']['after']['p95']:.3f}，max {visual['snapshot_export']['before']['maximum']:.3f}/{visual['snapshot_export']['after']['maximum']:.3f}。所有正常长尾样本保留，不从 deepcopy 次数直接推断加速。",'',
        '完整导出计算契约逐帧一致，来源检查未删。cProfile、实际缓存发布、kernel trace 和样本见 paired40/；无 GC 采样，不推断长尾来源。','',
        '## 三轮完整服务','',
        '顺序为原生→正式、正式→原生、原生→正式。同 40 帧/问题/seed/生成配置，预热后重置状态。120 次测量是 40 个唯一驾驶帧的重复，非 120 个独立样本。官方任务评分取第 0 轮真实 predictions.pkl。','',
        '| 轮 | 原生 mean/P50/P95 ms | 正式 mean/P50/P95 ms | 加速比 |','|---|---|---|---:|']
    for i,r in enumerate(result['rounds']):
        a,b=r['native']['request_ms'],r['formal']['request_ms']
        lines.append(f"| {i} | {a['mean']:.2f}/{a['p50']:.2f}/{a['p95']:.2f} | {b['mean']:.2f}/{b['p50']:.2f}/{b['p95']:.2f} | {r['speedup']:.4f}× |")
    lines += ['','| 三轮合计 | 原生 | 正式 |','|---|---:|---:|']
    for label,key in [('完整请求 mean ms','request_ms'),('视觉 wrapper CUDA Span mean ms','visual_cuda_span_ms'),('Prefill CUDA mean ms','prefill_cuda_ms'),('Decode CUDA mean ms','decode_cuda_ms')]:
        lines.append(f"| {label} | {n[key]['mean']:.2f} | {f[key]['mean']:.2f} |")
    for label,key,scale in [('吞吐 frame/s','throughput_fps',1),('峰值 allocated GiB','peak_allocated_bytes',1024**3),('峰值 reserved GiB','peak_reserved_bytes',1024**3),('生成 token','generated_tokens',1),('M3 实际复用 token','m3_reused_tokens',1)]:
        lines.append(f"| {label} | {n[key]/scale:.4f} | {f[key]/scale:.4f} |")
    ci=result['round_mean_saved_ms_ci95']
    lines += ['',f"完整服务加速比 {result['complete_service_speedup']:.4f}×；平均时延降低：{result['complete_service_mean_acceleration']}。三轮配对节省时间 95% t 区间 [{ci['lower']:.2f}, {ci['upper']:.2f}] ms，正数表示节省；仅同场景三轮，不是跨场景保证。",'',
        '重复输出一致性：`'+json.dumps(result['repeat_output_consistency'])+'`。回答长度会影响 Decode 与总请求时延，不把全部变化归因于视觉或 KV 复用。视觉 wrapper 包含状态管理和主机间隙；嵌套 M1/M2/M3 host 时间不可直接求和。','']
    for i,r in enumerate(result['rounds']):
        t=r['formal']
        lines.append(f"轮 {i}：实际后缀复用 {t['global_suffix_reuse_frames']}/{t['frames']} 帧，独立 hooks 验证跳过 {t['global_suffix_skipped_blocks']} 个 Block 调用，锚帧更新 {t['anchor_updates']} 次。缓存 {t['cache_bytes']/1024**2:.3f} MiB。M1/M2/M3 host 均值 {t['m1_wall_ms']['mean']:.2f}/{t['m2_wall_ms']['mean']:.2f}/{t['m3_wall_ms']['mean']:.2f} ms。")
    lines += ['','## 任务精度与独立验证','',
        '官方 Detection/Lane 在声明的 GT 子集上评分，不是全量 6019 帧成绩。Token F1 对照相同真实参考答案；词法代理检查不等于人工语义评判。','',
        '| 指标 | 开发原生 | 开发正式 | 独立原生 | 独立正式 | 独立下降 | 预算 | 通过 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for key in METRICS:
        a,b=quality['metrics'][key],val['quality']['metrics'][key]
        lines.append(f"| {key} | {a['native']:.6f} | {a['candidate']:.6f} | {b['native']:.6f} | {b['candidate']:.6f} | {b['loss']:.6f} | {budgets[key]:.6f} | {val['quality']['metric_budget_pass'][key]} |")
    lines += ['',f"独立 {len(val['quality']['per_scene'])} 场景、{val['native']['frames']} 帧；四项预算通过：{val['quality']['within_quality_budgets']}。场景按固定规则预先选择，排除开发及旧验证场景，未按候选结果挑选；详见 validation_sequence24.json。",'',
        '| 独立场景 | 帧 | 全局复用帧 | mAP 下降 | NDS 下降 | DET_l 下降 |','|---|---:|---:|---:|---:|---:|']
    for scene,v in val['quality']['per_scene'].items():
        lines.append(f"| {scene} | {v['frames']} | {v['global_suffix_reuse_frames']} | {v['mAP_loss']:.6f} | {v['NDS_loss']:.6f} | {v['DET_l_loss']:.6f} |")
    worst_scene,worst=max(val['quality']['per_scene'].items(),key=lambda item:item[1]['mAP_loss'])
    lines += ['',f"独立集最大 mAP 下降位于 {worst_scene}：{worst['frames']} 帧中 {worst['global_suffix_reuse_frames']} 帧全局复用，mAP/NDS 下降 {worst['mAP_loss']:.6f}/{worst['NDS_loss']:.6f}。开发 H2 漂移阈值没有保证跨场景任务误差稳定；任务头与 M1 后续状态也会传播历史差异。",'',
        f"开发原生 {quality['native_questions_below_perfect_f1']}/40 个问题的 Token F1 小于 1；逐问表同时保留原生答案、GT 和新答案，用于识别原有错误与新增退化。"]
    lines += ['',f"开发集关键代理退化检查数：对原生 {quality['critical_proxy_regressions_vs_native']}，对历史 WSGRR {quality['new_critical_proxy_regressions_vs_wsgrr']}；独立集对原生 {val['quality']['critical_proxy_regressions_vs_native']}。一个问题可触发多项检查，不等于独立错误数。原 88a6b5... / fd612e... 案例及改善/退化均保留在逐问结果。",'',
        '历史 WSGRR 闭环任务差：`'+json.dumps(result['quality_vs_historical_wsgrr'])+'`。任务头与 M1 后续状态会传播误差，不能当作同状态残差的因果精度损失。复用年龄、特征差和逐问损失见 visual_error_by_age.json / *_quality_pairs.json。','',
        '## 结论与复现','',
        f"独立精度预算通过：{result['independent_quality_budget_pass']}；完整请求平均加速：{result['complete_service_mean_acceleration']}；本子集同时满足精度和平均时延目标：{result['quality_latency_goal_met']}。三项结论分开记录，预算放宽不等于模型精度提高。",'',
        '实际正式默认参数：`'+json.dumps(result['formal_default_parameters'])+'`；本轮实验参数覆盖默认值：'+str(result['experimental_parameters_installed_as_default'])+'；当前默认参数在旧独立子集按本轮预算重评通过：'+str(result['formal_default_historical_subset_quality_pass'])+'。独立预算失败时保留原默认参数，不依据独立集再搜索阈值。原参数按 0.05 的旧独立子集重评见 previous_budget_reassessment/。','',
        f'只运行实验服务，未发布外部服务。平均指标达标不认证所有场景/关键问答；独立集不用于继续调参。单一路径执行与 hooks 计数核对见 formal_execution_audit.json；最新 GPU 回归见 {test_log}。保留正常负收益样本。','',
        '本轮不继续使用独立数据调参。后续若研究更稳定的残差条件，需要更丰富的开发场景及另一份未参与调参的独立数据。','',
        '环境见 environment.json、各 worker manifest；命令/退出码见 commands.json。旧输出保留原预算含义，新结果记录本轮预算。','',
        '```bash',f'taskset -c 0-3,6-31 bash {root}/reproduce.sh outputs/m2_global_residual_reproduction_budget_0p05','```','',
        '[质量—时延曲线 PNG](quality_latency_curve.png) / [SVG](quality_latency_curve.svg)；[最终机器结果](final_summary.json)；[同快照结果](paired_visual_summary.json)；[校准结果](calibration_curve.json)。']
    if (root/'refinement_history.md').exists():lines += ['', '[先前开发阈值与本轮校准的完整记录](refinement_history.md)。']
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')


def reassess(root, evidence_root):
    """Reclassify immutable previous predictions without pretending to rerun them."""
    evidence=Path(evidence_root).resolve()
    for name in ('cal_native','cal_age2','cal_age3','cal_age4','r0_native','r0_formal','validation_native','validation_formal'):
        for suffix in ('','_official'):
            dest=root/(name+suffix)
            if not dest.exists():dest.symlink_to(evidence/(name+suffix),target_is_directory=True)
    evidence_names=['final_summary.json','selection.json','experiment_report.md']
    result={'evidence_root':str(evidence),'new_inference_performed':False,
            'evidence_sha256':{name:hashlib.sha256((evidence/name).read_bytes()).hexdigest() for name in evidence_names},
            'interpretation':'Only acceptance criteria changed; predictions and measured timings unchanged',
            'calibration':{str(age):compare_quality(root,'cal_native','cal_age'+str(age)) for age in (2,3,4)},
            'service':compare_quality(root,'r0_native','r0_formal'),
            'validation':compare_quality(root,'validation_native','validation_formal')}
    old=read(evidence/'final_summary.json')
    result['complete_service_speedup']=old['aggregate']['native']['request_ms']['mean']/old['aggregate']['formal']['request_ms']['mean']
    result['selection_unchanged']=old['selection']
    result['calibration_timing']={str(age):service_summary(root/('cal_age'+str(age)),True) for age in (2,3,4)}
    eligible=[age for age in (2,3,4) if result['calibration'][str(age)]['within_quality_budgets']]
    result['best_previous_candidate_under_new_budget']=min(eligible,key=lambda age:(result['calibration_timing'][str(age)]['request_ms']['mean'],age)) if eligible else None
    write(root/'reassessment.json',result)
    lines=['# 原配置：仅更新预算的回顾性重评','',
           '原预测和计时未重跑、未改动；旧验证集已经看过，本表不作为新参数的独立验证。四项允许的绝对下降见 quality_budget.json。','',
           '| 指标 | 原生 | 原正式 | 绝对下降 | 新预算通过 |','|---|---:|---:|---:|---|']
    for key in METRICS:
        m=result['validation']['metrics'][key]
        lines.append(f"| {key} | {m['native']:.6f} | {m['candidate']:.6f} | {m['loss']:.6f} | {result['validation']['metric_budget_pass'][key]} |")
    lines += ['',f"原独立子集精度按新预算通过：{result['validation']['within_quality_budgets']}。原三轮完整服务加速比仍为 {result['complete_service_speedup']:.4f}×；修改预算没有产生推理加速或精度提高。",'',
              '| 旧间隔 | 旧阈值 | 旧请求 mean ms | 旧复用帧 | mAP 下降 | NDS 下降 | DET_l 下降 | Token F1 下降 | 新预算通过 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for age in (2,3,4):
        q=result['calibration'][str(age)];t=result['calibration_timing'][str(age)]
        lines.append(f"| {age} | {old['selection']['threshold']:.9f} | {t['request_ms']['mean']:.2f} | {t['global_suffix_reuse_frames']} | "+' | '.join(f"{q['metrics'][k]['loss']:.6f}" for k in METRICS)+f" | {q['within_quality_budgets']} |")
    lines += ['',f"按新预算筛选旧开发候选，最低旧完整请求均值的间隔为 {result['best_previous_candidate_under_new_budget']}；这些是历史计时，不能与本轮候选作同时段因果性能比较。新阈值候选与本轮原生另行实测；通过开发预算后才使用预留新场景验证。"]
    (root/'reassessment.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['calibration','final','reassess'],required=True)
    p.add_argument('--quality-budget',type=float,default=DEFAULT_QUALITY_BUDGET,help='Maximum absolute loss per task metric (default: 0.05)')
    p.add_argument('--vlm-quality-budget',type=float,default=None,help='Legacy reproduction override only')
    p.add_argument('--evidence-root',help='Immutable previous run for retrospective reassessment')
    args=p.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    budgets=quality_budgets(args.quality_budget,args.vlm_quality_budget)
    path=root/'quality_budget.json'
    if path.exists() and read(path)['quality_budgets']!=budgets:raise ValueError('Frozen budget differs')
    write(path,dict(quality_budgets=budgets,units='absolute native-minus-candidate score loss'))
    if args.stage=='reassess':
        if not args.evidence_root:p.error('--evidence-root is required for reassess')
        reassess(root,args.evidence_root)
    else:(calibration if args.stage=='calibration' else final)(root)
