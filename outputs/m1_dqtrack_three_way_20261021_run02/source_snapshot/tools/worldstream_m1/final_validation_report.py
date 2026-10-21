"""Derive final M1 freeze decision from fresh raw evidence only."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import beta
from .io_utils import write_json
from .state_interface_gpu_report import read,percentile,junit,summarize
from .final_validation_sampling import PROTOCOL

ROOT=Path('outputs/module1_final_validation_refinement')


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def physical_stats(rows):
    scored=[s for s in rows if s['scorable']];false=sum(s['correct'] is False for s in scored)
    total=len(scored)
    return dict(samples=len(rows),scorable=total,unscorable=len(rows)-total,
        scorable_fraction=total/len(rows) if rows else None,
        correct=total-false,false=false,error_rate=false/total if total else None,
        descriptive_iid_upper95=float(beta.ppf(.95,false+1,total-false)) if total and false<total else 1. if total else None,
        error_types=dict(Counter('non_finite_or_reference_error_above_2px' for s in scored if not s['correct'])),
        unscorable_reasons=dict(Counter(s.get('unscorable_reason','no_reference') for s in rows if not s['scorable'])),
        scenes=sorted({s['scene'] for s in scored}),logs=sorted({s['log'] for s in scored}),
        independence='Descriptive interval only. Pixels/frames/objects from one log are correlated; no qualification.')


def summarize_gpu(path):
    out=summarize(path)
    out['execution_boundaries']={m:('full recompute' if m in ('original','full_maintain') else 'camera') for m in out['arms']}
    out['comparison_scope']='Old/current/final share Camera boundary, same cost table and task gate; all M1 modes maintain persistent state.'
    runs=read(path/'runs.json',[])
    for mode,arm in out['arms'].items():
        fs=[f for r in runs if r['mode']==mode and not r['warmup'] for f in r['frames']];ds=[f['runtime'] for f in fs]
        for k in ('candidate_cameras','complete_candidate_windows','verified_exact_cameras','eligible_exact_cameras','recomputed_cameras'):
            arm[k]=sum(d.get(k,0) or 0 for d in ds)
        arm['adapter_ms']=percentile([d.get('M1_M2_adapter_ms') for d in ds])
        arm['association_ms']=percentile([(f.get('m1_commit') or {}).get('association_ms') for f in fs])
        arm['transfer_bytes']={kind:sum((f.get('m1_precheck_diagnostics') or {}).get(kind,0) or 0 for f in fs)
            for kind in ('input_gpu_to_cpu_transfer_bytes',)}
        arm['transfer_bytes']['output_gpu_to_cpu_transfer_bytes']=sum((f.get('m1_commit') or {}).get('output_gpu_to_cpu_transfer_bytes',0) or 0 for f in fs)
    return out


def main():
    root=ROOT;p=root/'replay';g=root/'gpu_validation'
    def write(name,value):write_json(root/name,value)
    completion=read(p/'completion.json',{});methods=read(p/'method_comparison.json',{})
    if not methods:raise RuntimeError('Final replay incomplete')
    root_causes=read(root/'critical_event_root_cause.json',{})
    for event in root_causes.get('events',[]):
        event.setdefault('original_diagnostic_label',event['root_cause'])
        same_class=[a for a in event['online_anchors'] if a['semantic_class']==event['semantic_class']]
        event['same_class_overlapping_committed_anchors']=len(same_class)
        event['GT_instance_identity_established_online']=False
        if event['changed_evidence_cleared_by_exclusion']:
            event['root_cause']='measured_change_discarded_by_acceptance_exclusion'
            event['online_identifiability']='Localized rejected samples exist, but insufficient to explain the entire offline ROI.'
        elif not same_class:
            event['root_cause']='no_same_class_overlapping_committed_source'
            event['online_identifiability']='No defensible object-specific motion envelope. Other-class overlap is not identity.'
        elif not any(a.get('previous_region') is not None for a in same_class):
            event['root_cause']='historical_geometry_without_camera_observation_support'
            event['online_identifiability']='Predicted geometry alone cannot supply measured source occupancy.'
        else:
            event['root_cause']='historical_projection_partial_or_static_visibility_ambiguity'
            event['online_identifiability']='Projected crop overlaps only part of ROI; RGB consistency does not establish unchanged visibility or foreground ownership.'
    root_causes['refined_diagnosis_counts']=dict(Counter(e['root_cause'] for e in root_causes.get('events',[])))
    write('critical_event_root_cause.json',root_causes)
    pixels=jsonl(p/'pixels.jsonl');times=read(p/'timings.json',[]);identity=read(p/'identity.json',[])
    byframe=defaultdict(dict)
    for s in identity:byframe[s['frame']][s['method']]=s['digest']
    state_equal=all(len(set(v.values()))==1 for v in byframe.values())
    rows_by={name:[r for r in pixels if r['method']==name] for name in methods}
    event_map={name:{(r['frame'],r['camera'],e['instance']):dict(frame=r['frame'],camera=r['camera'],scene=r['scene'],**e)
        for r in rs for e in r.get('events',[])} for name,rs in rows_by.items()}
    baseline=event_map['current_optimized'];final=event_map['final']
    failed={k for k,e in baseline.items() if not e['recalled']}
    assert len(failed)==14
    fixed=[final[k] for k in sorted(failed) if final[k]['recalled']]
    remaining=[final[k] for k in sorted(failed) if not final[k]['recalled']]
    new_failures=[v for k,v in final.items() if not v['recalled'] and k not in failed]
    paired=dict(population='frozen_96_frames_exploratory_reused_population',old_failures=14,
        fixed_count=len(fixed),fixed=fixed,remaining_count=len(remaining),remaining=remaining,new_failures=new_failures,
        same_event_keys=all(set(events)==set(baseline) for events in event_map.values()),
        event_pairs=[dict(key=list(k),by_method={n:e[k] for n,e in event_map.items()}) for k in sorted(failed)],
        current_candidate_pixels=sum(r['effective_pixels'] for r in rows_by['current_optimized']),
        final_candidate_pixels=sum(r['effective_pixels'] for r in rows_by['final']))
    paired['candidate_pixel_loss']=paired['current_candidate_pixels']-paired['final_candidate_pixels']
    paired['candidate_relative_loss']=paired['candidate_pixel_loss']/max(1,paired['current_candidate_pixels'])
    for n in ('current_optimized','motion_only','measurement_only','final'):
        paired[n+'_accepted_correct']=methods[n]['accepted_scorable_points']-methods[n]['false_stable_points']
        paired[n+'_accepted_false']=methods[n]['false_stable_points']
    write('candidate_paired_analysis.json',paired)
    motion={name:dict(method=methods[name],revoked_candidate_pixels=sum(r.get('motion',{}).get('revoked_candidate_pixels',0) for r in rs),
        observation_links=sum(r['stable_pixels'] for r in rs),
        strict_minimum_ROI=min((e['invalidation_coverage'] for r in rs for e in r.get('strict_refresh_score',{}).get('events',[])),default=None))
        for name,rs in rows_by.items()}
    write('motion_risk_ablation.json',dict(methods=motion,paired_event_summary=dict(fixed=len(fixed),remaining=len(remaining),new=len(new_failures)),
        input='Committed finite predicted velocity and source occupancy, measured gray change and failed tracks only',
        observation_links='Preserved independently from computation-candidate eligibility. No new acceptance authority.',
        stop='No further motion model or GT-dependent tuning in this round'))
    historical=read(Path('outputs/module1_source_state_refinement/replay_final/method_comparison.json'),{})
    controls={k:historical[k] for k in ('strict_surface','dense_observation') if k in historical}
    write('refresh_quality_comparison.json',dict(methods=methods,paired=paired,
        historical_controls=controls,historical_source='outputs/module1_source_state_refinement/replay_final/method_comparison.json',
        historical_scope='Same 96-frame population, frozen prior revision; retained controls, not fresh execution of this revision.',
        targets=dict(event=.998,point=.998,min_ROI=.9),
        refresh_scopes='Candidate complement is diagnostic; deployed strict refresh and M2 actual recompute are separate.'))
    runtime={}
    for name in methods:
        ts=[s for s in times if s['method']==name]
        runtime[name]={k:percentile([s[k] for s in ts]) for k in ('precheck_ms','commit_ms','total_ms')}
        runtime[name]['association_ms']=percentile([s['commit'].get('association_ms') for s in ts])
        runtime[name]['precheck_stages']={k:percentile([s['precheck'].get(k) for s in ts]) for k in
            ('temporal_correspondence','dense_flow','dense_validation','projection','computation_state_build','snapshot_fork')}
        runtime[name]['commit_stages']={k:percentile([s['commit'].get(k) for s in ts]) for k in
            ('commit_history_copy_ms','snapshot_fork_ms','cache_provenance_receipt_ms')}
    write('runtime_optimization_ablation.json',dict(methods=runtime,completion=completion,state_decisions_equal=state_equal,
        recommended_configuration='configs/worldstream_m1_engineering_frozen.yaml (measurement reuse on; motion revocation off)',
        observer='M1 Precheck/Commit wall excludes offline scoring/sampling and adapter preprocessing. Methods rotate each frame; one paired traversal, no independent seeds.',
        optimization='Invocation-local float32 grayscale and Sobel arrays; no association population/threshold/identity edits',
        timing='Threaded camera stage sums overlap. Do not sum their percentiles as request latency.',
        maintain_on_demand='Every arm keeps Precheck/Commit, histories, ordinary object IDs and source/cache receipts; on-demand may omit optional dense evidence after CheapAdmission rejects. Optional candidates/support/shadow diagnostic evidence are not promised identical.'))
    samples=jsonl(p/'physical_samples.jsonl');census=jsonl(p/'physical_reference_census.jsonl')
    denominators=jsonl(p/'sampling_denominators.jsonl')
    # Resolve actual nuScenes Log IDs from official local metadata. The replay
    # fallback field holds scene IDs; never count those as independent Logs.
    scene_path=Path('data/nuscenes/v1.0-trainval/scene.json')
    scene_logs={s['token']:s['log_token'] for s in json.loads(scene_path.read_text())} if scene_path.exists() else {}
    for s in samples+census+denominators:s['log']=scene_logs.get(s['scene'])
    physical={}
    for name in ('current_optimized','motion_only','final'):
        ss=[s for s in samples if s['method']==name];cs=[s for s in census if s['method']==name]
        physical[name]=dict(stratified_sample=physical_stats(ss),reference_intersection_census=physical_stats(cs),
            reference_census_by_camera={cid:physical_stats([s for s in cs if s['camera']==cid]) for cid in sorted({s['camera'] for s in cs})},
            reference_census_by_scene={scene:physical_stats([s for s in cs if s['scene']==scene]) for scene in sorted({s['scene'] for s in cs})},
            reference_census_by_log={log:physical_stats([s for s in cs if s['log']==log]) for log in sorted({s['log'] for s in cs if s['log'] is not None})},
            reference_census_by_origin={str(o):physical_stats([s for s in cs if s['origin']==o]) for o in (0,1,2)},
            by_origin={str(o):physical_stats([s for s in ss if s['origin']==o]) for o in (0,1,2)},
            census_by_motion={kind:physical_stats([s for s in cs if s['moving_reference']==flag]) for kind,flag in (('moving',True),('static',False))},
            by_camera={cid:physical_stats([s for s in ss if s['camera']==cid]) for cid in sorted({s['camera'] for s in ss})},
            by_context={context:physical_stats([s for s in ss if s['context']==context]) for context in PROTOCOL['strata']},
            by_scene={scene:physical_stats([s for s in ss if s['scene']==scene]) for scene in sorted({s['scene'] for s in ss})},
            time_baselines=percentile([s['baseline_s'] for s in ss]),
            inclusion_weighted=dict(scorable_link_mass=sum(s['weight'] for s in ss if s['scorable']),
                false_link_mass=sum(s['weight'] for s in ss if s['scorable'] and not s['correct']),
                warning='Weighted sample estimates only; unscorable mass never labelled correct'),
            original_sparse_FAR=dict(accepted=methods[name]['accepted_scorable_points'],false=methods[name]['false_stable_points'],
                FAR=methods[name]['sparse_reference_far'],meaning='Stationary refresh-candidate metric also rejects moving points; different from physical link correspondence'))
    # Pair exact link coordinates, not arbitrary nearest projected neighbours.
    def key(s):return (s['frame'],s['camera'],tuple(s['current']),tuple(s['source']))
    old={key(s):s for s in census if s['method']=='current_optimized'}
    latest={key(s):s for s in census if s['method']=='final'}
    pair_stats=dict(common=len(set(old)&set(latest)),old_only=len(set(old)-set(latest)),final_only=len(set(latest)-set(old)),
        revoked_correct=sum(old[k]['correct'] and old[k]['reuse_candidate_eligible'] and not latest[k]['reuse_candidate_eligible'] for k in set(old)&set(latest)),
        revoked_false=sum(not old[k]['correct'] and old[k]['reuse_candidate_eligible'] and not latest[k]['reuse_candidate_eligible'] for k in set(old)&set(latest)))
    write('candidate_physical_quality.json',dict(methods=physical,pairing=pair_stats,
        protocol=PROTOCOL,denominators=denominators,actual_log_ids=sorted(set(scene_logs.values())&{s['log'] for s in samples}),
        scene_log_map={s['scene']:s['log'] for s in samples},
        limits=['No dense ground truth.','No multi-sweep fusion or 2px-neighbour truth labels.',
            'Exact projected pixel still assumes rigid instance and sensor-time alignment; not physical certification.',
            'ROI-context strata are not foreground ownership labels.','Background without independent identity stays Unscorable.']))
    natural=summarize_gpu(g/'natural');repeat=summarize_gpu(g/'repeat');regression=junit(g/'regression_cuda.xml')
    dirty_examples=[]
    for run in read(g/'repeat/runs.json',[]):
        if run['mode']!='optimized_dirty_front' or run['warmup']:continue
        for frame in run['frames']:
            d=frame['runtime']
            dirty_examples.append(dict(frame=frame['frame'],**{k:d.get(k) for k in
                ('force_recompute_cameras','forced_camera_ids','verified_exact_cameras','eligible_exact_cameras',
                 'actual_reused_cameras','recomputed_cameras','fallback_reason','global_cache_reset_reason',
                 'cache_version','source_age','output_camera_provenance')}))
    write('m1_m2_integration.json',dict(natural=natural,controlled_repeat=repeat,
        controlled_single_camera_dirty=dirty_examples,
        experimental_lease_separation_production_default=False,
        matched='Camera boundary, existing zero-update cost qualification, weights, Exact gate and persistent M1 service',
        natural_speedup_claim=False))
    write('original_model_parity.json',dict(natural=natural['quality'],repeat=repeat['quality'],
        source_hashes_match=dict(natural=natural['all_runtime_sources_match'],repeat=repeat['all_runtime_sources_match']),
        source_audit=dict(natural=natural['source_audit'],repeat=repeat['source_audit']),
        status='passed' if natural['all_quality_pass'] and repeat['all_quality_pass'] else 'Not Complete'))
    write('regression_results.json',dict(CUDA=regression,shadow_state_equivalence=state_equal,compared_frames=len(byframe)))
    f=methods['final'];targets=f['critical_change_recall']>=.998 and f['critical_point_recall']>=.998 and f['minimum_ROI_refresh']>=.9
    engineering=(regression['status']=='passed' and state_equal and natural['all_quality_pass'] and repeat['all_quality_pass']
        and natural['all_runtime_sources_match'] and repeat['all_runtime_sources_match']
        and natural['source_audit']['status']=='passed' and repeat['source_audit']['status']=='passed')
    freeze=dict(engineering_interface_freeze='Go' if engineering else 'No-Go',candidate_quality_freeze='Go' if targets else 'No-Go',
        engineering_configuration='configs/worldstream_m1_engineering_frozen.yaml',
        experimental_motion_default=False,
        conditional_reuse_authorization='No-Go',formal_physical_certification='No-Go',handoff_to_M2='Go' if engineering else 'No-Go',
        remaining_M1_blockers=['Critical refresh targets unmet on exposed development population.' if not targets else 'Independent candidate physical qualification missing.',
            'Sparse-only centres largely have no exact independent physical reference.',
            'Missing historical anchors cannot provide motion envelopes for new/small objects.',
            'Scope-aware engineering configuration still has higher M1 P50 than old Source-Linked; no natural vision speedup.'],
        gates=dict(G1=dict(source_audit_natural=natural['source_audit']['status'],source_audit_repeat=repeat['source_audit']['status']),
            G2=dict(research_targets_pass=targets,fixed_old_events=len(fixed),remaining_old_events=len(remaining),new_failures=len(new_failures)),
            G3=dict(new_link_quality=physical['final']['by_origin']['1']),G4=dict(runtime=runtime,
                measurement_only_vs_current=dict(p50_improved=runtime['measurement_only']['total_ms']['p50']<runtime['current_optimized']['total_ms']['p50'],
                    p95_improved=runtime['measurement_only']['total_ms']['p95']<runtime['current_optimized']['total_ms']['p95']),
                engineering_vs_old_source=dict(p50_improved=runtime['measurement_only']['total_ms']['p50']<runtime['old_source']['total_ms']['p50'],
                    p95_improved=runtime['measurement_only']['total_ms']['p95']<runtime['old_source']['total_ms']['p95'])),
            G5=dict(exact_gates_changed=False,natural_camera_reuse=sum(a['actual_reused_cameras'] for a in natural['arms'].values())),
            G6=dict(CUDA=regression['status'],state_equivalence=state_equal),G7='No-Go'),
        qualification=dict(Calibration='Not Run',Heldout='Not Run',Final_Test='Not Run',Qualified_Certification=0,Qualified_FAR=None),
        stop='Freeze verified engineering interface; keep candidates research-only and stop stacking physical certification methods.')
    write('formal_go_nogo.json',freeze)
    md=f'''# M1 最终冻结判断

- engineering_interface_freeze: **{freeze['engineering_interface_freeze']}**
- candidate_quality_freeze: **{freeze['candidate_quality_freeze']}**
- conditional_reuse_authorization: **No-Go**
- formal_physical_certification: **No-Go**
- handoff_to_M2: **{freeze['handoff_to_M2']}**

工程接口可冻结的条件是当前源码的 CUDA、完整原模型、来源和持久状态回归通过；它不授予候选跳算或物理认证。候选仍有 {len(remaining)} 个旧失败事件，另有 {len(new_failures)} 个新失败事件。新出现或微小目标缺少历史 Anchor 时，不能由局部自洽光流推导静止或输入相同。

保持严格刷新、Exact 输入及依赖闭包、现有成本/任务门。M2 可继续研究真实计算依赖与条件复用；不得把普通 Observation Link 当作物理真值或安全跳算证明。停止继续叠加 M1 运动模型或认证生产器。

推荐工程配置 configs/worldstream_m1_engineering_frozen.yaml 仅启用实测获益的 measurement_reuse；motion_risk_revocation=False。projects/configs/OmniDrive/worldstream_m2_final_validation.py 指向该配置。启用风险的完整方法只保留为独立实验对照，不列为质量改善或生产加速。
'''
    (root/'m1_freeze_decision.md').write_text(md)
    figures(root,methods,runtime,natural,repeat)
    table=['| 方法 | Candidate % | 接受中错误/接受 | Event/Point Recall % | 最低 ROI % | M1 P50/P95 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in methods.items():
        t=runtime[name]['total_ms'];table.append(f"| {name} | {m['effective_stable_area']*100:.6f} | {m['false_stable_points']}/{m['accepted_scorable_points']} | {m['critical_change_recall']*100:.4f}/{m['critical_point_recall']*100:.4f} | {m['minimum_ROI_refresh']*100:.4f} | {t['p50']:.2f}/{t['p95']:.2f} |")
    q=physical['final'];current=runtime['current_optimized'];optimized=runtime['final']
    model_table=['| 人口/方法 | 请求 | E2E P50/P95 ms | 实际 Camera 复用 | 任务门 |','|---|---:|---:|---:|---:|']
    for case,ss in (('自然',natural),('受控重复',repeat)):
        for name,a in ss['arms'].items():
            t=a['E2E_wall_ms'];quality=ss['quality'].get(name)
            model_table.append(f"| {case}/{name} | {a['requests']} | {t['p50']:.2f}/{t['p95']:.2f} | {a['actual_reused_cameras']} | "+(f"{quality['frozen_target_pass']}/{quality['checks']}" if quality else 'baseline')+' |')
    report=f'''# M1 动态风险、候选物理评价与最终性能优化

冻结对照为实际 HEAD 3113df2，工作区开始时干净；历史输出未覆盖。本轮使用相同 96 帧、576 个相机 Development 观测，统一标记 exploratory_reused_population。线上只读取已提交历史与当前真实 RGB/标定/时间；Annotation/LiDAR 在 Precheck 后只供离线分析、分层与评分。

14 个旧失败全部逐帧重放：10 个没有重叠的历史投影，3 个存在重叠投影，1 个存在被动态排除清除的已测变化。重叠的类别/投影不证明与 GT 对象为同一身份。原始变化、排除、稠密/稀疏测量、残差和 GT 均在 critical_event_root_cause.json 与 critical_event_overlays/ 保存。

实现 Source-Aware Motion Risk Revocation：在已提交且时间/相机/签名相符的有限历史运动占用范围内，实测灰度变化撤销计算候选；源占用冲突或失败对应仅给 Unknown。历史 Box/Hull 不作前景或真值。Observation Link 和 Identity 保留，reuse_candidate_eligible 单独输出，M2 Exact gate 不变。动态排除前的变化另存风险通道，未改旧身份/严格刷新输入。

'''+'\n'.join(table)+f'''

旧 14 个失败修复 **{len(fixed)}** 个，仍失败 **{len(remaining)}** 个，新失败 **{len(new_failures)}** 个；候选相对面积损失 {paired['candidate_relative_loss']*100:.4f}%。逐事件、同一卡车连续失败和 CAM_FRONT 行人配对在 candidate_paired_analysis.json。严格部署刷新最低 ROI 为 {motion['final']['strict_minimum_ROI']}，候选补集失败不等于发生了非法跳算。

运动风险没有表现出事件召回或已评分错误拒绝的独立收益，不继续添加模型。工程配置 worldstream_m1_engineering_frozen.yaml 保持该分支关闭，采用测量复用单项优化；风险开启的 final 是实验全方法，不能与推荐工程配置混称。

物理评价预先固定随机种子、每层 32 个链接上限和完整分母/权重。最终随机样本 {q['stratified_sample']['samples']}，可评分 {q['stratified_sample']['scorable']}，Correct/False={q['stratified_sample']['correct']}/{q['stratified_sample']['false']}。精确参考交集 census 可评分 {q['reference_intersection_census']['scorable']}，Correct/False={q['reference_intersection_census']['correct']}/{q['reference_intersection_census']['false']}。两者分开，不用 census 冒充随机样本覆盖。稀疏 FAR 仍保留“接受中错误”原口径，运动对应即使物理可跟踪也不证明无需刷新。背景无独立身份参考仍 Unscorable；没有使用 2px 邻近真值或未经运动补偿的多帧点云。实例刚体/可见性与传感器时差假设仍限制评分能力，置信区间仅描述性，不形成认证。

96 帧的 Shadow/Object 状态决策摘要等价：{state_equal}。优化未修改身份生产器、关联候选、阈值或 StateStore。cProfile 确认身份测量主导 Commit，复用单调用灰度 float32/Sobel 中间结果后，当前优化到最终方法 M1 P50/P95 为 {current['total_ms']['p50']:.2f}/{current['total_ms']['p95']:.2f} → {optimized['total_ms']['p50']:.2f}/{optimized['total_ms']['p95']:.2f} ms。运动与测量单独消融见 runtime_optimization_ablation.json；线程阶段有重叠，不相加推导请求时延。

工程推荐的 measurement_only 对照为 {runtime['measurement_only']['total_ms']['p50']:.2f}/{runtime['measurement_only']['total_ms']['p95']:.2f} ms，优于当前 Optimized 的 P50/P95；相对更早 old_source 的 {runtime['old_source']['total_ms']['p50']:.2f}/{runtime['old_source']['total_ms']['p95']:.2f} ms，P50 仍较慢，P95 改善。不能宣称新接口全面优于旧接口。身份生产器的协议摘要绑定源码和配置，新标志使诊断签名变化；实际状态决策与 M2 计算来源另行核验，不把这些签名字符串宣称逐位相同。

'''+'\n'.join(model_table)+f'''

本轮 CUDA 回归：{regression.get('passed',0)}/{regression.get('total',0)}。真实完整模型包括 EVA、Detection、Map、Memory、原 LLM 文本和规划输出。全部计时包含统一诊断特征 CPU 拷贝，原生 Memory 写盘在计时外；基线先执行，其余臂旋转，单次测量重复不支持小幅速度差的统计显著性结论。新旧接口使用同 Camera Boundary、成本表、数值/任务质量门；maintain 每帧执行 Precheck/Commit，on-demand 可省可选 Dense 验证但持续维护历史身份。自然输入无合法实际跳算时不宣称视觉计算加速。GPU 内存、传输字节、Adapter 和开销详见 m1_m2_integration.json。

模型臂中 current_optimized 是上一轮完整优化配置，measurement_only 是本轮推荐工程配置，optimized_exact 是包含风险撤销的实验完整配置，optimized_on_demand 在其上使用 CheapAdmission。full_maintain 持续维护相同 M1 服务并完整重算视觉。on-demand 的等价边界是必要历史维护、普通 Object ID、来源和缓存回执；可选 Dense 支持、候选与 Shadow 诊断证据不承诺逐位相同，不把省略这些证据伪装成完全相同的观测服务。

Dense Observation 与 Strict Surface 保留为上一轮相同 96 帧历史对照，原始结果和源码版本不改；数值嵌入 refresh_quality_comparison.json 的 historical_controls，明确不是本轮新执行，时延不与本轮配对优化混用。

工程接口冻结：**{freeze['engineering_interface_freeze']}**；候选质量冻结：**{freeze['candidate_quality_freeze']}**；转入 M2 研究：**{freeze['handoff_to_M2']}**。条件复用授权、正式物理认证均 **No-Go**。Calibration/Heldout/Final Test = Not Run，Qualified Certification=0，Qualified FAR=null。当前证据不足的风险不由 GT 补齐，不再堆叠 M1 物理认证方法。

![Risk and coverage](figures/risk_coverage.png)
![Paired failed events](figures/paired_events.png)
![Maintenance runtime](figures/runtime.png)
![Independent exact admission](figures/admission.png)

实现和采样修改清单见 implementation_changes.md；复现命令见 commands.sh；最终验收见 artifact_validation.json。
'''
    (root/'experiment_report.md').write_text(report)
    print(json.dumps(dict(engineering=engineering,targets=targets,old_fixed=len(fixed),old_remaining=len(remaining),state_equal=state_equal)))


def figures(root,methods,runtime,natural,repeat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(folder/name,dpi=150);plt.close(fig)
    names=list(methods)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for n,m in methods.items():
        x=m['effective_stable_area']*100
        axes[0].scatter(x,(1-m['critical_point_recall'])*100,label=n)
        if m['sparse_reference_far'] is not None:
            axes[1].scatter(x,m['sparse_reference_far']*100,label=n)
    axes[0].set_ylabel('Missed critical points (%)');axes[0].set_title('Candidate-complement refresh diagnostic')
    axes[1].set_ylabel('Conditional sparse FAR (%)');axes[1].set_title('Scorable references only; no population guarantee')
    for ax in axes:ax.set_xlabel('Candidate coverage (%)');ax.legend(fontsize=7)
    save(fig,'risk_coverage.png')
    ps=read(root/'replay/paired_events.json',[])
    fig,ax=plt.subplots(figsize=(12,4))
    keys=sorted({(s['frame'],s['camera'],s['instance']) for s in ps})
    for n in ('current_optimized','motion_only','final'):
        lookup={(s['frame'],s['camera'],s['instance']):s['invalidation_coverage'] for s in ps if s['method']==n}
        ax.plot(range(len(keys)),[lookup[k] for k in keys],marker='.',label=n)
    ax.axhline(.9,color='red',ls='--');ax.set_xticks(range(len(keys)),[f'{k[1]}\n{k[0][:5]}' for k in keys],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Same historical failure ROI refresh');ax.legend();save(fig,'paired_events.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for k,label in (('precheck_ms','Precheck'),('commit_ms','Commit'),('association_ms','Association')):
        axes[0].plot(range(len(names)),[runtime[n][k]['p50'] for n in names],marker='.',label=label+' P50')
        axes[1].plot(range(len(names)),[runtime[n][k]['p95'] for n in names],marker='.',label=label+' P95')
    for ax in axes:ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.set_ylabel('ms');ax.legend(fontsize=8)
    save(fig,'runtime.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,ss,label in ((axes[0],natural,'natural'),(axes[1],repeat,'controlled repeat')):
        ns=[n for n in ss['arms'] if n not in ('original','full_maintain')];x=np.arange(len(ns))
        for offset,key,title in ((-.2,'candidate_cameras','Candidate cameras'),(0.,'verified_exact_cameras','Exact cameras'),(.2,'actual_reused_cameras','Actual reuse')):
            ax.bar(x+offset,[ss['arms'][n][key] for n in ns],.2,label=title)
        ax.set_xticks(x,ns,rotation=30,ha='right',fontsize=7);ax.set_title(label);ax.set_ylabel('Independent camera unit counts');ax.legend(fontsize=7)
    save(fig,'admission.png')


if __name__=='__main__':main()
