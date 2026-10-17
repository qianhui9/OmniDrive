"""Evidence-linked v4 reporting, including explicit gated/unmeasured results."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .audit import write
from .streaming_replay import distribution, run as replay

ROOT=Path('outputs/system_optimization_v4')
PREVIOUS=Path('outputs/system_optimization_v3')


def read(path):return json.loads(Path(path).read_text())


def paired(rows, mode, reference='m1'):
    base={(r['dataset_identity']['sample_token'],r['repeat']):r for r in rows if r['mode']==reference}
    groups=defaultdict(list)
    for row in rows:
        if row['mode']!=mode:continue
        token=row['dataset_identity']['sample_token'];ref=base[(token,row['repeat'])]
        groups[row['dataset_identity']['scene_token']].append((ref['frame']['e2e_wall_ms'],row['frame']['e2e_wall_ms']))
    pairs={s:dict(frames=len(v),reference_mean_ms=float(np.mean([p[0] for p in v])),candidate_mean_ms=float(np.mean([p[1] for p in v]))) for s,v in groups.items()}
    if not pairs:return None
    a=np.array([[v['reference_mean_ms'],v['candidate_mean_ms']] for v in pairs.values()]);delta=a[:,1]-a[:,0]
    rng=np.random.RandomState(20261017);idx=rng.randint(0,len(a),(10000,len(a)));samples=a[idx].mean(1)
    return dict(mode=mode,reference=reference,scenes=len(a),paired_frames=sum(v['frames'] for v in pairs.values()),
        mean_scene_increment_ms=float(delta.mean()),scene_mean_speedup=float(a[:,0].mean()/a[:,1].mean()),
        scene_bootstrap_increment_95ci_ms=np.percentile(samples[:,1]-samples[:,0],[2.5,97.5]).tolist(),
        scene_bootstrap_speedup_95ci=np.percentile(samples[:,0]/samples[:,1],[2.5,97.5]).tolist(),scene_pairs=pairs)


def source_analysis():
    path=PREVIOUS/'m1_false_accept_diagnosis.json';old=read(path);cases=old['cases']
    angles=[p['angle_deg'] for c in cases for camera in c.get('cameras',[]) for p in camera.get('bidirectional',{}).get('points',[]) if 'angle_deg' in p]
    sources=[c['source_center_error_m'] for c in cases if c.get('source_center_error_m') is not None]
    value=dict(status='source_code_audit_and_prior_exploratory_evidence',new_independent_physical_experiment=False,
        prior_source=str(path),prior_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        old_exploratory_cases=len(cases),old_exploratory_scenes=old['scene_count'],
        old_primary_mechanisms=old['primary_mechanisms'],source_center_error_m=distribution(sources),
        old_correspondence_parallax_deg=distribution(angles),
        source_head='StreamPETR pts_bbox_head.get_bboxes -> boxes_3d.gravity_center / corners -> ego_to_world in outputs_from_model',
        coordinate_contract='decoded gravity center (not bottom center); actual pose world transform',
        localization_uncertainty='geometry_uncertainty defaults to 0.0 but adapter does not estimate a localization error bound; zero does NOT certify zero error',
        source_score='scores_3d detection confidence only, not a metric localization probability',
        bias_propagation='predict_geometry transports the source estimate until fresh perception; commit replaces geometry with the current actual prediction, never fuses observations into a debiased estimate',
        multiframe='history_observations counts associations; last_innovation_m measures prediction disagreement, neither proves independent accuracy',
        parallax='old measured angles above are exploratory only; new scenes not evaluated for physical triangulation',
        foreground='LK/gray patch + box membership do not independently establish target foreground ownership',
        dynamic='adapter trust_detection_velocity=False; static triangulation rejects dynamic classes',
        ambiguity='same-class Hungarian/nearest geometry can confuse nearby cones/cars; association does not establish annotated identity',
        independent_correction_available=False,online_gt_used=False,
        conclusion='prior 29 cases are consistent with source localization bias; no new independent causal or physical-quality success claimed',
        sources=['worldstream/m1/state_types.py','worldstream/m1/omnidrive_adapter.py',
            'worldstream/m1/state_manager.py','worldstream/m1/observation_validator.py',
            'worldstream/m1/bidirectional_geometry.py','worldstream/m1/geometry.py','worldstream/m1/state_store.py'])
    write(ROOT/'source_geometry_error_analysis.json',value)
    return value


def gated_results():
    prior=read(PREVIOUS/'task_consumption_results.json')
    common=dict(status='not_run_gate_a',reason='No established input-complete original task; user requires staged No-Go before B/C online changes',
        new_independent_evidence=False,online_gt_used=False,controlled_task_state_probe=dict(status='not_run',replacement_of_original_workload=False))
    if not (ROOT/'task_fact_results.json').exists() or read(ROOT/'task_fact_results.json').get('status')!='completed_independent_offline_diagnostic':
        write(ROOT/'task_fact_results.json',dict(common,acceptance=None,accuracy=None,valid_fact_recovery=None,
            physical_false_acceptance=None,spatial_error=None,motion_error=None,task_fact_coverage=None,
            independent_scene_support=None,verification_wall_ms=None))
        write(ROOT/'task_fact_risk_coverage.json',dict(common,far_upper95=None,physical_quality_pass=False,
            frozen_risk=read(PREVIOUS/'protocol.json')['physical_risk'],
            minimum_zero_error_independent_accepted_samples=math.ceil(math.log(.05)/math.log(.95)),
            new_tolerances_frozen=False,scope='no new fact candidate selected; old failures are not independent success evidence'))
    write(ROOT/'task_state_consumption.json',dict(common,new_actual_consumption=None,
        prior_evidence=dict(source=str(PREVIOUS/'task_consumption_results.json'),queries=prior['actual_queries'],
            activations=prior['actual_query_activations'],consumed_records=prior['actual_consumed_record_occurrences']),
        historical_value_over_current_perception='unestablished; v4 does not substitute historical inputs'))
    write(ROOT/'effective_token_reduction.json',dict(common,actual_stateful_prefill_tokens=None,
        actual_skipped_prefill_tokens=None,prior_actual_activations=0,
        v4_d_execution='all original visual carriers and generation retained; scheduling changes no language input'))
    write(ROOT/'llm_computation_savings.json',dict(common,actual_historical_prefill_savings_ms=None,
        actual_historical_decode_savings_ms=None,avoided_language_computation=False,
        overlap_scope='CPU M1 and full EVA execute concurrently; same language computation remains'))


def figures(audit,performance,stream):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=180);fig.savefig(dest/(name+'.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));items=list(audit['categories'].items())
    ax.barh([k for k,v in items],[v for k,v in items]);ax.set_xlabel('Original question occurrences (lexical screening only)');ax.invert_yaxis()
    save(fig,'task_dependency_inventory')
    fig,ax=plt.subplots(figsize=(8,4));pairs=performance['scene_paired']
    for j,p in enumerate(pairs):
        values=list(p['scene_pairs'].values())
        ax.scatter([j]*len(values),[v['candidate_mean_ms']-v['reference_mean_ms'] for v in values],alpha=.7)
        ci=p['scene_bootstrap_increment_95ci_ms'];ax.plot([j,j],ci,color='black');ax.scatter(j,p['mean_scene_increment_ms'],marker='D',color='black')
    ax.axhline(0,color='gray',linewidth=1);ax.set_xticks(range(len(pairs)));ax.set_xticklabels([p['mode'] for p in pairs]);ax.set_ylabel('Scene paired increment vs persistent serial (ms)')
    save(fig,'scene_paired_e2e')
    fig,ax=plt.subplots(figsize=(8,4))
    for mode in ('m1','overlap','unified_auto'):
        events=[e for r in stream['scenes'] if r['mode']==mode for e in r['events']]
        by_ordinal=defaultdict(list)
        for row in stream['scenes']:
            if row['mode']==mode:
                for j,e in enumerate(row['events']):by_ordinal[j].append(e['output_age_s'])
        ax.plot(sorted(by_ordinal),[np.mean(by_ordinal[i]) for i in sorted(by_ordinal)],marker='o',label=mode)
    ax.set_xlabel('Original frame ordinal');ax.set_ylabel('Mean output age after FCFS replay (s)');ax.legend();save(fig,'streaming_output_age')


def main():
    audit=read(ROOT/'task_state_feasibility.json');source=source_analysis();gated_results()
    facts=read(ROOT/'task_fact_results.json')
    if facts.get('status')!='completed_independent_offline_diagnostic':raise RuntimeError('Complete frozen offline fact diagnostic before final reporting')
    if read(ROOT/'gpu_status.json')['status']!='completed':raise RuntimeError('Formal GPU execution incomplete; do not publish a final performance report')
    paths=sorted(ROOT.glob('final_test_*/frames.jsonl'));rows=[]
    for path in paths:
        status=read(path.parent/'status.json')
        if status['status']!='completed':raise RuntimeError('Partial final test')
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows=[r for r in rows if not r['warmup']]
    # Verify stable frozen identities, no source-scene reuse, no question edits.
    protocol=read(ROOT/'protocol.json');splits=[{s['scene'] for s in protocol['selections'][split]} for split in ('calibration','heldout','final_test')]
    assert not any(a&b for i,a in enumerate(splits) for b in splits[i+1:])
    old={c['oracle']['scene'] for c in read(PREVIOUS/'m1_false_accept_diagnosis.json')['cases']}
    assert not old&set.union(*splits)
    actual_tokens={r['dataset_identity']['sample_token'] for r in rows}
    expected_tokens={t for s in protocol['selections']['final_test'] for t in s['tokens']}
    assert actual_tokens==expected_tokens
    assert all(len(r['tasks'])==1 and r['tasks'][0]['question']=='Please provide the planning trajectory for the ego car without reasons.' for r in rows)
    auto=[r for r in rows if r['mode']=='unified_auto'];admitted=[r for r in auto if r['frame']['admission']['allowed']]
    quality_rows=[]
    for r in rows:
        q=r['frame']['quality']
        quality_rows.append(dict(mode=r['mode'],token=r['dataset_identity']['sample_token'],scene=r['dataset_identity']['scene_token'],
            passed=bool(q['exact_token_equality'] and q['token_count_matches'] and q['exact_perception_memory_digest']
                and all(v['max_abs']==0 and not v['shape_mismatches'] for v in q['numerical'].values())),
            planning=r['tasks'][0]['planning']))
    strict=all(r['passed'] for r in quality_rows)
    state_checks=[]
    for path in paths:
        d=read(path.parent/'strict_quality.json');state_checks.extend(d['pairs'])
    state_pass=bool(state_checks) and all(r['passed'] for r in state_checks)
    write(ROOT/'task_quality_results.json',dict(status='completed_original_planning_full_carrier',
        complete_tokens_perception_temporal_exact=strict,persistent_state_same_service_exact=state_pass,
        scope='Output equality for fixed original inputs; no new physical-state truth or compact/direct-query quality certificate',
        rows=quality_rows,state_pairs=state_checks,
        original_vqa=dict(status='offline_inventory_only',quality_rerun=False),
        compact_context=dict(status='not_run_gate_a'),direct_query=dict(status='not_run_gate_a')))
    stats={}
    for mode in sorted({r['mode'] for r in rows}):
        part=[r for r in rows if r['mode']==mode]
        stats[mode]=dict(frames=len(part),service=part[0]['frame']['service'],
            e2e_wall_ms=distribution([r['frame']['e2e_wall_ms'] for r in part]),
            cost_lookup_wall_ms=distribution([r['frame']['admission']['lookup_wall_ms'] for r in part]),
            actual_overlap_frames=sum(bool(r['frame']['m1'].get('overlap_active')) for r in part),
            exact_tokens=all(r['frame']['quality']['exact_token_equality'] for r in part))
    pairs=[paired(rows,m) for m in ('overlap','unified_auto')]
    value=dict(status='completed_fresh_final_test',methods=stats,
        accepted_workload_coverage=len(admitted)/len(auto),actual_admitted_frames=len(admitted),
        actual_overlap_frames=sum(bool(r['frame']['m1'].get('overlap_active')) for r in auto),
        total_auto_frames=len(auto),fallback_reasons=dict(Counter(r['frame']['admission']['reason'] for r in auto)),
        quality_pass=strict and state_pass,scene_paired=pairs,cost_calibration=read(ROOT/'cost_calibration.json'),
        previous_v3=dict(actual_admitted_frames=1,total_auto_frames=16,coverage=1/16,
            attribution='different fresh scenes and balanced calibration support; no causal claim that the lookup code alone increased coverage or reduced runtime'),
        certificate_scope='current instrumented implementation only; no inheritance by uninstrumented deployment',
        exact_m2_scope='full EVA; Task Fact never establishes attention dependency equality',
        stateful_language_speedup_claim=False)
    write(ROOT/'auto_cost_results.json',value)
    ablations=[dict(name='Original OmniDrive / M3 Full Carrier',mode='original',status='measured',service='current_output'),
        dict(name='Persistent Serial / M1 Original State',mode='m1',status='measured',service='persistent_state'),
        dict(name='Persistent Overlap',mode='overlap',status='measured',service='persistent_state'),
        dict(name='Unified Auto',mode='unified_auto',status='measured',service='persistent_state')]
    ablations.append(dict(name='M1 Task Fact',status='measured_offline_only',evidence='task_fact_results.json',online_consumption=False))
    for name in ('M3 Task Context','M3 Direct Query','M1 + M3 historical consumption','M1 + M2 + M3 task-state execution'):
        ablations.append(dict(name=name,status='not_run_gate_b',reason='Offline fact diagnostic has zero certified acceptance; preserve staged stop'))
    write(ROOT/'system_ablation.json',dict(ablations=ablations,measured_methods=stats,
        previous_module_ablations_source=str(PREVIOUS/'experiment_report.md'),
        prior_only_not_new=dict(m12='v3 main',m123='v3 main',task_full='v3 main/multi',task_compact='v3 main/multi'),
        controlled_task_state_probe=dict(status='not_run',original_workload_replaced=False)))
    stream=replay(ROOT)
    auto_pair=pairs[1];net_pass=auto_pair['scene_bootstrap_increment_95ci_ms'][1]<0 and strict and state_pass
    go=dict(gate_a=audit['gate_a'],gate_b=dict(decision='No-Go: zero certified facts in independent offline diagnostic',independent_reliable_fact_established=False,
            evidence='task_fact_results.json',frames=facts['frames'],scenes=facts['scenes']),
        gate_c=dict(decision='not_run_gate_b',language_computation_reduction_established=False),
        gate_d=dict(task_state='No-Go; no certified language substitution',overlap_auto='Go within measured fixed workload' if net_pass else 'No-Go for stable whole-workload Auto benefit',
            paired=auto_pair),gate_e=dict(decision='Refocus on dependency-safe execution, CPU/GPU overlap and validated workload scheduling',
            workload_change='Do not replace original Planning/VQA with controlled tasks; any new fact-friendly workload must be a separately declared research objective'),
        complete_world_state_closed_loop_approved=False,old_results_overwritten=False)
    write(ROOT/'go_nogo_decisions.json',go)
    figures(audit,value,stream)
    lines=['# OmniDrive 第四轮系统完善实验报告','',
        '**结论：原始 Planning 的 Task-Sufficient World State 替代路径 No-Go；少量真实 VQA 距离问题允许离线 B 诊断，但零认证覆盖使 Gate B No-Go，停止 C 在线替代。完成原始任务审计、严格准入优化、真实 GPU 重测与流式回放。**','',
        f'读取 {audit["frames"]} 帧、{audit["independent_scenes"]} 场景的原始任务，conv/VQA {audit["vqa_screening"]["total"]} 条。局部词法候选 {audit["vqa_screening"]["semantic_candidates"]} 条；已建立完整任务依赖与独立质量证据 0 条。词法筛查不代表候选均可替代，也不证明所有 VQA 都不可能。',
        '默认 Planning 单独保留，6,019 条均需要完整当前场景。未改变问题、权重、LoRA、Sampling、答案长度或质量阈值。', '',
        '## 新 GPU 证据','',
        'Calibration / Heldout / Final Test 各 6 个独立场景，每城 3 个，均未使用 v3 场景或旧 29 例所在场景。每场景 8 个连续 Keyframe。最终 48 唯一帧、192 方法帧；另有原生参考执行，不计独立样本。校准与验证另有 192 方法帧。',
        'CPU 全程排除 4/5；解释器为 OmniDrive。Checkpoint/嵌入 LoRA 全文件哈希、代码快照、环境与线程亲和性证据均保存。场景切分使用本地验证集，不是官方隐藏测试。','',
        '| 方法 | 服务合同 | 帧 | E2E P50/P95 ms | Lookup P50/P95 ms | 实际重叠帧 |',
        '|---|---|---:|---:|---:|---:|']
    for mode,s in stats.items():
        e=s['e2e_wall_ms'];c=s['cost_lookup_wall_ms']
        lines.append(f'| {mode} | {s["service"]} | {s["frames"]} | {e["p50"]:.2f}/{e["p95"]:.2f} | {c["p50"]:.3f}/{c["p95"]:.3f} | {s["actual_overlap_frames"]} |')
    lines+=['',f'完整 Token/感知/时序张量严格一致：{strict}；同服务完整 M1 状态摘要一致：{state_pass}。Full Carrier 输出相等不构成 Compact/Direct Query 质量证据。',
        f'Auto 实际批准 {len(admitted)}/{len(auto)} 帧（{len(admitted)/len(auto):.2%}），质量通过且实际重叠 {value["actual_overlap_frames"]} 帧。新批准成本键 {value["cost_calibration"]["approved_keys"]} 个；完整拒绝原因见 auto_cost_results.json。', '',
        '| 同服务比较 | 场景配对平均增量 ms | 95% 场景 Bootstrap CI ms | 时间比 |','|---|---:|---|---:|']
    for p in pairs:lines.append(f'| {p["mode"]} / m1 | {p["mean_scene_increment_ms"]:.2f} | {p["scene_bootstrap_increment_95ci_ms"]} | {p["scene_mean_speedup"]:.4f} |')
    lines += ['', '## 持续输入回放','',
        '使用本轮实际每帧 E2E 服务时间，真实传感器时间作为到达时刻，FCFS 单 Worker，所有请求保留，逐场景独立计算。下表是有限轨迹驱动回放，不是在线吞吐实测，也未验证积压时历史状态的实时新鲜度质量。', '',
        '| 方法 | 请求 | 最后到达时完成/待处理 | 输出年龄 P50/P95 s | 排空完成率 req/s | 0.5s Deadline Miss |','|---|---:|---:|---:|---:|---:|']
    for mode,s in stream['modes'].items():
        a=s['output_age_s'];lines.append(f'| {mode} | {s["requests"]} | {s["completed_at_last_arrival"]}/{s["pending_at_last_arrival"]} | {a["p50"]:.2f}/{a["p95"]:.2f} | {s["effective_completed_per_s_including_drain"]:.3f} | {s["deadline_miss_rate_after_drain"]:.2%} |')
    lines += ['', '未分配真实排队图像，不编造队列内存增长；该指标为 null。资源监测记录实际进程 RSS 和缓存负载，终生高水位不作为优化净内存增量。', '',
        '## 十一项回答','',
        f'1. 原始 Planning 可建立的替代覆盖 0/6,019；VQA 筛查候选 {audit["vqa_screening"]["semantic_candidates"]}/{audit["vqa_screening"]["total"]}，建立完整事实依赖 0，余下可行性未知。详见 task_dependency_audit.md。',
        '2. 旧 29 例诊断与源帧 3D 定位偏差一致。当前 Head 实际 gravity_center 为历史来源；检测 score 不是测距误差概率。commit 使用新预测替换几何，观测次数不实施几何融合。旧例是探索数据，不能计为新成功证据。',
        f'3. 本轮对 {facts["frames"]} 帧/{facts["scenes"]} 新场景、{facts["overall"]["trials"]} 条 Fact 候选实际完成离线提取和独立标注评分；认证接受 {facts["overall"]["accepted"]}，覆盖 0，准确率和 FAR 未定义。提取全部完成后才加载 Oracle；旧 29 例未参与。即使理想独立零错误也至少需 59 接受样本才可能使单侧 95% FAR 上界 <5%，还需场景独立支持。',
        '4. 有独立身份、前景归属、校准投影符号区间时，方向/持续性可能无需完整 3D；本轮未认证这类 Fact。距离必须独立误差界，动态关系必须时序运动证据。',
        '5. 没有新增 M1→M3 历史状态消费。本轮未实施状态替代；v3 原始 224 查询/0 激活作为旧证据单列，不冒充 v4 测量。',
        '6. 相对仅使用当前 Perception，历史状态增加的任务价值未建立。M1 历史维护服务及与 EVA 并发是实际执行职责，不是语言输入替代。',
        '7. 没有状态化 Token 或 Prefill/Decode 计算减少。D 保留完整 Carrier 和原生生成；不把时延噪声归为语言缓存收益。',
        f'8. Task-State 完整闭环不成立。持久服务重叠收益与 Auto 全负载收益分开判断；Auto Gate D 为：{go["gate_d"]["overlap_auto"]}。Current-output 的原始模型承担不同历史职责，不能作为同服务加速基线。',
        f'9. 新 Auto 覆盖 {len(admitted)}/{len(auto)}；上轮 1/16 仅作描述对照。新场景、平衡支持和实现同时变化，不能将覆盖变化单独归因于缓存代码。每次仍检查实际 gap、证书内容、模型/线程/插桩合同及当次开销。',
        '10. 流式回放数值见上表及逐事件 traces/streaming_events.jsonl。相同请求数下比较输出年龄；它不证明真实在线新鲜度通过或无限持续吞吐稳定，数秒服务时间仍无法满足 0.5s 到达节奏。',
        '11. 建议论文主线收敛到依赖安全的执行、CPU/GPU Overlap 和经验证的调度。若研究局部事实任务，应另立工作负载和质量协议，不能替换原始 Planning/VQA 后称原负载加速。', '',
        '## 修改、阶段门与复现','',
        '- A：新增 tools/system_optimization/task_dependency_audit.py；只读原始问题、逐条记录依赖筛查，No-Go 后不改 M3 输入。',
        '- B：新增 worldstream/m1/task_facts.py 最小独立 Fact 接口，当前缺少独立身份/前景/测距/运动误差界时维持 Unknown；v4_facts.py 实际离线评分，容限在查看新 Fact Oracle 前冻结。没有新增三角化阈值或把检测 score 转为测距置信度。Gate B No-Go。',
        '- C：因零可靠 Fact 覆盖，任务检索/Prompt/Carrier 替代均未接入线上，相关消融明确 not_run_gate_b。',
        '- D：worldstream/runtime.py 按固定执行域缓存候选，实际 gap 和实际成本每次重验；worldstream/m3/visual_state.py 在单次检查内去重共享函数指纹，无跨请求隐藏失效。未改 Overlap Worker/生命周期/M2 依赖语义。',
        '- E：新增 streaming_replay.py；FCFS、所有请求保留、真实时间输入、未测内存留空。',
        '- 新测试覆盖区间缓存首次拒绝后恢复、超域/开销拒绝、原地证书改动失效、函数默认参数变化、流队列边界和非法时间输入。完整回归日志见 logs/。',
        '- 新证书仅绑定本轮插桩/代码/资源/合同；不授权普通无插桩部署。未调整旧协议、覆盖旧结果或改动权重。',
        '- 没有声称状态闭环或真实 VQA 替代成功。独立物理和任务质量证据仍是后续前置条件。',
        '', '复现命令见 commands.sh；冻结协议 protocol.json；各次执行 source_manifest.json、原始 frames.jsonl 与 cross_module_trace.jsonl 位于对应场景目录。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(auto_coverage=value['accepted_workload_coverage'],quality=strict and state_pass,paired=pairs),indent=2))


if __name__=='__main__':main()
