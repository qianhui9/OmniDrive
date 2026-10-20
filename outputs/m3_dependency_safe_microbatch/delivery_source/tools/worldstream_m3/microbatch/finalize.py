"""Conservative report from actual service, numerical witnesses and CUDA tests."""
from collections import Counter,defaultdict
import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT,OLD,read,write_json,stats


def rows(path):
    import json
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));args=p.parse_args();root=Path(args.output)
    service=read(root/'service/status.json');root_cause=read(root/'sgvd_numerical_root_cause.json')
    census=read(root/'native_question_dependency_census.json');feasible=read(root/'microbatch_gpu_feasibility.json')
    numerical=read(root/'microbatch_numerical_parity.json')
    census['ready_group_size_distribution']={str(k):dict(Counter(str(len(group)) for frame in census['frames']
        for group in frame['ready_groups'][str(k)])) for k in (1,2,4,8)}
    write_json(root/'native_question_dependency_census.json',census)
    runs=rows(root/'service/runs.jsonl');checks=rows(root/'service/quality.jsonl')
    testcases=list(ET.parse(root/'regression_cuda.xml').getroot().iter('testcase'))
    regression=dict(tests=len(testcases),failures=sum(t.find('failure') is not None for t in testcases),
        errors=sum(t.find('error') is not None for t in testcases),skipped=sum(t.find('skipped') is not None for t in testcases),cuda=True)
    regression['passed']=regression['tests']>0 and not any(regression[k] for k in ('failures','errors','skipped'))
    write_json(root/'regression_results.json',regression)
    table=[];eos=read(OLD/'phase1/protocol.json')['default_generation']['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos or []
    runtime=[];fallbacks=Counter();batches=Counter();isolation=[]
    for r in runs:
        d=r['m3'].get('sampling_microbatch',{})
        if d:
            runtime.append(dict(population=r['population'],scene=r['scene'],frame=r['frame'],repeat=r['repeat'],mode=r['mode'],diagnostics=d))
            if not d.get('admitted'):fallbacks[d.get('fallback_reason','unknown')]+=1
            batches.update(str(s['batch']) for s in d.get('steps',[]))
        if r['mode'] in ('isolated_serial','microbatch2','microbatch4'):
            isolation.append(dict(mode=r['mode'],population=r['population'],scene=r['scene'],frame=r['frame'],repeat=r['repeat'],
                requests=d.get('requests',r['m3'].get('requests',[])),private_rng_does_not_advance_global=d.get('private_rng_does_not_advance_global'),
                isolated_comparison='service/quality.jsonl',admitted=d.get('admitted',r['mode']=='isolated_serial')))
    for pop in sorted({r['population'] for r in runs}):
        for mode in sorted({r['mode'] for r in runs if r['population']==pop}):
            rr=[r for r in runs if r['population']==pop and r['mode']==mode]
            qq=[q for q in checks if q['population']==pop and q['mode']==mode]
            perquestion=[];commitlat=[];planning_latency=[]
            for r in rr:
                diag=r['m3'].get('sampling_microbatch',{})
                if diag.get('admitted'):
                    perquestion.extend(q['compute_complete_ms'] for q in diag['requests'])
                    commitlat.extend(q['ordered_transaction_commit_ms'] for q in diag['requests'])
                    planning_latency.append(diag['requests'][0]['compute_complete_ms'])
                elif r['language_timings']:
                    first=r['language_timings'][0]
                    ready=first['complete_from_frame_start_ms']-first['generation_wall_ms']
                    perquestion.extend(q['complete_from_frame_start_ms']-ready for q in r['language_timings'])
                    commitlat.extend([r['language_timings'][-1]['complete_from_frame_start_ms']-ready]*len(r['language_timings']))
                    planning_latency.append(first['generation_wall_ms'])
            token_count=sum(len(a[0])-1 for r in rr for a in r['generated_ids'])
            language_wall=[r['m3'].get('sampling_microbatch',{}).get('full_generation_wall_ms',sum(q['generation_wall_ms'] for q in r['language_timings'])) for r in rr]
            table.append(dict(population=pop,mode=mode,frames=len(rr),e2e_wall_ms=stats([r['wall_ms'] for r in rr]),
                cold_e2e_wall_ms=stats([r['wall_ms'] for r in rr if r['cold']]),
                state_ready_e2e_wall_ms=stats([r['wall_ms'] for r in rr if not r['cold']]),
                language_all_questions_wall_ms=stats(language_wall),
                per_question_ready_to_compute_complete_ms=stats(perquestion),
                per_question_ready_to_compute_complete_mean_ms=float(np.mean(perquestion)) if perquestion else None,
                per_question_ready_to_ordered_publication_ms=stats(commitlat),planning_compute_latency_ms=stats(planning_latency),
                generated_tokens=token_count,aggregate_tokens_per_language_second=token_count*1000/sum(language_wall) if sum(language_wall) else None,
                target_forward_calls=sum(len(r['invocations']) for r in rr),
                peak_gpu_allocated=max(r['peak_gpu_allocated'] for r in rr),
                quality_checks=len(qq),frozen_quality_passes=sum(q['frozen_targets_pass'] for q in qq),
                isolated_serial_quality_passes=sum(q.get('isolated_serial',{}).get('frozen_targets_pass',False) for q in qq),
                perception_memory_exact=sum(q['perception_memory_exact'] for q in qq),
                eos_generations=sum(a[0][-1] in eos for r in rr for a in r['generated_ids']),
                budget_generations=sum(len(a[0])-1==320 and a[0][-1] not in eos for r in rr for a in r['generated_ids']),
                token_lengths=[len(a[0])-1 for r in rr for a in r['generated_ids']],
                state_maintenance_ms=stats([r['m1'].get('m1_total_including_adapters_ms',0.) for r in rr]),
                scientific_control_only=mode=='greedy_control'))
    paired=[]
    lookup={(r['population'],r['scene'],r['frame'],r['repeat'],r['mode']):r for r in runs}
    for r in runs:
        if not r['mode'].startswith('microbatch'):continue
        key=(r['population'],r['scene'],r['frame'],r['repeat'])
        refs={mode:lookup.get(key+(mode,)) for mode in ('original','full_maintain','m3_native','isolated_serial')}
        paired.append(dict(population=r['population'],mode=r['mode'],scene=r['scene'],frame=r['frame'],repeat=r['repeat'],
            ratios={m:ref['wall_ms']/r['wall_ms'] for m,ref in refs.items() if ref},
            quality_pass=r['quality_pass'],admitted=r['m3'].get('sampling_microbatch',{}).get('admitted',False),
            original_tokens=sum(len(a[0])-1 for a in refs['original']['generated_ids']),
            actual_tokens=sum(len(a[0])-1 for a in r['generated_ids'])))
    comparison=dict(rows=table,paired=paired,population='exploratory_reused_population',heldout=False,
        timing='Full request includes original EVA, Detection, Map, Temporal, all questions, M1 full maintenance '
            'and attempt/fallback cost. CPU snapshots, sampling syncs, gather/pack/compaction and output write are charged. '
            'CUDA event intervals are not kernel-only times. Cold means state-reset first frame, excluding model loading.',
        question_latency='Ready-to-compute completion and ordered-publication latency are separate. Native times are from '
            'first language call ready; microbatch times from runtime entry. Both start after heads/commit. No streaming latency claim.',
        greedy='do_sample=False engineering control; excluded from sampling speedup and production certification')
    write_json(root/'end_to_end_comparison.json',comparison)
    runtime_artifact=dict(implemented=True,default_enabled=False,production_enabled=False,rows=runtime,
        actual_batched_requests=sum(r['diagnostics'].get('actual_batched_requests',0) for r in runtime),
        batch_size_distribution=dict(batches),fallback_reasons=dict(fallbacks),
        fixed_groups=[2,4],length_grouping_optimization='not_run_no_qualification_to_expand_runtime',
        no_persistent_prefix_cache_access=True)
    write_json(root/'microbatch_runtime_results.json',runtime_artifact)
    census['current_service_dependency_checks']=[dict(population=r['population'],mode=r['mode'],
        scene=r['scene'],frame=r['frame'],repeat=r['repeat'],
        ready_task_count=r['diagnostics'].get('ready_task_count',0),
        admitted=r['diagnostics'].get('admitted',False),dependencies=r['diagnostics'].get('dependencies',[]))
        for r in runtime]
    write_json(root/'native_question_dependency_census.json',census)
    rng_pairs=[]
    for r in runs:
        d=r['m3'].get('sampling_microbatch',{})
        if r['population']!='native_multi' or not r['mode'].startswith('microbatch') or not d.get('admitted'):continue
        ref=lookup.get((r['population'],r['scene'],r['frame'],r['repeat'],'isolated_serial'))
        if ref is None:continue
        reference_requests=ref['m3']['requests']
        for request in d['requests']:
            index=request['request_id'];original=reference_requests[index]
            ref_length=len(ref['generated_ids'][index][0])-1
            rng_pairs.append(dict(mode=r['mode'],scene=r['scene'],frame=r['frame'],repeat=r['repeat'],request_id=index,
                seed_equal=request['seed']==original['seed'],generated_length=request['generated_tokens'],native_length=ref_length,
                equal_draw_count=request['generated_tokens']==ref_length,
                final_rng_state_equal=request['final_rng_sha256']==original['final_rng_sha256'],
                token_ids_equal=r['generated_ids'][index]==ref['generated_ids'][index]))
    write_json(root/'sampling_rng_isolation.json',dict(rows=isolation,real_request_pairs=rng_pairs,
        matching_draw_count_pairs=sum(r['equal_draw_count'] for r in rng_pairs),
        matching_draw_count_rng_states_equal=sum(r['equal_draw_count'] and r['final_rng_state_equal'] for r in rng_pairs),
        private_generators=True,seed_mapping='SHA256(seed,scene,frame,original_question_index), no output/token/GT dependence',
        tests='test_sampling_microbatch.py, test_sampling_microbatch_cuda.py: same distributions with group-size changes; '
            'different EOS, compaction, exception/OOM fallback and shared global RNG restoration',
        rng_isolation_does_not_certify_target_logits=True,
        original_vs_isolated_serial='Original uses one shared RNG stream; isolated serial is a separately labeled original-model comparator '
            'with the same per-request seed assignment as microbatch. Same-seed token differences alone are not a distribution test.'))
    mbchecks=[q for q in checks if q['mode'].startswith('microbatch') and q['population']=='native_multi']
    quality_ok=bool(mbchecks) and all(q['frozen_targets_pass'] for q in mbchecks)
    write_json(root/'downstream_task_quality.json',dict(checks=checks,
        complete_model_service=service,full_EVA_Detection_Map_Temporal_LLM_Planning=True,
        frozen_microbatch_quality_pass=quality_ok,
        comparison='Frozen quality_pass unchanged; separately report private-RNG serial comparator. '
            'Independent RNG is not permission to ignore text/planning differences.',
        planning=[dict(mode=r['mode'],population=r['population'],scene=r['scene'],frame=r['frame'],seed=r['seed'],metrics=r['planning']) for r in runs],
        gt_usage='Only masked offline ADE/FDE, never online generation/admission.',
        semantic_correctness_limit='Frozen exact-text/task-output regression; no new human annotation or official mAP/safety certification.'))
    numrows=numerical.get('matched_prefix_rows',[])
    logit_equal=all(r['logits']['equal'] for r in numerical['rows']+numrows)
    numerical['all_measured_native_logits_equal']=logit_equal
    numerical['sampling_distribution_qualified']=False
    numerical['qualification_reason']='Measured probability differences' if not numerical['all_targets_equal'] else \
        'No general native FP16 equality certificate; finite target-support checks cannot establish the full sampling law'
    write_json(root/'microbatch_numerical_parity.json',numerical)
    ratios={}
    for mode in ('microbatch2','microbatch4'):
        pp=[r for r in paired if r['mode']==mode and r['population']=='native_multi']
        ratios[mode]={ref:stats([r['ratios'][ref] for r in pp if ref in r['ratios']]) for ref in ('original','full_maintain','m3_native','isolated_serial')}
    overlap=read(root/'m1_decode_overlap_analysis.json') if (root/'m1_decode_overlap_analysis.json').exists() else dict(opportunity='not_run')
    fields=dict(sgvd_numerical_root_cause=root_cause['root_cause'],sgvd_frozen=True,
        native_independent_task_population=dict(requests=census['independent_requests'],frames=len(census['frames']),population=census['population']),
        sampling_microbatch_feasible=feasible['opportunity'],sampling_distribution_qualified=False,
        microbatch_runtime_implemented=True,microbatch_task_quality_pass=quality_ok,
        microbatch_gpu_speedup={str(r['batch']):next(s['gpu_ms']['p50'] for s in feasible['summary'] if s['batch']==1)*r['batch']/r['gpu_ms']['p50'] for r in feasible['summary']},
        microbatch_gpu_speedup_definition='Aggregate sampled-token throughput ratio from CUDA event intervals, including '
            'native processors/private sampling and host gaps; not a kernel-only or complete-service gain.',
        microbatch_full_service_speedup=ratios,
        performance_qualification='Actual experimental timing ratios, including all attempts/fallbacks; not quality-qualified acceleration. '
            'Generated token totals and frozen quality are reported separately; generation budget was never lowered.',
        single_request_optimization_result='Measured original GEMM/LoRA/LM-head diagnostics only; no certified replacement' if (root/'single_request_gemm_analysis.json').exists() else 'not_run',
        m1_decode_overlap_opportunity=overlap['opportunity'],production_microbatch_enabled=False,
        formal_go=False,reason='No qualified native FP16 sampling distribution'+(' and frozen microbatch task quality failed' if not quality_ok else '; no independent heldout certification'),
        remaining_system_bottleneck='Batch=1 FP16 weight traffic/shape efficiency, full prefill and M1 maintenance. '
            'Batched arithmetic/RNG semantics remain a formal sampling/quality qualification barrier.',
        next_research_priority='Use measured per-question planning latency and source-bound native FP16 probabilities to evaluate '
            'any future batching contract; do not expand runtime or enable production under current evidence.')
    write_json(root/'formal_go_nogo.json',fields)
    import json
    lines=['# M3 数值归因与依赖安全 Microbatch 实验报告','',
        '完成真实原模型数值归因、同帧任务依赖人口检查、Sampling Batch CUDA 测量、默认关闭的 RNG 隔离 Microbatch 原型与完整服务回归。',
        '',f"SGVD 最终归因：`{root_cause['root_cause']}`。原生处理协议一致，最早分歧算子为 {', '.join(root_cause['earliest_operations'])}；原生 FP16、温度、Top-k/Top-p 未改动。SGVD 保持冻结。",
        '',f"依赖人口：{census['independent_requests']}/{census['total_requests']} 个真实问题通过单轮 Prompt/共享完整视觉输入的独立性检查。数据为 exploratory_reused_population，没有独立 Heldout 资格。",
        '', 'Batch=1/2/4/8 单步 GPU 实验是不同真实问题的吞吐测量，不代表完整服务收益。实际原型仅运行固定 Batch=2/4；每个请求独立 RNG、KV、Position、Mask、EOS，完成行退出 Batch，整帧结果按原问题顺序事务提交。',
        '',f"Microbatch 目标分布最大 TV={numerical['max_target_total_variation']:.9g}；全部已测 Logits 逐位相同：{logit_equal}。独立 RNG 不消除 FP16 Batch 形状差异，也不保证与原始共享 RNG 在同 Seed 下逐 Token 相同。",
        '', '| 工作负载 | 方法 | 全帧 E2E P50 / P95 ms | 冻结质量通过 / 检查 |', '|---|---|---:|---:|']
    for r in table:lines.append(f"| {r['population']} | {r['mode']} | {r['e2e_wall_ms']['p50']:.2f} / {r['e2e_wall_ms']['p95']:.2f} | {r['frozen_quality_passes']} / {r['quality_checks']} |")
    lines.extend(['','| Native Multi 方法 | 每问题完成 P50 / P95 ms | 规划完成 P50 ms | 有序发布 P95 ms |',
        '|---|---:|---:|---:|'])
    for r in table:
        if r['population']!='native_multi' or not r['per_question_ready_to_compute_complete_ms']:continue
        q=r['per_question_ready_to_compute_complete_ms'];planning=r['planning_compute_latency_ms'];publish=r['per_question_ready_to_ordered_publication_ms']
        lines.append(f"| {r['mode']} | {q['p50']:.2f} / {q['p95']:.2f} | {planning['p50']:.2f} | {publish['p95']:.2f} |")
    lines.extend(['',f"完整帧请求 {len(runs)} 次；CUDA 回归 {regression['tests']} 项，失败 {regression['failures']}、错误 {regression['errors']}、跳过 {regression['skipped']}。",
        '', '主要性能指标是同一帧完成全部真实问题的时间。各方法保留完整视觉输入、问题、生成预算和 M1/Temporal 状态维护；拒绝、单问题回退和所有额外采样/打包成本计入。每问题完成尾延迟、规划请求延迟、输出事务提交延迟、Token 总量与 EOS 分别见 end_to_end_comparison.json。',
        '', 'Original 使用原始共享 RNG；isolated_serial 使用原始 generate 和与 Microbatch 相同的独立请求 Seed，为区分 RNG 与算术差异提供对照。greedy_control 仅为工程对照，不能代表 Sampling 速度或质量；若其内存预算导致串行回退，诊断中完整保留。没有改变原 batched_decode 对 Sampling 的保护。',
        '', '生产结论：No-Go。实测吞吐或全帧服务收益不能替代严格采样和任务质量资格。单问题 Planning 不使用多问题 Batch 收益。备用方向仅做有限真实 Batch=1 GEMM 与 Commit 关键路径诊断；没有大幅改写 Attention、重新捕获 CUDA Graph 或建立另一套异步状态系统。',
        '', 'Detection/Map/Temporal 和完整文本/Planning 使用原冻结标准。Planning ADE/FDE 仅以合法 GT Mask 离线计算；未提供额外人工文本标注或完整官方 nuScenes 安全评估。模型加载不包含在 state-cold 首帧计时内。',
        '', '复现命令见 commands.sh，模型/环境/SHA、原始逐帧结果、数值差异张量和日志均保留。',
        '', '```json',json.dumps(fields,ensure_ascii=False,indent=2),'```',''])
    (root/'experiment_report.md').write_text('\n'.join(lines))
    (root/'implementation_changes.md').write_text('''# Implementation changes

- `question_dependencies.py`: exact native single-turn prompt binding, answer-reference and state/ordering barriers, ready groups in source order.
- `microbatch_kv.py`: owned independent left-padded legacy KV rows, explicit per-row positions, active-row compaction; no prefix sharing.
- `sampling_microbatch.py`: offline fixed 2/4 full-input sampled decode, original native serial prefill, private request RNG, EOS/budget/processor handling, request-local temporary state, atomic ordered return, failure/OOM cleanup and Native fallback.
- `M3Config`: append-only default-off research switch and mutually exclusive offline guards. `M3Runtime`: the new branch is reached only with that switch; original Native Decode and original Sampling rejection in BatchedExecutor remain unchanged.
- New numerical/dependency/KV/RNG/EOS/compaction/exception tests, CUDA test, real-model probe and complete-service tools under `tools/worldstream_m3/microbatch/`.
- Frozen M1/M2, original network/weights, generation budget and task-quality gate are not modified. No production certificate is issued.
''')
    hashes=read(root/'validation/sources.json')
    late=read(root/'fallback_execution_sources.json') if (root/'fallback_execution_sources.json').exists() else {}
    confirmation=read(root/'final_source_confirmation.json') if (root/'final_source_confirmation.json').exists() else {}
    sampler_path='worldstream/m3/sampling_microbatch.py'
    import ast
    class RemoveMaxTimeGuard(ast.NodeTransformer):
        def visit_BoolOp(self,node):
            self.generic_visit(node)
            node.values=[v for v in node.values if not (isinstance(v,ast.Compare) and
                isinstance(v.left,ast.Attribute) and isinstance(v.left.value,ast.Name) and
                v.left.value.id=='config' and v.left.attr=='max_time' and len(v.ops)==1 and
                isinstance(v.ops[0],ast.IsNot) and len(v.comparators)==1 and
                isinstance(v.comparators[0],ast.Constant) and v.comparators[0].value is None)]
            return node
    old_sampler=ast.dump(ast.parse((root/'validation/source'/sampler_path).read_text()))
    normalized_new=ast.dump(RemoveMaxTimeGuard().visit(ast.parse(Path(sampler_path).read_text())))
    sampler_unchanged=(hashlib.sha256((root/'validation/source'/sampler_path).read_bytes()).hexdigest()==
        hashlib.sha256(Path(sampler_path).read_bytes()).hexdigest())
    admission_only_verified=((sampler_unchanged or old_sampler==normalized_new) and confirmation.get('complete_token_ids_equal') is True and
        confirmation.get('private_rng_states_equal') is True and
        confirmation.get('final_source_sha256')==hashlib.sha256(Path(sampler_path).read_bytes()).hexdigest() and
        read(root/'attribution/processing_protocol.json')['offline_config'].get('max_time') is None)
    regression_sources=read(root/'regression_preflight/source_manifest.json')
    unexpected=[];changes=[]
    for path,h in hashes.items():
        current=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if current!=h:
            changes.append(path)
            allowed=(path=='tools/worldstream_m3/microbatch/finalize.py' or late.get(path)==current or
                path==sampler_path and admission_only_verified or
                path.startswith('tests/') and regression_sources.get(path)==current)
            if not allowed:unexpected.append(path)
    late_unexpected=[path for path,h in late.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h]
    unexpected.extend(path for path in late_unexpected if path not in unexpected)
    regression_execution_changes=[path for path,h in regression_sources.items()
        if path.startswith(('worldstream/','tests/')) and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h]
    write_json(root/'source_audit.json',dict(changed_since_validation_snapshot=changes,
        late_loaded_fallback_sources=late,unexpected_execution_changes=unexpected,
        admission_only_guard_extension_verified=admission_only_verified,
        late_loaded_sources_still_identical=not late_unexpected,
        regression_test_and_runtime_sources_unchanged=not regression_execution_changes,
        explanation='Fallback modules are imported after complete service; exact loaded versions archived before fallback execution. '
            'Finalizer edits are reporting-only. The max_time admission rejection is AST-isolated and confirmed by '
            'complete real CUDA generation against measured source; tests are pinned again before pytest.'))
    if unexpected:raise ValueError('Unexpected execution source changes: '+str(unexpected))
    write_json(root/'artifact_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json') if p.name!='artifact_manifest.json'})
    print(json.dumps(fields,indent=2))


if __name__=='__main__':main()
