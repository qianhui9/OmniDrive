"""Measured coarse scope, dependency rejections and deliberately generous bounds."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from tools.worldstream_m123.stateful_opportunity.common import manifest,protected_files,sha256,stats,write
from tools.worldstream_m123.stateful_opportunity.report import opportunity_gate,build as refine_old


def read(root,name):return json.loads((root/name).read_text())


def removal_bound(total,removable,mandatory):
    """Zero-cost scope oracle; preserve the actual input pipeline lower bound.

    Removable is an intentionally loose CPU+GPU envelope, not a measured
    speedup. Mandatory input preprocessing/scatter cannot disappear in this task.
    """
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (total,removable,mandatory)):
        raise ValueError('Unmeasured bounds cannot be replaced by zero')
    if not 0<mandatory<=total or removable<0:raise ValueError('Invalid service envelope')
    saving=min(removable,total-mandatory)
    return dict(removable_upper_ms=saving,speedup_upper=total/(total-saving))


def scope_sum(frame,names,field):
    scopes=frame['scopes']['inclusive_module_intervals']
    return sum(scopes[n][field] for n in names)


def build(run,output,regression=None):
    status=read(run,'status.json')
    for name in ('protected_source_unchanged','historical_outputs_unchanged','parameters_unchanged','audit_source_unchanged','hooks_restored','gpu_accounting_valid'):
        if status.get('status')!='completed' or status.get(name) is not True:
            raise ValueError('Incomplete or invalid run: '+name)
    protocol=read(run,'protocol.json');frames=read(run,'service_frames.json');profiles=read(run,'profiles.json')
    witnesses=read(run,'dependency_witnesses.json');states=read(run,'m1_states.json')
    expected={(g['scene'],token,mode) for g in protocol['groups'] for token in g['tokens']
              for mode in ('Native Original','Persistent Full')}
    if {(f['scene'],f['frame'],f['mode']) for f in frames}!=expected or len(frames)!=len(expected):
        raise ValueError('Missing or duplicate service requests')
    output.mkdir(parents=True,exist_ok=True)
    refine_old(Path('outputs/stateful_cross_frame_opportunity/gpu_audit02'),output/'p0_refinement',
        Path('outputs/stateful_cross_frame_opportunity/gpu_audit02/regression.json'))
    native=[f for f in frames if f['mode']=='Native Original' and not f['cold']]
    persistent=[f for f in frames if f['mode']=='Persistent Full' and not f['cold']]
    nindex={(f['scene'],f['frame']):f for f in native}
    m1costs=[f['m1']['m1_total_including_adapters_ms'] for f in persistent]
    population=dict(original_task='planning',task_classifier_label='global_safety',population='Exploratory',
        native_multi_cross_frame_requalification='Not Run',scenes=len(protocol['groups']),
        unique_frames=sum(len(g['tokens']) for g in protocol['groups']),service_requests=len(frames),
        measured_requests_per_mode=len(native),profile_requests=len(profiles),witness_requests=len(witnesses),
        independent_heldout_scenes=0)
    service={mode:dict(request_wall_ms=stats(f['request_wall_ms'] for f in rows),
        model_wall_ms=stats(f['model_wall_ms'] for f in rows),
        input_pipeline_ms=stats(f['input_ms'] for f in rows),
        peak_allocated_bytes=max(f['peak_allocated_bytes'] for f in rows))
        for mode,rows in [('Native Original',native),('Persistent Full',persistent)]}
    stage_names=set().union(*(p['summary']['stages'].keys() for p in profiles))
    hotspots={name:dict(kernel_ms=stats(p['summary']['stages'].get(name,{}).get('kernel_ms',0) for p in profiles),
        compute_fraction=stats(p['summary']['stages'].get(name,{}).get('kernel_ms',0)/p['summary']['device_compute_kernel_ms'] for p in profiles))
        for name in stage_names}
    paired=[dict(scene=f['scene'],frame=f['frame'],native_ms=nindex[f['scene'],f['frame']]['request_wall_ms'],
                 persistent_ms=f['request_wall_ms'],increment_ms=f['request_wall_ms']-nindex[f['scene'],f['frame']]['request_wall_ms'])
            for f in persistent]
    paired_index={(p['scene'],p['frame']):p['increment_ms'] for p in paired}
    incremental_budget=[max(0,p['increment_ms']) for p in paired]
    cost_scope=('Descriptive single-pass, same-frame Persistent Full minus Native Original service increment. '
        'Keep signed measurements; use max(0,increment) only for optimistic required-savings budgets. '
        'Internal M1 wall can include GPU waits and is not additive extra work. '
        'Persistent Full versus Active would charge new validation/cache costs only, currently Not Run. '
        'Rotated scene order reduces but does not remove timing noise; no causal latency certificate.')
    guards=read(run,'service_output_guards.json')
    output_guards=[dict(key=k,bitwise_equal=v['Native Original']==v['Persistent Full']) for k,v in guards.items()]
    if len(output_guards)!=population['unique_frames'] or not all(g['bitwise_equal'] for g in output_guards):
        raise ValueError('Native/Persistent population or output guard mismatch')
    if not all(w['service_outputs_bitwise_equal'] for w in witnesses):
        raise ValueError('Profiler/witness replay changed native outputs')
    write(output/'full_model_gpu_profile.json',dict(population=population,service=service,
        paired_native_persistent=paired,stage_hotspots=hotspots,profiles=profiles,
        service_scope='Coarse CUDA events retained in both arms; full input pipeline included. No active strategy.',
        profiler_scope='Two separately profiled natural requests; trace/kernel time is not latency calibration. Unmatched runtime correlations retained as unattributed, never reassigned or scaled.',
        profiler_overhead=[dict(scene=p['scene'],frame=p['frame'],profiled_model_wall_ms=p['profiled_model_wall_ms'],
            unprofiled_same_frame_model_wall_ms=next(f['model_wall_ms'] for f in frames if f['scene']==p['scene'] and f['frame']==p['frame'] and f['mode']=='Persistent Full'),
            teardown_ms=p['profiler_teardown_wall_ms']) for p in profiles],
        native_persistent_output_guards=output_guards,
        active_task_quality='Not Run',active_e2e='Not Run',production_go=False))
    # Three coarse boundaries only. Sort by actual measured kernel cost.
    definitions=[
        dict(name='whole_visual_backbone',scopes=['eva'],stages=['eva'],
            source=['projects/mmdet3d_plugin/models/backbones/eva_vit.py:EVAViT.forward',
                    'worldstream/m2/dependency_planner.py:DependencyPlanner.plan'],
            inputs=['all six current normalized RGB tensors','camera order','model/LoRA/backend signature','absolute positions and RoPE','global attention context'],
            dependencies=['rgb'],stop_reason='current_rgb_changed_global_dependency',
            why='Any changed camera input reaches its global blocks. M1 stable regions do not prove whole EVA tensor equivalence. No displaced-window method reconsidered.'),
        dict(name='detection_map_transformer_subgraphs',scopes=['detection','map'],stages=['detection','map'],
            source=['projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py:StreamPETRHead.forward',
                    'projects/mmdet3d_plugin/models/dense_heads/petr_head_map.py:PETRHeadM.forward',
                    'projects/mmdet3d_plugin/models/utils/petr_transformer.py:PETRTransformerDecoderLayer.forward'],
            inputs=['current projected visual features','current positional embeddings','queries','temporal memory and position','masks','pose/timestamps','model/execution version'],
            dependencies=['detection/current_visual','detection/temporal_pos','detection/query',
                          'map/current_visual','map/temporal_pos'],stop_reason='current_visual_queries_and_temporal_keys_changed',
            why='Attention/FFN outputs depend on current queries and current image keys. Native temporal Memory is already in the baseline. Historical V alone does not close a Transformer subgraph.'),
        dict(name='state_triggered_large_llm_prefill_decode',scopes=['llm_generation'],
            stages=['llm_prefill','llm_decode','llm_sampling_and_preparation'],
            source=['projects/mmdet3d_plugin/models/detectors/petr3d.py:Petr3D.simple_test_pts',
                    'worldstream/m3/world_state.py:query_world_state',
                    'worldstream/m3/prefill_executor.py:PrefillExecutor.generate'],
            inputs=['complete current vision carriers','native token/position/mask history','all layer KV','generated token history/EOS','RNG and sampling processors','model/LoRA/execution version'],
            dependencies=['llm/inputs_embeds','llm/layer0_key','llm/layer0_value'],
            stop_reason='planning_state_insufficient_and_multimodal_kv_changed',
            why='Planning requires full current evidence. Current multimodal inputs and KV differ; RNG and generated history add decode dependencies. No historical answer substitution or carrier deletion is allowed.')]
    for d in definitions:
        d['profile_kernel_ms']=[sum(p['summary']['stages'].get(s,{}).get('kernel_ms',0) for s in d['stages']) for p in profiles]
    definitions.sort(key=lambda d:-sum(d['profile_kernel_ms']))
    candidates=[];bounds=[];decisions=[]
    for rank,d in enumerate(definitions,1):
        inspected=[w for w in witnesses if w['history_legal']]
        checks={name:dict(observed=sum(name in w['dependencies'] for w in inspected),
            exactly_equal=sum(w['dependencies'].get(name,{}).get('exact_equal') is True for w in inspected)) for name in d['dependencies']}
        if any(c['observed']!=len(inspected) for c in checks.values()):raise ValueError('Missing necessary dependency witnesses')
        necessary_unchanged=bool(inspected) and all(c['exactly_equal']==len(inspected) for c in checks.values())
        # These witnesses can disprove a coarse closure. Passing a subset of
        # necessary equalities would require a separate complete proof, never
        # automatic promotion (especially masks/all-layer KV/RNG).
        closure=False
        candidate=dict(name=d['name'],hotspot_rank=rank,operators=d['scopes'],source=d['source'],
            current_required_inputs=d['inputs'],source_origin='actual previous native frame, same scene/cameras/epoch/execution signature; detailed versions in raw M1 states',
            necessary_dependency_witnesses=checks,mathematical_closure=closure,
            necessary_inputs_all_unchanged=necessary_unchanged,
            closure_proof_status='Not Established' if necessary_unchanged else 'Rejected by changed necessary inputs',
            state_evidence_qualified=False,M1_necessary_sufficient=False,
            m1_certified_anchors=sum(s['physically_certified_anchors'] for s in states),
            m1_complete_task_coverage=False,source_legal_transitions=len(inspected),
            all_native_model_computation_preserved=True,actually_skipped=0,
            online_oracle_used=False,blocker=d['why'])
        candidates.append(candidate)
        per_request=[]
        for f in persistent:
            gpu=scope_sum(f,d['scopes'],'cuda_event_interval_ms')
            cpu=scope_sum(f,d['scopes'],'cpu_enqueue_wall_ms')
            oracle=removal_bound(f['request_wall_ms'],gpu+cpu,f['input_ms'])
            gpu_only=removal_bound(f['request_wall_ms'],gpu,f['input_ms'])
            per_request.append(dict(scene=f['scene'],frame=f['frame'],request_ms=f['request_wall_ms'],
                gpu_interval_upper_ms=gpu,cpu_gpu_envelope_upper_ms=gpu+cpu,
                m1_internal_wall_ms=f['m1']['m1_total_including_adapters_ms'],
                paired_service_increment_ms=paired_index[f['scene'],f['frame']],
                minimum_required_savings_ms=max(0,paired_index[f['scene'],f['frame']]),
                gpu_only_host_fixed_speedup_estimate=gpu_only['speedup_upper'],**oracle))
        bound=dict(name=d['name'],candidate_max_compute_fraction=stats(
            value/p['summary']['device_compute_kernel_ms'] for value,p in zip(d['profile_kernel_ms'],profiles)),
            optimistic_gpu_savings_upper_ms=max(r['gpu_interval_upper_ms'] for r in per_request),
            gpu_upper_kind='All candidate CUDA event envelopes free, including launch gaps; not kernel sum or achieved savings',
            actual_profile_kernel_ms=stats(d['profile_kernel_ms']),
            validation_and_cache_cost_ms=None,validation_cost_status='Not Run: dependency pre-screen failed; no new cache/executor built',
            state_maintenance_cost_ms=stats(p['increment_ms'] for p in paired),
            m1_internal_wall_ms=stats(m1costs),minimum_required_savings_ms=stats(incremental_budget),
            required_savings_scope=cost_scope,
            optimistic_e2e_speedup_upper_bound=max(r['speedup_upper'] for r in per_request),
            gpu_only_host_fixed_speedup_estimate=stats(r['gpu_only_host_fixed_speedup_estimate'] for r in per_request),
            gpu_only_estimate_scope='Conditional serialized GPU-event removal with host work fixed; includes device launch gaps. Descriptive estimate, not an unconditional bound or achieved speedup.',
            oracle_speedup_distribution=stats(r['speedup_upper'] for r in per_request),
            oracle_kind='Loose zero-cost whole-scope CPU+GPU envelope, double-counted overlap deliberately favours candidate; preserve actual input pipeline. Per-request quantities only, no sum of unrelated P50s.',
            per_request=per_request,qualified_gpu_savings_ms=None,qualified_savings_status='Not Run',qualified_e2e_speedup=None,
            actual_cross_frame_reuse=0,stop_reason=d['stop_reason'])
        bounds.append(bound)
        gate=opportunity_gate(dict(historical_sources_legal=bool(inspected),mathematical_closure=closure,
            state_evidence_qualified=False,avoids_native_repeated_work=False,
            optimistic_gpu_savings_upper_ms=bound['optimistic_gpu_savings_upper_ms'],
            minimum_required_savings_ms=min(incremental_budget),net_cost_probe_positive=False,
            whole_service_upper_bound_worthwhile=bound['optimistic_e2e_speedup_upper_bound']>1,
            validation_plan_executable=True))
        decisions.append(dict(name=d['name'],**gate,decision='No-Go',stop_reason=d['stop_reason']))
    write(output/'large_boundary_candidates.json',dict(population=population,selection='Descending actual profiler compute-kernel time; three whole-subgraph boundaries only',candidates=candidates))
    write(output/'dependency_closure_analysis.json',dict(candidates=candidates,witnesses=witnesses,
        diagnostic_rule='Equality is necessary, never semantic similarity. Post-forward KV differences only disprove reuse; they cannot authorize avoiding that forward.',
        M1_observation_physical_transformer_task_claims_separate=True))
    # Prior failed mechanisms get bounds only; do not execute them again.
    old=Path('outputs/stateful_cross_frame_opportunity/gpu_audit02')
    vprobes=read(old,'temporal_cost_probes.json');prefix=read(output/'p0_refinement','dependency_analysis.json')['measured_cost']['prefix_by_scene']
    byscene=defaultdict(list)
    for v in vprobes:byscene[v['scene']].append(v)
    v_upper=max(sum(max(p['full']['gpu_samples_ms']) for p in rows) for rows in byscene.values())
    prefix_upper=max(scope_sum(f,['llm_prefill'],'cuda_event_interval_ms') for f in persistent)
    legacy=[]
    for name,upper,validation in [('temporal_v_only',v_upper,None),
        ('35_token_prefix',prefix_upper,stats(p['measured_validation_publication_wall_ms'] for p in prefix))]:
        legacy.append(dict(name=name,candidate_max_compute_fraction=upper/min(p['summary']['device_compute_kernel_ms'] for p in profiles),
            compute_fraction_scope='Cross-experiment isolated GPU-event upper divided by minimum measured full profiler kernel time; generous descriptive scope ratio, not a paired kernel saving',
            optimistic_gpu_savings_upper_ms=upper,validation_and_cache_cost_ms=validation,
            state_maintenance_cost_ms=stats(p['increment_ms'] for p in paired),
            m1_internal_wall_ms=stats(m1costs),minimum_required_savings_ms=stats(incremental_budget),
            required_savings_scope=cost_scope,
            optimistic_e2e_speedup_upper_bound=min(f['request_wall_ms'] for f in native)/(min(f['request_wall_ms'] for f in native)-upper),
            upper_scope='Whole original eligible V GEMMs eliminated, or ENTIRE native prefill eliminated (deliberately overgenerous for 35 tokens); no proportional token extrapolation',
            stop_reason=['negative_net_cost']+(['insufficient_compute_scope'] if upper<stats(incremental_budget)['p50'] else []),
            insufficient_scope_comparison='Generous GPU upper versus median paired nonnegative service increment; descriptive, not a uniform per-frame lower bound',
            new_execution='Not Run / prohibited by task',
            source=str(old),cross_experiment_limit='Descriptive generous stopping bound. Old isolated maxima and new minimum service time are not paired latency or qualification; never sum experiment P50s.',
            current_measured_net_cost='Previously measured negative; preserved in p0_refinement/dependency_analysis.json',
            validation_cost_scope='V-only pure validation/cache not isolated in old run; candidate plan includes fresh GEMM/gather/concat. Prefix includes measured signature/match/publication. Missing isolated cost stays null.'))
    write(output/'optimistic_savings_bounds.json',dict(candidates=bounds,legacy_no_go=legacy,
        population=population,actual_skipped_forwards=0,achieved_cross_frame_gpu_savings_ms=None,
        achieved_cross_frame_gpu_savings_status='Not Run: no Stateful Active implementation',achieved_active_e2e='Not Run',
        no_bound_promoted_to_quality_or_production=True))
    write(output/'validation_cost_results.json',dict(M1_internal_wall_including_waits_ms=stats(m1costs),
        state_maintenance_cost_ms=stats(p['increment_ms'] for p in paired),
        minimum_required_savings_ms=stats(incremental_budget),cost_scope=cost_scope,
        native_persistent_paired_increments_ms=stats(p['increment_ms'] for p in paired),
        m1_precheck_internal_timing=[dict(scene=f['scene'],frame=f['frame'],timings_ms=f['m1_precheck']) for f in persistent],
        witness_source_equality_wall_ms=stats(w['whole_input_verification_wall_ms'] for w in witnesses if w['history_legal']),
        diagnostic_snapshot_copy_wall_ms=stats(w['diagnostic_snapshot_copy_wall_ms'] for w in witnesses),
        snapshot_scope='Read-only diagnostic snapshots, not an implemented reusable cache or online admission cost. Outside service/profile timers.',
        m1_cpu_gpu_transfers=[dict(scene=f['scene'],frame=f['frame'],
            output_transfer_ms=f['m1'].get('output_gpu_to_cpu_transfer_ms'),
            output_transfer_bytes=f['m1'].get('output_gpu_to_cpu_transfer_bytes')) for f in persistent],
        new_candidate_verification_cache='Not Run: no coarse candidate passes dependency pre-screen',
        existing_m2_m3=dict(active=False,actual_reuse=0,cache_maintenance_ms=0,
            reason='Native Original and Persistent Full keep M2/M3 disabled; no existing temporal memory savings credited twice'),
        legacy_prefix_validation_cache=legacy[1]['validation_and_cache_cost_ms']))
    numerical=read(Path('outputs/stateful_cross_frame_opportunity'),'correctness_results.json')
    write(output/'numerical_diagnostics.json',dict(legacy_value_projection=numerical['temporal_projection_diagnostics'],
        legacy_prefix_prefill=numerical['prefix_prefill_diagnostics'],
        source=str(Path('outputs/stateful_cross_frame_opportunity/correctness_results.json')),
        sampling_distribution='Not Certified',stateful_active_task_quality='Not Run',stateful_active_e2e='Not Run',
        new_candidate_same_input_alternate_shape_probe=dict(status='Not Run',reason='No complete identical coarse input was found; no candidate passed static pre-screen'),
        native_persistent_output_guards=output_guards,
        profiled_native_output_guards=[w['service_outputs_bitwise_equal'] for w in witnesses],
        numerical_failure_not_inferred_for_unexecuted_active=True))
    write(output/'candidate_go_nogo.json',dict(candidates=decisions,production_go=False))
    formal=dict(mathematical_closure=False,state_evidence_qualified=False,performance_opportunity_go=False,
        state_evidence_go=False,dependency_closure_go=False,actual_cross_frame_reuse_go=False,
        numerical_correctness_go=False,frozen_task_quality_go=False,gpu_net_savings_go=False,e2e_speedup_go=False,
        production_go=False,active_numerical_status='Not Run',active_quality_status='Not Run',active_e2e_status='Not Run',
        P0='Refined; historical results preserved',P1='Coarse boundary audit completed / No-Go',
        P2='Not Implemented / No-Go',stop_reason='No whole-subgraph dependency closure or sufficient state evidence. Large oracle time alone does not authorize reuse.',
        four_mode_protocol=dict(status='Retained / Not Run',modes=['Native Original','Persistent Full','State Shadow','Stateful Active'],
            primary='Persistent Full vs Stateful Active',ablation='Disable historical reuse, retain identical other engineering'),
        population=population)
    write(output/'formal_go_nogo.json',formal)
    regression_note='完整回归记录尚未导入；不以此替代任何任务质量认证。'
    if regression:
        regression_result=read(regression.parent,regression.name)
        write(output/'regression_results.json',regression_result)
        regression_note=(f"完整回归 {regression_result['tests']} 项，Failure={regression_result['failures']}，"
            f"Error={regression_result['errors']}，Skip={regression_result['skipped']}；旧 658 项历史记录保留。"
            '该回归不替代 Stateful Active 原始任务质量认证。')
    executed_source=read(run,'source_manifest.json')
    runtime_paths=[p for p in executed_source if Path(p).name not in ('report.py','__init__.py')]
    write(output/'delivery_integrity.json',dict(protected_source_unchanged=manifest(protected_files())==read(run,'protected_before.json'),
        historical_outputs_unchanged=all(sha256(p)==digest for p,digest in read(run,'historical_output_manifest.json').items()),
        executed_runtime_unchanged=all(sha256(p)==executed_source[p] for p in runtime_paths),
        executed_source=executed_source,current_analysis_source=manifest(Path(__file__).parent.glob('*.py')),
        analysis_revision_note='GPU completed with its archived source manifest. Subsequent report.py-only changes clarify paired service costs and conditional oracle accounting; no runtime/profiler code changed or results replayed.',
        raw_run=str(run),old_regression_658_preserved=True))
    costs='\n'.join(f"| {d['name']} | {sum(d['profile_kernel_ms'])/len(profiles):.2f} | {next(b for b in bounds if b['name']==d['name'])['candidate_max_compute_fraction']['p50']*100:.2f}% | {d['stop_reason']} |" for d in definitions)
    bound_rows='\n'.join(f"| {b['name']} | {b['optimistic_gpu_savings_upper_ms']:.3f} | {b['gpu_only_host_fixed_speedup_estimate']['p50']:.3f}× | {b['optimistic_e2e_speedup_upper_bound']:.3f}× |" for b in bounds)
    report=f'''# 大粒度跨帧计算边界审计

结论：**No-Go，未新增 Runtime 或启用 Production/Auto。** 保留 M1→M2→M3 研究约束、原始权重/LoRA/模型/Planning 及完整 Sampling 配置。旧 V-only、35 Token Prefix、EVA Window 不再优化或执行。

本轮服务人口为 {population['scenes']} 个已曝光场景、{population['unique_frames']} 个自然连续帧、两种原生服务共 {len(frames)} 个请求。`original_task=planning`，`task_classifier_label=global_safety`，`population=Exploratory`，`native_multi_cross_frame_requalification=Not Run`。详细 Profiler 为另一次回放的 {len(profiles)} 个请求，来源/依赖 Witness 为 {len(witnesses)} 帧，不与主服务人口相加。

| 候选 | 独立 Profiler 平均 Kernel ms | GPU Compute 占比 P50 | 停止理由 |
|---|---:|---:|---|
{costs}

Kernel 由原始 Trace 的真实 device activity 逐条计数，与独立 Profiler device event 总时长/数量核对；能建立唯一 CPU runtime correlation 的活动才分配阶段。归因时间覆盖率分别为 {[round(p['summary']['attributed_time_fraction']*100,5) for p in profiles]}%，其余明确保留为 unattributed，不补齐、不按比例缩放。Torch 1.13 的 CPU FunctionEvent.kernel 关联存在重复/漏项，故未采用该错误总量。三级嵌套 Attention/FFN/Projection 仅作 inclusive 明细，不重复相加。CUDA Event Interval 含主机发射间隙，不等于 Kernel Sum。两次详细插桩不是延迟校准，开销及原始 Trace 单列。

Native Original 完整请求 E2E P50/P95 为 {service['Native Original']['request_wall_ms']['p50']:.2f}/{service['Native Original']['request_wall_ms']['p95']:.2f} ms；Persistent Full 为 {service['Persistent Full']['request_wall_ms']['p50']:.2f}/{service['Persistent Full']['request_wall_ms']['p95']:.2f} ms，均包含实际输入加载/预处理/Scatter、完整 VLM、原生回答写盘及同步。冷启动另存，以上不含每场景 ordinal 0。M1 内部 Wall P50/P95 为 {stats(m1costs)['p50']:.2f}/{stats(m1costs)['p95']:.2f} ms，可能包含既有 GPU 工作的等待，不能全部再次收费。88 个同帧服务配对增量 P50/P95 为 {stats(p['increment_ms'] for p in paired)['p50']:.2f}/{stats(p['increment_ms'] for p in paired)['p95']:.2f} ms，保留原始有符号值；这仍是单次、轮换顺序的描述性结果，尚非重复随机化因果结论。不同 P50 不能相减或相加当作配对增益。

本轮按实际热点筛选三种大边界。EVA 接收变化的当前 RGB，Global Attention 阻断局部历史扩张；Detection/Map 的当前视觉、查询和时序位置/Key 变化；LLM Prefill/Decode 消费变化的当前 Carrier、KV、生成历史与 RNG。M1 局部观测/对象历史既不能证明这些整个张量相同，也不能完整覆盖 Planning。原模型已有 Temporal Memory 不计作新节省。新人口未发现可合法跳过的大计算子图。

`optimistic_savings_bounds.json` 按每个实际请求计算整个候选免费、验证为零的宽松 Oracle 上界，保留输入流水线不可省成本。CPU+GPU envelope 刻意对候选有利，不能读作实测加速；同时单列真实 Kernel、GPU Event、CPU Wall 与 M1 维护成本。大 LLM Oracle 可以很大，但依赖/状态不合格；Oracle 不能授权跳算。旧局部候选另用旧隔离测量与新服务作描述性宽松上界，明确跨实验限制，不伪装成配对 E2E，也不按 Token 比例外推 GPU 时间。

| 候选 | GPU Event 整段最大 ms | 固定 Host、只消去 GPU 的条件估计 P50 | CPU+GPU 宽松上界最大值 |
|---|---:|---:|---:|
{bound_rows}

最后一列会重复计算 CPU/GPU 重叠，LLM 的宽松上界尤其松，仅用于不漏掉潜在大机会；它不是可信加速预测。条件估计也假设 GPU 区间可串行消除，包含发射间隙，不能当作实测。均逐请求计算，没有把不相关分位数相加。Native Original 目标还须抵消上述配对状态服务增量；Persistent Full→Active 只需另付新验证/缓存成本，其纯成本为 Not Run/null，因为三个候选均在依赖初筛失败。只读全输入核对与快照成本另存，不能冒充真正可复用 Cache 的成本。

旧 V-only 即使整个合格 V GEMM 免费，描述性 GPU 上界仅 {legacy[0]['optimistic_gpu_savings_upper_ms']:.6f} ms，服务倍率上界 {legacy[0]['optimistic_e2e_speedup_upper_bound']:.6f}×。35 Token Prefix 刻意放宽到整个 Prefill 免费，仍仅 {legacy[1]['optimistic_gpu_savings_upper_ms']:.3f} ms、{legacy[1]['optimistic_e2e_speedup_upper_bound']:.4f}×；旧 Prefix 验证/发布 P50 为 {legacy[1]['validation_and_cache_cost_ms']['p50']:.3f} ms。旧实测净成本负收益与 `insufficient_compute_scope` 停止结论保留，绝不重建缓存。

上一轮数值结论保持：Value 新 GEMM 行 MaxAbs={numerical['temporal_projection_diagnostics']['max_abs']:.8g}；Prefix 最后位置 logits 8 个样本中 3 个逐位相同，MaxAbs=0.046875。原生 Sampling 分布为 Not Certified。Stateful Active 数值、完整任务质量、四模式完整 E2E 均为 **Not Run**，不能写成执行后任务失败。

Native/Persistent 的输出、Token 与 Memory 守卫 {len(output_guards)}/{len(output_guards)} 帧逐位一致，额外 {len(witnesses)} 帧 Profiler/Witness 回放也全部一致，但不替代 Stateful Active 冻结质量资格。{len(inspected)} 个合法历史转换上，各候选列出的必要变化输入均未出现整张量精确相同；这是完整输入相同这一闭包条件的反例，不是对未观测帧或其他数学不变量的穷尽证明。M1 见证状态的物理认证 Anchor 总数为 {sum(s['physically_certified_anchors'] for s in states)}，且并未捕获可证明整个子图等价的完整输入。实际减少 GPU Forward=0，实际跨帧复用=0，质量合格的服务净加速尚未建立。前置依赖失败，因此没有建立新 Cache、专用 Kernel 或多个执行器，也没有伪造四模式结果。

建议停止沿当前边界继续堆叠缓存。仅当出现自然帧中非零、来源合法、必要输入完整闭包、节省可覆盖状态维护、且可按原始 Sampling/任务质量验证的大子图，才重新进入最小实现。当前结果限制在该模型、实际 Planning 人口与所审计三类边界，不证明所有未来状态化模型都不可能。

{regression_note}

实际 HEAD/环境、完整模型与数据 SHA、全部生效生成默认项及覆盖项、源码前后哈希、原始服务帧/Kernel Trace 和失败日志保存在 {run}。旧实验保持原样；复现见 commands.sh，变更见 implementation_changes.md。
'''
    (output/'experiment_report.md').write_text(report)
    (output/'audit_refinement.md').write_text('''# 旧 P0 修正

旧 8 场景/96 帧/88 转换/96 请求是原生 Planning，而分类器返回 global_safety；不是 native_multi VQA。全部已曝光，保持 Exploratory；不与旧 VQA 相加。

Value 与 Prefix 的局部数值差异是已测事实，Sampling 分布是 Not Certified；Stateful Active 完整质量与 E2E 是 Not Run，不改写为实际质量失败。旧 658 项通过记录与全部失败/成功目录保持原样。

report.py 的 P0 不再由任一局部正收益触发。新 opportunity_gate 独立检查真实历史来源、数学闭包、对应 Claim 的状态证据、原生重复工作、覆盖维护成本的上界、正净成本潜力及可执行验收计划。production_go 始终需要后续正式认证，本轮未给予。缺项、未知或非有限测量均 Fail-Closed。

新增性能字段见 optimistic_savings_bounds.json（含旧局部候选）；纯验证/缓存没有隔离测量时保留 null 和具体范围，绝不用零冒充。旧报告重建到新 p0_refinement/，工具拒绝覆盖原始历史研究目录。''')
    environment=read(run,'environment.json')
    provenance=dict(raw_run=str(run),head=environment['commit'],gpu=environment['gpu'],
        checkpoint_sha256=environment['checkpoint_sha256'],dataset=protocol['dataset'],
        seed=environment['seed'],sampling=protocol['sampling'],population=population,
        environment_ref=str(run/'environment.json'),effective_generation_ref=str(run/'effective_generation.json'),
        executed_source_manifest_ref=str(run/'source_manifest.json'))
    artifacts=['full_model_gpu_profile','large_boundary_candidates','dependency_closure_analysis',
        'optimistic_savings_bounds','validation_cost_results','numerical_diagnostics',
        'candidate_go_nogo','formal_go_nogo','delivery_integrity']
    if regression:artifacts.append('regression_results')
    for name in artifacts:
        value=read(output,name+'.json');value['provenance']=provenance;write(output/(name+'.json'),value)
    return formal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--regression',type=Path)
    args=p.parse_args();print(json.dumps(build(args.run,args.output,args.regression),indent=2))


if __name__=='__main__':main()
