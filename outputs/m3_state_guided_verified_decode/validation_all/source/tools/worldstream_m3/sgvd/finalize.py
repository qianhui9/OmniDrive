"""Build claim-limited artifacts from immutable request logs and CUDA checks."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from tools.worldstream_m1.io_utils import write_json
from .summarize_probe import summarize,read_rows


def stats(values):
    return dict(n=len(values),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    root=Path(parser.parse_args().output);gate=summarize(root)
    exact=json.loads((root/'exact_target_gate.json').read_text())
    numerical=json.loads((root/'kv_state_correctness.json').read_text())
    if numerical['parallel_native_target_bitwise_equal']:
        raise ValueError('No measured target-distribution failure: this No-Go finalizer must not replace prototype work')
    sampling=json.loads((root/'sampling_distribution_tests.json').read_text())
    service_status=json.loads((root/'service/status.json').read_text())
    phase1_status=json.loads((root/'phase1/status.json').read_text())
    requests=read_rows(root/'phase1/requests.jsonl');runs=read_rows(root/'service/runs.jsonl');checks=read_rows(root/'service/quality.jsonl')
    suites=ET.parse(root/'regression_cuda.xml').getroot()
    cases=list(suites.iter('testcase'))
    regression=dict(tests=len(cases),failures=sum(c.find('failure') is not None for c in cases),
        errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases),
        cuda_enabled=True,source='regression_cuda.xml',command='commands.sh')
    regression['passed']=regression['tests']>0 and regression['failures']==regression['errors']==regression['skipped']==0
    write_json(root/'regression_results.json',regression)
    table=[]
    for population in sorted({r['population'] for r in runs}):
        for mode in ('original','full_maintain','frozen_exact_m2','m3_native'):
            group=[r for r in runs if r['population']==population and r['mode']==mode]
            table.append(dict(population=population,mode=mode,requests=len(group),
                e2e_wall_ms=stats([r['wall_ms'] for r in group]),
                decode_gpu_interval_ms=stats([sum(c['gpu_interval_ms'] for c in r['invocations'] if c['stage']=='decode') for r in group]),
                decode_span_wall_ms=stats([sum(c['decode_span_wall_ms'] for c in r['generation_profiles']) for r in group]),
                cold_e2e_ms=stats([r['wall_ms'] for r in group if r['cold']]),
                history_ready_e2e_ms=stats([r['wall_ms'] for r in group if not r['cold']]),
                generated_tokens=sum(len(a[0])-1 for r in group for a in r['generated_ids']),
                target_model_forward_calls=sum(len(r['invocations']) for r in group),
                peak_gpu_allocated=max(r['peak_gpu_allocated'] for r in group),
                frozen_quality_checks=sum(q['mode']==mode and q['population']==population for q in checks),
                frozen_quality_passes=sum(q['mode']==mode and q['population']==population and q['frozen_targets_pass'] for q in checks)))
    ratios={}
    for population in sorted({r['population'] for r in runs}):
        lookup={r['mode']:r for r in table if r['population']==population}
        ratios[population]=dict(m3_native_vs_full_maintain=lookup['full_maintain']['e2e_wall_ms']['p50']/lookup['m3_native']['e2e_wall_ms']['p50'],
                               m3_native_vs_original=lookup['original']['e2e_wall_ms']['p50']/lookup['m3_native']['e2e_wall_ms']['p50'])
    old=Path('outputs/m3_decode_optimization/end_to_end_comparison.json')
    comparison=dict(rows=table,ratios=ratios,
        archived_cuda_graph_negative_baseline=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
            reused_measurement=True,recaptured=False),
        sgvd_shadow=dict(source='phase1/probes.jsonl',actual_speedup_credited=False),
        experimental_sgvd=dict(executed=False,reason=exact['decision']),
        optimized_sgvd=dict(executed=False,reason='Stop gate prevents unqualified runtime optimization'),
        population='exploratory_reused_population',independent_heldout=False,
        timing='request-boundary synchronization; decode intervals are deferred CUDA events, not kernel-only or isolated wall latency',
        limitations='No actual SGVD runtime was admitted, therefore SGVD decode/E2E P50/P95 and speedups are unavailable, not zero or theoretical.')
    write_json(root/'end_to_end_comparison.json',comparison)
    write_json(root/'sgvd_runtime_results.json',dict(implemented=False,executed=False,production_enabled=False,
        actual_sgvd_accepted_tokens=None,actual_sgvd_verification_calls=0,actual_native_decode_calls_avoided=0,
        actual_target_model_forward_calls_sgvd=None,actual_speedup=None,
        default_native_generation_unchanged=True,reason=exact['decision'],
        shadow_source='phase1/probes.jsonl',shadow_savings_are_not_runtime_speedup=True))
    write_json(root/'natural_sequence_results.json',dict(phase1=phase1_status,service=service_status,
        requests=[{k:r[k] for k in ('key','population','scene','frame','timestamp','ordinal','task','question',
            'historical_draft_available','stop_reason','parsed','generated_token_ids')} for r in requests],
        scene_reset='Every scene/population creates a fresh HistoricalDraftCache; tests cover reset/version/task invalidation',
        gaps_s=sorted({r['gap_s'] for r in read_rows(root/'phase1/probes.jsonl')}),
        population='exploratory_reused_population',source='phase1/protocol.json; phase1/requests.jsonl; service/protocol.json'))
    write_json(root/'downstream_task_quality.json',dict(frozen_native_service_pass=service_status['frozen_quality_pass'],
        checks=len(checks),passes=sum(q['frozen_targets_pass'] for q in checks),
        detection_map_temporal_eva_full_execution=True,
        temporal_memory_exact=sum(q['temporal_memory_exact'] for q in checks),feature_exact=sum(q['feature_exact'] for q in checks),
        token_ids_equal=sum(q['token_ids_equal'] for q in checks),
        planning=[dict(population=r['population'],mode=r['mode'],scene=r['scene'],frame=r['frame'],seed=r['seed'],metrics=r['planning']) for r in runs],
        native_parse_fraction=sum(r['parsed'] for r in requests)/len(requests),
        sgvd_task_quality_pass=False,sgvd_task_quality_evaluated=False,
        reason='Real target distribution stop gate; no SGVD task outputs were generated or qualified',
        text_correctness='Native-vs-frozen exact text regression only; no new human semantic labels or official nuScenes mAP claim',
        gt_usage='Only offline planning_metric using valid gt_planning_mask, outside timers; never draft selection'))
    categorical=sampling['passed']
    fields=dict(historical_draft_available=any(r['historical_draft_available'] for r in requests),
        draft_acceptance_sufficient=bool(gate['profitable_candidates']),
        verification_profitable=bool(gate['profitable_candidates']),
        verification_profitable_scope='Exploratory native-prefix expected cost only, not a runtime gain',
        sampling_distribution_correct=bool(categorical and numerical['parallel_native_target_bitwise_equal']),
        categorical_sampling_primitive_correct=categorical,
        sgvd_runtime_implemented=False,sgvd_task_quality_pass=False,
        sgvd_decode_speedup=None,sgvd_e2e_speedup=None,sgvd_production_enabled=False,
        remaining_bottleneck='Original FP16 dense GEMM and charged M1 maintenance/full prefill; '
            'parallel-vs-serial target arithmetic is an additional exact-distribution barrier',
        next_research_priority='Profile original dense GEMM bandwidth/compute and full-maintenance cost under frozen output contracts; '
            'require a serial/parallel target-distribution certificate before any future SGVD execution',
        formal_go=False,decision=exact['decision'],independent_heldout=False,
        unexecuted='Experimental SGVD and its GPU optimizations stop at the target-distribution gate; no fabricated speedup or quality.')
    write_json(root/'formal_go_nogo.json',fields)
    lines=['# M3 SGVD 连续驾驶验证报告','',
        '本轮完成真实六摄像头连续 nuScenes 历史答案采集、Teacher Forcing、CUDA 成本测量、采样原语验证、并行/串行目标分布及拒绝后 KV 检查，以及完整 OmniDrive 冻结服务质量回归。',
        '',f"结论：`{exact['decision']}`。没有安装或启用 SGVD Generation Loop；原生采样参数、320 Token 预算、M1/M2、任务质量门、Prefix 策略和模型权重保持原样。",
        '',f"历史答案可用 {sum(r['historical_draft_available'] for r in requests)}/{len(requests)} 次请求；所有数据标记为 exploratory_reused_population，没有独立人口资格。",
        f"小词表采样原语验证通过：{categorical}。真实 FP16 并行与串行处理后目标分布逐位相同：{numerical['parallel_native_target_bitwise_equal']}；最大 TV={numerical['max_target_total_variation']:.9g}。数学上的接受/拒绝恒等式不能消除上游目标分布差异。",
        '', '部分固定长度的离线预计收益为正，但离线原生前缀探测不代表拒绝后随机路径的收益，也未减少正式 Decode Forward。没有将共同前缀、影子节省或单次并行 Forward 数量当作实际加速。',
        '', '| 工作负载 | 方法 | E2E P50 / P95 ms | 冻结质量通过 / 检查 |', '|---|---|---:|---:|']
    for r in table:
        lines.append(f"| {r['population']} | {r['mode']} | {r['e2e_wall_ms']['p50']:.2f} / {r['e2e_wall_ms']['p95']:.2f} | {r['frozen_quality_passes']} / {r['frozen_quality_checks']} |")
    alt=json.loads((root/'alternative_kv_feasibility.json').read_text())
    lines.extend(['',f"完整质量回归 {sum(q['frozen_targets_pass'] for q in checks)}/{len(checks)}；CUDA 工程测试 {regression['tests']} 项，失败 {regression['failures']}、错误 {regression['errors']}、跳过 {regression['skipped']}。",
        '',f"备用 KV 分析：实测 aten::cat GPU 归属 {alt['measured_cat_self_cuda_ms_per_decode']:.4f} ms/Token；移除全部 cat 的乐观 Decode 上界为 {alt['optimistic_decode_speedup_upper_bound']:.4f} 倍。该上界并非 Adapter 实测收益。旧版 Attention 内部直接拼接，纯 Tuple Adapter 无法消除它；预分配视图还会改变 stride，因此没有改写 Attention。",
        '', '扩展多问题存在同帧任务，但原始生成串行执行，现有 BatchedExecutor 对 do_sample=True 保守回退；没有构造虚假并发，也未用贪心替换原始采样。',
        '', '完整服务各次都运行 EVA、Detection、Map、Temporal Memory、LLM、Planning。质量为相对原模型的冻结回归；Planning ADE/FDE 只使用合法 GT Mask 离线评价，不冒充完整官方 mAP 或驾驶安全验证。冷启动、第二帧、独立采样 Seed、所有未命中和不同间隔分别保留。',
        '', '旧 CUDA Graph 负面基线以来源与 SHA 归档引用，没有重捕获 160 个子图。新实验不覆盖历史目录。实际命令见 commands.sh，逐请求来源/Token/时序见 phase1/requests.jsonl，原始完整模型输出见 phase1/raw_outputs 与 service/raw_outputs。',
        '', 'SGVD 未执行，因此其任务质量、Decode/E2E 收益没有可报告的实测值；正式 No-Go 不用理论收益替代。下一步应聚焦原始 FP16 GEMM 与完整 M1 维护成本，而不是继续新增缓存或启发式解码。',
        '', '```json',json.dumps(fields,ensure_ascii=False,indent=2),'```',''])
    (root/'experiment_report.md').write_text('\n'.join(lines))
    (root/'implementation_changes.md').write_text('''# Implementation changes

- `worldstream/m3/historical_draft.py`: bounded immutable history and strict full text prompt/scene/camera/task/model/tokenizer/generation contracts. Explicit completed, parsed and submitted commits; no logits, GT or visual facts stored.
- `worldstream/m3/verified_sampling.py`: deterministic proposal acceptance and normalized rejection primitive, independently enumerated and Monte Carlo tested. No generation loop installation.
- `tools/worldstream_m3/sgvd/`: real complete-model history census, offline probability probes, paired CUDA costs, numerical target/KV checks, fallback KV analysis, frozen complete-service regression and artifact generation.
- Tests cover history ownership/budgets/reset/protocol/task invalidation and sampling support, EOS/joint law, extreme probabilities and processor order.
- Original Native Decode, M3Runtime/config/PrefixKVCache safety policy, M1/M2 and model files are unmodified. SGVD execution and optimization were stopped at the real target-distribution gate; no unused enable switch or unqualified decoder is installed.
''')
    artifacts=[p for p in root.glob('*.json') if p.name!='artifact_manifest.json']
    write_json(root/'artifact_manifest.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    print(json.dumps(fields,indent=2))


if __name__=='__main__':main()
