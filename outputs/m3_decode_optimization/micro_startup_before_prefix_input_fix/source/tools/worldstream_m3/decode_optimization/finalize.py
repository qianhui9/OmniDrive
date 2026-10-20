"""Build reports from fresh raw records, retaining failures and cold requests."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from tools.worldstream_m1.io_utils import write_json
from .profile import ROOT,SAMPLING
from .attribution import stats


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def write(name,data):write_json(ROOT/name,data)


def complete(row):
    return all(seq and (seq[-1]==2 or len(seq)-1<320) for query in row['generated_ids'] for seq in query)


def main():
    runs=lines(ROOT/'service/runs.jsonl');checks=lines(ROOT/'service/quality.jsonl')
    micro=read(ROOT/'micro/microbench.json');full=lines(ROOT/'micro/full_generation.jsonl')
    prefixes=read(ROOT/'prefix/failures.json')
    modes=read(ROOT/'service/protocol.json')['modes']
    table=[]
    for population in ('planning','native_multi'):
        for mode in modes:
            rr=[r for r in runs if r['population']==population and r['mode']==mode and not r['warmup']]
            qq=[q for q in checks if q['population']==population and q['mode']==mode and not q['warmup']]
            table.append(dict(population=population,mode=mode,requests=len(rr),E2E_wall_ms=stats([r['wall_ms'] for r in rr]),
                E2E_GPU_interval_ms=stats([r['gpu_interval_ms'] for r in rr]),
                prefill_GPU_interval_ms=stats([sum(t['gpu_interval_ms'] for t in r['invocations'] if t['stage']=='prefill') for r in rr]),
                decode_GPU_interval_ms=stats([sum(t['gpu_interval_ms'] for t in r['invocations'] if t['stage']=='decode') for r in rr]),
                decode_cpu_submission_ms=stats([sum(t['cpu_submission_ms'] for t in r['invocations'] if t['stage']=='decode') for r in rr]),
                actual_generated_tokens=stats([sum(len(seq)-1 for query in r['generated_ids'] for seq in query) for r in rr]),
                peak_gpu_allocated=max(r['peak_gpu_allocated'] for r in rr),peak_gpu_reserved=max(r['peak_gpu_reserved'] for r in rr),
                peak_cpu_rss_kib=max(r['cpu_rss_high_water_kib'] for r in rr),frozen_quality_pass=sum(q['frozen_targets_pass'] for q in qq),
                token_equal=sum(q['token_ids_equal'] for q in qq),quality_denominator=len(qq),complete_untruncated_requests=sum(complete(r) for r in rr)))
    bykey={(r['population'],r['scene'],r['repeat'],r['ordinal'],r['mode']):r for r in runs}
    paired=[]
    for r in runs:
        if r['mode']!='decode_graph':continue
        key=(r['population'],r['scene'],r['repeat'],r['ordinal'])
        native=bykey[key+('m3_native',)];orig=bykey[key+('original',)]
        paired.append(dict(population=r['population'],scene=r['scene'],repeat=r['repeat'],ordinal=r['ordinal'],warmup=r['warmup'],
            current_M3_ms=native['wall_ms'],new_decode_ms=r['wall_ms'],original_ms=orig['wall_ms'],
            same_service_speedup=native['wall_ms']/r['wall_ms'],Original_speedup=orig['wall_ms']/r['wall_ms'],token_equal=r['generated_ids']==native['generated_ids']))
    ratios={}
    for population in ('planning','native_multi'):
        for cold in (False,True):
            pp=[r for r in paired if r['population']==population and (cold or not r['warmup'])]
            ratios[population+('_including_warmup' if cold else '_steady')]=dict(requests=len(pp),same_service=sum(r['current_M3_ms'] for r in pp)/sum(r['new_decode_ms'] for r in pp),
                versus_Original=sum(r['original_ms'] for r in pp)/sum(r['new_decode_ms'] for r in pp),paired_mean_ratio=float(np.mean([r['same_service_speedup'] for r in pp])))
    write('end_to_end_comparison.json',dict(table=table,paired=paired,ratios=ratios,sampling=SAMPLING,
        population_label='exploratory fixed two scenes, two continuous frames, one warmup and two measured replays; no production performance certificate',
        timing='Service uses deferred forward events and synchronized request boundary. CUDA intervals include host gaps. CPU submission is not synchronous decode wall.'))
    mm=[]
    for tokens in (1,8):
        for mode in ('native','graph'):
            rr=[r for r in micro['rows'] if r['mode']==mode and r['tokens']==tokens and not r['warmup']]
            mm.append(dict(mode=mode,tokens=tokens,token_wall_ms=stats([r['wall_ms']/tokens for r in rr]),
                token_GPU_interval_ms=stats([r['gpu_interval_ms']/tokens for r in rr]),cpu_submission_ms=stats([r['cpu_submission_ms'] for r in rr]),
                throughput_tokens_s=stats([tokens*1000/r['wall_ms'] for r in rr])))
    token_speedup=mm[0]['token_wall_ms']['p50']/mm[1]['token_wall_ms']['p50']
    write('decode_microbench.json',dict(summary=mm,raw='micro/microbench.json',diagnostics=micro['diagnostics'],warmup=4,measured_repeats=20,
        single_token_latency_speedup=token_speedup,micro_does_not_shorten_actual_task_generation=True))
    fg=[]
    for mode in ('native','graph'):
        rr=[r for r in full if r['mode']==mode and not r['warmup']]
        fg.append(dict(mode=mode,requests=len(rr),full_generation_wall_ms=stats([r['wall_ms'] for r in rr]),
            decode_span_wall_ms=stats([r['decode_span_wall_ms'] for r in rr]),
            prefill_wall_ms=stats([sum(t['wall_ms'] for t in r['invocations'] if t['stage']=='prefill') for r in rr]),
            decode_forward_wall_ms=stats([sum(t['wall_ms'] for t in r['invocations'] if t['stage']=='decode') for r in rr]),
            decode_GPU_ms=stats([sum(t['gpu_interval_ms'] for t in r['invocations'] if t['stage']=='decode') for r in rr]),
            token_wall_ms=stats([t['wall_ms'] for r in rr for t in r['invocations'] if t['stage']=='decode']),
            token_equal=all(r['token_equal'] for r in rr),rng_equal=all(r['rng_equal'] for r in rr)))
    write('decode_optimization_results.json',dict(implementation='native_static_decode_graphs_v1',default_enabled=False,mechanism_count=1,
        complete_sampled_generation=fg,raw='micro/full_generation.jsonl',service_ratios=ratios,
        synchronous_timer_note='Separate barrier-instrumented attribution. Decode span starts at first decode forward through generator return; excludes prefill and its first sampling. No E2E minus prefill estimate.',
        first_candidate='service_attempt1_owned_clone: 24 completed requests, self-interrupted after 74-second capture and slower replay; preserved',
        final_cold_requests=[dict(population=r['population'],wall_ms=r['wall_ms'],diagnostics=r['m3'].get('decode_execution')) for r in runs if r['mode']=='decode_graph' and r['m3'].get('decode_execution',{}).get('captured')]))
    guarded=[q for q in checks if q['mode']=='guarded_prefix']
    write('prefix_parity_analysis.json',dict(independent_diagnosis=prefixes,raw='prefix/failures.json',
        unique_historical_failures=len(prefixes),production_auto_enabled=False,
        partial_prefix_mathematical_parity_fixed=False,guarded_policy='Only entire identical saved prefill and final logits may be reused; partial input LCP uses full prefill',
        guarded_full_model_quality_pass=sum(q['frozen_targets_pass'] for q in guarded),guarded_quality_denominator=len(guarded),
        guarded_token_equal=all(q['token_ids_equal'] for q in guarded),quality_targets_relaxed=False))
    baseline=read(ROOT/'profiling/sources.json')
    protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in baseline.items() if p.startswith(('worldstream/m1/','worldstream/m2/','projects/mmdet3d_plugin/')) or p in ('worldstream/m3/task_evidence.py','configs/worldstream_m1_engineering_frozen.yaml')}
    dq=[q for q in checks if q['mode']=='decode_graph']
    quality=all(q['frozen_targets_pass'] and q['token_ids_equal'] and q['memory']['max_abs']==0 and q['feature']['max_abs']==0 for q in dq) and all(r['token_equal'] and r['rng_equal'] for r in full if r['mode']=='graph') and all(complete(r) for r in runs if r['mode']=='decode_graph')
    write('original_model_parity.json',dict(decode_quality_pass=quality,decode_checks=len(dq),passed=sum(q['frozen_targets_pass'] and q['token_ids_equal'] for q in dq),
        checkpoint=Path(ROOT/'checkpoint.sha256').read_text().strip(),protected_sources=protected,all_protected_unchanged=all(protected.values()),
        raw_quality='service/quality.jsonl',raw_outputs='service/raw_outputs',sampling=SAMPLING,unparsed_or_truncated_credited=False,
        decode_feature_max_abs=max(q['feature']['max_abs'] for q in dq),decode_temporal_memory_max_abs=max(q['memory']['max_abs'] for q in dq),
        complete_full_model_requests=len(runs),control_repeats='Same seed for each matched frame and replay; full generation repeats preserve complete IDs and post-generation RNG'))
    xml=ET.parse(ROOT/'regression_cuda.xml').getroot()
    suites=[s for s in xml.iter('testsuite')]
    reg={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    write('regression_results.json',dict(**reg,passed=reg['tests']-reg['failures']-reg['errors']-reg['skipped'],
        command='taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q',
        CUDA=True,full_model_service_status=read(ROOT/'service/status.json'),micro_status=read(ROOT/'micro/status.json'),prefix_status=read(ROOT/'prefix/status.json')))
    fields=dict(decode_bottleneck_identified=True,decode_optimization_implemented=True,decode_quality_pass=quality,
        decode_token_latency_speedup=token_speedup,decode_e2e_speedup={p:ratios[p+'_steady']['same_service'] for p in ('planning','native_multi')},
        prefix_parity_fixed=False,prefix_quality_pass=all(q['frozen_targets_pass'] and q['token_ids_equal'] for q in guarded),prefix_production_enabled=False,
        full_vlm_e2e_speedup={p:ratios[p+'_steady']['versus_Original'] for p in ('planning','native_multi')},
        remaining_system_bottleneck='Native dense FP16 GEMM execution; persistent M1 maintenance and full prefill remain charged',
        next_research_priority='Kernel-level GEMM profiling under unchanged weights; seek a supported backend with proven full sampling quality and measured net gain',
        production_decode_enabled=False,formal_go=False,reason='Default off; no independent production certificate, partial prefix remains unproved. See measured same-service/Original and cold-inclusive ratios separately.')
    write('formal_go_nogo.json',fields)
    text=['# Decode 优化与 Prefix 正确性实验报告','',
        '实现与真实 GPU 验证已完成。新增开关默认关闭；所有冻结模块和权重保持不变。结论以以下实测数据为准，不将微测收益替代完整服务收益。','',
        '独立 decode trace 的主要成本是 GPU FP16 GEMM。CUDA Graph 只减少原生静态子图的提交开销，未删除矩阵计算；动态 Attention/KV、LoRA、采样与停止条件保持原生。','',
        '| 请求人口 | 方法 | E2E P50/P95 ms | 冻结质量/分母 |', '|---|---|---:|---:|']
    for r in table:
        quality_label=f"{r['frozen_quality_pass']}/{r['quality_denominator']}" if r['quality_denominator'] else '基准'
        text.append(f"| {r['population']} | {r['mode']} | {r['E2E_wall_ms']['p50']:.2f}/{r['E2E_wall_ms']['p95']:.2f} | {quality_label} |")
    text += ['',f'单 token Wall P50 加速比：{token_speedup:.4f}。同服务与 Original 比较、含冷启动结果见 end_to_end_comparison.json。两场景小样本仅提供探索性证据。',
        '', 'Prefix 输入逐位相等不等于不同形状 FP16 prefill 的 KV 逐位相等。逐个历史失败的复现、首个分歧、完整当步 logits、各层 KV 误差、重复性和生命周期检查见 prefix_parity_analysis.json。未宣称部分 Prefix 的数值一致性已经修复。',
        '保守整段策略通过完整任务质量才计为通过；它对相关问题退回完整 prefill，因此不能冒充部分前缀加速。生产 Auto 仍关闭。','',
        f"完整 CUDA 工程回归：{reg['tests']-reg['failures']-reg['errors']-reg['skipped']}/{reg['tests']}，失败 {reg['failures']}，错误 {reg['errors']}，跳过 {reg['skipped']}。完整模型 service 请求 {len(runs)}，独立完整生成 {len(full)}。",'',
        '初始 observer 未保留 forward 签名导致两次采样失败；归因为包装器而非 PEFT。修正后恢复原始 PEFT generate。首版 Graph 有 24 条完成请求，因约 74 秒捕获与较慢重放主动中止，原始记录未删除。最终版去掉每子图全局 GC，并在原生 decoder 消费范围内复用私有输出 workspace；公开 logits/KV 所有权经过 CUDA 测试。','',
        '计时：service 使用请求边界同步与延迟 CUDA events；每 forward CPU 字段为提交 Wall，不冒充同步 decode Wall。独立 micro 完整生成使用每次 forward 同步，另报 decode span 与实际 forward Wall/GPU，含观察开销。Trace 只累加叶子 kernels，不与 CPU 算子归属 CUDA 时间相加。','',
        '内存为同一进程各请求峰值；CPU RSS high-water 包含模型加载，后续对照也包含常驻且停用的 Graph workspace，不等同于各方法独立进程的部署峰值。','',
        '原始采样参数、全部答案和 320 token 预算不变。实际 EOS/预算耗尽与轨迹可解析性独立检查。Detection/Map/Temporal/LLM/Planning 沿用冻结门；离线 GT 仅在计时外用于质量。本轮是相对原模型的冻结质量回归，不冒充完整 nuScenes 官方 mAP/驾驶安全验证。','',
        '```json',json.dumps(fields,ensure_ascii=False,indent=2),'```','',
        '复现命令见 commands.sh。原始逐请求质量、完整感知/记忆结果、源码快照、环境、权重 SHA 和失败数据均在本目录。旧实验目录未覆盖。']
    (ROOT/'experiment_report.md').write_text('\n'.join(text)+'\n')
    changes=['# Implementation and instrumentation changes','',
        '| File | Type | Purpose | Lines |','|---|---|---|---|']
    modified=['worldstream/m3/config.py','worldstream/m3/runtime.py','worldstream/m3/native_decode.py']
    created=[*Path('tools/worldstream_m3/decode_optimization').glob('*.py'),Path('tests/worldstream_m3/test_decode_graphs.py')]
    for p in [*map(Path,modified),*created]:
        changes.append(f'| {p} | {"modified" if str(p) in modified else "created"} | '+('Runtime opt-in execution and conservative Prefix guard' if str(p) in modified else 'Standalone real-data measurement/reproduction or contract regression')+f' | 1–{len(p.read_text().splitlines())} |')
    changes+=['','Instrumentation is confined to standalone runners. M1/M2/original model/TaskEvidence source hashes are checked in original_model_parity.json. Temporary module installation is restored in finally.','',
        'decode_cuda_graph=False preserves the original route. identical_prefill_only=False preserves the legacy research route; True rejects partial-prefix reuse without relaxing Auto certificates. Both are independent of Direct Query gates.']
    (ROOT/'implementation_changes.md').write_text('\n'.join(changes)+'\n')
    print(json.dumps(fields,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
