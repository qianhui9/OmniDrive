"""Aggregate only executed runs, distinguish service contracts and controlled evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT,write,stats


def read(name):return json.loads((ROOT/name).read_text())


def attribute_kernel_ranges(trace,names):
    cpu={e.get('args',{}).get('External id'):e for e in trace if e.get('cat')=='cpu_op'}
    ranges=[e for e in trace if e.get('cat')=='user_annotation' and e['name'] in names]
    counts=Counter();active=Counter()
    for e in trace:
        if e.get('cat')!='kernel':continue
        launch=cpu.get(e.get('args',{}).get('External id'))
        matching=[r for r in ranges if launch and r.get('pid')==launch.get('pid') and
            r.get('tid')==launch.get('tid') and r['ts']<=launch['ts']<=r['ts']+r.get('dur',0)]
        name=min(matching,key=lambda r:r['dur'])['name'] if matching else 'unattributed'
        counts[name]+=1;active[name]+=e.get('dur',0)/1000
    return dict(counts),dict(active)


def main():
    from .feasibility import main as feasibility
    feasibility()  # Rebuild before appending pipeline tables, so reruns are idempotent.
    data=read('latency_results.json');runs=data['runs'];q=data['quality'];summary={}
    repeated=[x for x in q if x['case']=='repeat' and x['mode']=='camera_exact' and x['ordinal']>0]
    if read('camera_costs.json')['repeat_full_task_exact']:
        assert repeated and all(x['frozen_targets_pass'] and x['feature']['max_abs']==0 and
            x['memory']['max_abs']==0 and not x['memory']['shape_mismatches'] and
            x['output']['max_abs']==0 and not x['output']['shape_mismatches'] for x in repeated)
    for case in sorted({r['case'] for r in runs}):
        for mode in sorted({r['mode'] for r in runs if r['case']==case}):
            records=[r for r in runs if r['case']==case and r['mode']==mode and not r['warmup']]
            # Single quality runs are deliberately NOT presented as warmed performance.
            frames=[f for r in records for f in r['frames']]
            if not frames:continue
            components={}
            for name in sorted({n for f in frames for n in f['components']}):
                components[name]=dict(gpu_ms=stats([f['components'].get(name,{}).get('gpu_ms',0.) for f in frames]),
                    cpu_enqueue_wall_ms=stats([f['components'].get(name,{}).get('cpu_enqueue_wall_ms',0.) for f in frames]))
            summary[f'{case}_{mode}']=dict(complete_wall_ms=stats([f['complete_wall_ms'] for f in frames]),
                m1_ms=stats([f['m1_ms'] for f in frames]),gpu_interval_ms=stats([f['gpu_interval_ms'] for f in frames]),
                frame_timings=len(frames),unique_frame_ids=len({(r['scene'],f['index']) for r in records for f in r['frames']}),
                distinct_image_batches=(len({r['scene'] for r in records}) if case=='repeat' else
                    len({(r['scene'],f['index']) for r in records for f in r['frames']})),
                components=components,actual_reused_cameras=sum(f['runtime'].get('actual_reused_cameras',0) for f in frames),
                actual_skipped_windows=sum(f['runtime'].get('actual_skipped_windows',0) for f in frames),
                fallback_reasons=dict(Counter(f['runtime'].get('fallback_reason') or 'reuse_or_baseline' for f in frames)),
                cold_wall_ms=stats([f['complete_wall_ms'] for f in frames if f['cold']]),
                steady_wall_ms=stats([f['complete_wall_ms'] for f in frames if not f['cold']]),
                peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in records),
                peak_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in records),
                process_cpu_rss_highwater_bytes=max(r['cpu_rss_highwater_bytes'] for r in records))
    write(ROOT/'summary.json',summary)
    unique_original=[f for r in runs if r['case']=='real' and r['mode']=='original' and r['repeat']==1 for f in r['frames']]
    windows=[w for f in unique_original for c in f['windows'].values() for w in c['windows']]
    real_coverage=dict(unique_frames=len(unique_original),attention_windows=len(windows),
        m1_observation_candidate_windows=sum(w['candidate'] for w in windows),
        complete_observation_patches=sum(c['stable_patches'] for f in unique_original for c in f['windows'].values()),
        real_camera_reuse=sum(v['actual_reused_cameras'] for k,v in summary.items() if k=='real_camera_auto'),
        real_window_skips=sum(v['actual_skipped_windows'] for k,v in summary.items() if k=='real_original'))
    write(ROOT/'real_reuse_coverage.json',real_coverage)
    real={m:v for m,v in summary.items() if m.startswith('real_')}
    write(ROOT/'ablation_results.json',dict(real=real,notes='maintain, explicit reset/reinitialize and output-only have different WorldState service guarantees; never combine as equivalent histories'))
    def qstats(rows):
        return dict(checks=len(rows),frozen_target_pass=sum(x['frozen_targets_pass'] for x in rows),
            max_feature_abs=max((x['feature']['max_abs'] for x in rows),default=None),
            max_detection_center_m=max((x['detection'].get('max_center_difference_m') or 0 for x in rows),default=None),
            max_map_control_abs_m=max((v['max_abs'] for x in rows for v in x['map']['leaves'] if v['path']=='root[0][0]'),default=None),
            max_native_memory_raw_abs=max((x['memory']['max_abs'] for x in rows),default=None),
            text_equal=sum(x['text_equal'] for x in rows),planning_equal=sum(x['planning_equal'] for x in rows))
    quality={f'{case}_{mode}':qstats([x for x in q if x['case']==case and x['mode']==mode])
             for case in ('real','mixed','repeat') for mode in sorted({x['mode'] for x in q if x['case']==case})}
    write(ROOT/'task_quality_summary.json',quality)
    write(ROOT/'camera_reuse_results.json',dict(costs=read('camera_costs.json'),quality=quality,
        balanced_formal_auto=read('balanced_camera_results.json'),
        balanced_real_continuous_m1=read('balanced_real_camera_results.json'),
        scope='controlled repeated input and camera subsets; distinct from real moving cameras',
        latency={k:v for k,v in summary.items() if 'camera' in k}))
    gates=[dict(scene=r['scene'],mode=r['mode'],repeat=r['repeat'],frames=[dict(index=f['index'],
        admission=f['admission'],state=f['m1_state'],m1_ms=f['m1_ms']) for f in r['frames']])
        for r in runs if r['case']=='real' and r['mode'] in ('gate_maintain','gate_reset','gate_output')]
    write(ROOT/'cheap_gate_results.json',dict(runs=gates,quality={k:v for k,v in quality.items() if k.startswith('real_gate')},
        timing={k:v for k,v in summary.items() if k.startswith('real_gate')},
        contract='rejection only; reset_on_reject resets before current precheck+commit; output_only resets and skips M1, next validation starts from empty state; Head memory is not reset'))
    drift=[x for x in q if x['case'] in ('mixed','repeat')]
    write(ROOT/'state_drift_results.json',dict(sequence_length=16,quality=drift,
        native_memory_source='integration_final/*_memory.npz',
        limitations='16 keyframes per sequence (~7.5s), not full scenes or long-horizon road safety; no approximate chained drift claim'))
    # Profiler kernel times are distinct from event intervals and exclude memcpy/memset.
    trace=read('traces/full_pipeline.json')['traceEvents']
    kernels=[e for e in trace if e.get('cat')=='kernel']
    cpu={e.get('args',{}).get('External id'):e for e in trace if e.get('cat')=='cpu_op'}
    ranges=[e for e in trace if e.get('cat')=='user_annotation' and
            (e['name'].startswith('block_') or e['name'] in ('patch_embedding','detection_head','map_head','carrier_query_assembly','llm_prefill','llm_decode'))]
    byrange=Counter();active=Counter()
    for e in kernels:
        launch=cpu.get(e.get('args',{}).get('External id'))
        matching=[r for r in ranges if launch and r.get('pid')==launch.get('pid') and r.get('tid')==launch.get('tid') and
                  r['ts']<=launch['ts']<=r['ts']+r.get('dur',0)]
        name=min(matching,key=lambda r:r['dur'])['name'] if matching else 'unattributed'
        byrange[name]+=1;active[name]+=e.get('dur',0)/1000
    base=summary['real_baseline']
    carrier_profile=read('carrier_profile.json')
    supplemental_components=carrier_profile['summary']
    supplement_count,supplement_active=attribute_kernel_ranges(read('traces/carrier_profile.json')['traceEvents'],
        {'det_carrier_output_projection','map_carrier_output_projection','carrier_query_assembly','detection_head','map_head'})
    write(ROOT/'pipeline_profile.json',dict(components=base['components'],carrier_supplement=supplemental_components,total_kernel_count=len(kernels),
        carrier_supplement_kernel_counts=supplement_count,carrier_supplement_kernel_active_ms=supplement_active,
        kernel_count_by_range=dict(byrange),kernel_active_ms_by_range=dict(active),
        kernel_active_total_ms=sum(e.get('dur',0) for e in kernels)/1000,
        notes='One separate warmed original full-model profile; no profiler kernel sum used as E2E latency. Head query generation is included in heads; carrier assembly is multimodal input preparation.'))
    profile_rows=[]
    for name,values in {**base['components'],**{k:v for k,v in supplemental_components.items() if 'carrier' in k}}.items():
        gpu=values['gpu_ms'];cpu_wall=values['cpu_enqueue_wall_ms']
        count=supplement_count.get(name,0) if 'carrier' in name else byrange.get(name,0)
        work=supplement_active.get(name,0.) if 'carrier' in name else active.get(name,0.)
        profile_rows.append(f"| {name} | {gpu['p50']:.3f}/{gpu['p95']:.3f} | {cpu_wall['p50']:.3f}/{cpu_wall['p95']:.3f} | {count} | {work:.3f} |")
    with (ROOT/'feasibility_report.md').open('a') as handle:
        handle.write('\n## 完整模型实测分解\n\nCUDA interval 与 CPU enqueue wall 为真实六场景、48 个测量帧的 P50/P95；kernel 为另一次预热后完整 trace。\n\n'
                     '| 阶段 | GPU interval ms | CPU enqueue wall ms | kernel数 | kernel active ms |\n'
                     '|---|---:|---:|---:|---:|\n'+'\n'.join(profile_rows)+'\n\n'
                     'Backbone 输出 permute/view 为元数据操作，其开销包含在 visual_service；有实际 layout copy 的相机组装计入 camera runtime。\n'
                     'Detection/Map 的 carrier query 生成包含在各 head 中；carrier_query_assembly 单列多模态拼接准备，不重复将 head 生成成本相加。\n'
                     'Carrier output_projection 与 assembly 来自独立四帧补充测量（一次预热、两次重放）；主计时对 Peft 外层的包装未触及实际 assembly，因此不伪造该缺失区间。补充 trace 为 traces/carrier_profile.json。\n'
                     f"完整 Baseline GPU peak allocated/reserved 为 {base['peak_gpu_allocated_bytes']/2**30:.3f}/{base['peak_gpu_reserved_bytes']/2**30:.3f} GiB，CPU RSS 生命周期高水位 {base['process_cpu_rss_highwater_bytes']/2**30:.3f} GiB。\n")
    numerical=read('numerical_raw.json')['cases']
    component=read('component_numerical_audit.json')
    component_table='\n'.join(f"| {r['block']} | {r['selected_windows']} | {r['metrics']['norm1']['max_abs']:.6g} | {r['metrics']['q_projection_fp32']['max_abs']:.6g} | {r['metrics']['q_flash_input_fp16_changed_elements']} | {r['metrics']['attention_output']['max_abs']:.6g} | {r['metrics']['mlp_same_input_valid_only']['max_abs']:.6g} |" for r in component['rows'])
    shadow=read('cache_substitution_shadow.json')['rows']
    shadow_max=max(r['final']['max_abs'] for r in shadow)
    ntable='\n'.join(f"| {x['seed']} | {x['case']} | {x['B_vs_A']['max_abs']:.6g} | {x['C_vs_A']['max_abs']:.6g} | {x['camera_microbatch_vs_A']['max_abs']:.6g} |" for x in numerical)
    (ROOT/'numerical_diagnosis.md').write_text(f'''# 数值误差归因

A=原始 EVA；B=前两层全部窗口走原 Packed Attention/MLP，不读取历史结果；
C=前两层闭包稀疏执行，其余保持原始 blocks。两个独立 seed，无阈值放宽。

| Seed | 输入 | B−A MaxAbs | C−A MaxAbs | 原始 Dense 单相机拆批−A MaxAbs |
|---|---|---:|---:|---:|
{ntable}

全部窗口的 B 在本组输入与 A 相同；Sparse 和纯 Dense 拆批都产生差异。
首层 Sparse 差异约 1e-3，后续 Global/deep blocks 放大；逐层位置/数值见 numerical_raw.json。
这支持 batch shape/GEMM/FP16 QKV 舍入是主要来源，不证明所有浮点误差都安全。
原始 norm1 在 padding 前；Packed 用 valid mask 将 padded norm1 保持零；RoPE 与原 attention 对象相同。
原始 MLP 在 unpad 后 40×40 token 上计算，Packed 对补齐的窗口计算；全量 B 无差异不表示所有子批次数值相同。
诊断恢复原始 MLP/unpad 顺序的结果也记录；没有修改原始 EVA Forward、attention scale 或 residual。
固定全量 batch 的 shadow 另计算全部窗口后，仅将闭包证明不变的窗口替换成旧 cache，
最终 MaxAbs 为 {shadow_max:.6g}，见 cache_substitution_shadow.json。这将历史替换本身与缩小 batch 的差异分开；
该 shadow 仍执行全部窗口，没有计算节省，也不作为在线优化。

同值输入的逐组件拆分进一步记录如下。QKV 在 FP32 中仅有数个 ULP 差异，也可能跨过 FP16 舍入边界；
attention_output 包括原 Flash/inner norm/output projection，不将整段差异全归给某个未隔离的 kernel。
K/V 同样逐项保留在 component_numerical_audit.json。Norm1 的零差异和固定 batch shadow 的零差异
提供缓存替换与浮点执行形状的区分证据，不把小特征误差当作任务安全。

| Block | 子批窗口 | Norm1 MaxAbs | Q投影 FP32 MaxAbs | Q进入Flash的FP16变化元素数 | Attention输出 MaxAbs | 同输入 MLP有效token MaxAbs |
|---|---:|---:|---:|---:|---:|---:|
{component_table}

Dense camera 拆批的 Patch Embedding MaxAbs 为 {component['patch_embedding_microbatch']['max_abs']:.6g}。
实际 matmul TF32 allowed={component['matmul_tf32_allowed']}，cuDNN TF32 allowed={component['cudnn_tf32_allowed']}，
这些是原始执行策略，未为通过目标而改变；allowed 不代表每个 conv kernel 都使用 TF32。

完整模型 A/B/C 与 camera 子批次的 Detection/Map/native Memory/greedy/planning 对照见 correctness_results.json
和 task_quality_summary.json。MaxAbs、同类检测匹配、Map 各叶与 Memory 联合匹配分别报告，
不能用全局 Relative L2 通过认证。未匹配 Memory 的大差异可涉及 top-k slot 顺序，匹配后残差和时间戳仍保留。
正式 Window Auto 沿用先前批准集合 [0]；没有因为 B 全量相同而批准 C 子批次。
Camera Auto 也只采用实际 table 标明的已验证数量；Exact 模式是显式实验开关。
''')
    modes=('baseline','original','gate_maintain','gate_reset','gate_output','camera_auto')
    table='\n'.join(f"| {m} | {summary['real_'+m]['complete_wall_ms']['p50']:.2f}/{summary['real_'+m]['complete_wall_ms']['p95']:.2f} | {summary['real_'+m]['m1_ms']['p50']:.2f} | {summary['real_'+m]['actual_reused_cameras']} | {summary['real_'+m]['actual_skipped_windows']} | {base['complete_wall_ms']['p50']/summary['real_'+m]['complete_wall_ms']['p50']:.4f}× |" for m in modes)
    rpt=summary['repeat_camera_exact'];rbase=summary['repeat_baseline']
    speed=rbase['complete_wall_ms']['p50']/rpt['complete_wall_ms']['p50']
    balanced=read('balanced_camera_results.json')
    real_balanced=read('balanced_real_camera_results.json')
    real_balanced_summary={}
    for mode in ('baseline','camera_auto_m1'):
        frames=[f for r in real_balanced['runs'] if r['mode']==mode and not r['warmup'] for f in r['frames']]
        real_balanced_summary[mode]=dict(wall_ms=stats([f['complete_wall_ms'] for f in frames]),
            reused_cameras=sum(f['runtime'].get('actual_reused_cameras',0) for f in frames),frames=len(frames))
    write(ROOT/'balanced_real_camera_summary.json',real_balanced_summary)
    balanced_summary={}
    for mode in ('baseline','camera_auto','camera_auto_m1'):
        frame=[f for r in balanced['runs'] if r['mode']==mode and not r['warmup'] for f in r['frames']]
        balanced_summary[mode]=dict(wall_ms=stats([f['complete_wall_ms'] for f in frame]),
            reused_cameras=sum(f['runtime'].get('actual_reused_cameras',0) for f in frame),frames=len(frame))
    write(ROOT/'balanced_camera_summary.json',balanced_summary)
    bspeed=balanced_summary['baseline']['wall_ms']['p50']/balanced_summary['camera_auto']['wall_ms']['p50']
    m1speed=balanced_summary['baseline']['wall_ms']['p50']/balanced_summary['camera_auto_m1']['wall_ms']['p50']
    bounds=read('cost_break_even.json')
    bounds['current_validation_medians_ms']={m:summary['real_'+m]['m1_ms']['p50']
        for m in ('original','gate_maintain','gate_reset','gate_output','camera_auto')}
    bounds['current_full_m1_window_no_go']=bounds['current_validation_medians_ms']['original']>next(
        r['saving_upper_bound_wall_ms'] for r in bounds['rows'] if r['scope']=='skip_first_2')
    bounds['measured_partial_window_service']=read('partial_window_costs.json')
    bounds['measured_camera_service']=read('camera_costs.json')
    bounds['measured_cheap_gate']=read('cheap_gate_microbenchmark.json')
    bounds['actual_real_reuse_coverage']=real_coverage
    write(ROOT/'cost_break_even.json',bounds)
    layout_profile=read('output_layout_profile.json')
    with (ROOT/'feasibility_report.md').open('a') as handle:
        handle.write(f"\n原始输出 permute/view 实测 CPU wall（每1000操作一组、30组的每操作平均）P50/P95 "
            f"{layout_profile['cpu_wall_ms_per_operation']['p50']:.6f}/{layout_profile['cpu_wall_ms_per_operation']['p95']:.6f} ms；"
            f"独立 trace 的 GPU kernel 数 {layout_profile['gpu_kernel_count']}。相机输出 copy 另计入执行器。\n")
    tests=ET.parse(ROOT/'logs/unit_tests.xml').getroot()[0].attrib
    policy=read('camera_costs.json')
    initialization=read('camera_signature_initialization.json')
    final_zero=[r for r in policy['records'] if r['changed_cameras']==0 and
        'numeric_details' in json.loads(r['key'])][-1]
    saving_per_hit=final_zero['full_wall_ms']-final_zero['reuse_wall_ms']
    initialization_hits=int(np.ceil(initialization['wall_ms']/saving_per_hit)) if saving_per_hit>0 else None
    bounds['initialization_wall_ms']=initialization['wall_ms']
    bounds['initialization_amortization_hits_current_output']=initialization_hits
    write(ROOT/'cost_break_even.json',bounds)
    visual=read('visual_service_results.json')['summary']
    approximate=read('approximate_probe_results.json')
    accepted_patch_count=sum(p.get('accepted_patches',0) for r in approximate['rows'] for p in r['probes'])
    visual_table='\n'.join(f"| {m} | {v['wall_ms']['p50']:.3f}/{v['wall_ms']['p95']:.3f} | {v['gpu_interval_ms']['p50']:.3f}/{v['gpu_interval_ms']['p95']:.3f} |" for m,v in visual.items())
    quality_table='\n'.join(f"| {name} | {v['frozen_target_pass']}/{v['checks']} | {v['max_feature_abs']:.6g} | {v['max_detection_center_m']:.6g} | {v['max_map_control_abs_m']:.6g} | {v['text_equal']}/{v['checks']} |" for name,v in quality.items())
    purposes=dict(phase_ab='GPU上限与逐层A/B/C',common='冻结参考/原始dtype比较/实验目录',
        feasibility='上限与认证成本的阶段决策',numerical_shadow='固定batch历史替换/真实kernel计数/部分窗口成本',
        component_audit='Norm/QKV/FP16/Attention/MLP数值拆分',integration='完整模型计时/Memory/greedy/场景对照',
        pipeline_probe='Peft实际carrier投影和组装计时',balanced_camera_probe='正式Auto的平衡顺序完整模型复测',
        repair_language_oracle='全新16帧语言Oracle与原始Memory核对',visual_service_probe='隔离到LLM入口的视觉服务计时',
        approximate='独立geometry-only/patch oracle必要条件',refresh_camera_cost='显式数值策略下的实测零更新成本',
        wait_gpu='按分钟检测可用GPU',monitor_gpu='按分钟核验外部竞争进程',postflight='补充验证/CUDA测试顺序编排',
        report='真实结果和kernel trace汇总',plot='测量数据的PDF/PNG导出',__init__='隔离实验包')
    instrumentation_table='\n'.join(f"| {p} | 新建 | {purposes.get(p.stem,'实验支持')} | — |" for p in
        sorted(Path('tools/worldstream_m2/rearchitecture').glob('*.py')))
    instrumentation_table+='\n| worldstream/m2/approximate_probe.py | 新建 | 仅离线特征对齐探针，生产Runtime不调用 | — |'
    report=f'''# OmniDrive M2 第三轮：计算边界、按需认证与相机级精确复用

研究决策：将 M2 转向**成本感知的更大粒度视觉状态执行**。保留 Window Packing 和 Dependency Closure，
但不继续以仅前两层局部跳算作为同步完整 M1 服务的核心收益来源。
本轮没有改变模型权重、窗口大小、刷新门限或先前冻结的下游目标，没有实现 M3。

## 实测依据与阶段决策

阶段 A 见 [feasibility_report.md](feasibility_report.md)。完整跳两层的上限低于历史 M1 成本，
Gate A No-Go；一个/多个相机全 Backbone 的收益上限更高，但只有严格相同输入才具有依赖条件。
阶段 B 见 [numerical_diagnosis.md](numerical_diagnosis.md)。全量 Packed 与 A 相同，子批次和 Dense 拆批出现差异，
不能把误差包络当作任务质量认证。Window Auto 维持 [0]，未扩大集合（Gate B）。
Camera 正式成本表批准的 changed_cameras 为 `{sorted({r['changed_cameras'] for r in policy['records'] if r['quality_certified']})}`，
只按实测 key/数量、GPU 和 wall 成本同时受益时复用。未验证部分相机路径保留为 Exact 实验，默认回退。
收益预算门另使用当前 input-adapter/precheck 实测加前帧 commit/output-adapter 成本估计。
Window Auto 若乐观的批准 Block 节省总和仍无法支付该预算，在 GPU closure 前拒绝；
Camera Auto 必须整体 wall 节省超过预算。它是简单的测量判据，不保证下一帧维护成本不会波动。
主16帧 camera_auto 是加入此预算门之前的诊断记录；最终含预算门的 Auto 以独立平衡复测为准。
真实六场景均零命中，预算门不改变其 GPU 执行分支；当前输出服务与完整 M1 服务分别报告。

## 完整模型真实六场景

冻结协议保留前轮四个 Scene，增加独立行人与夜间 Scene；每个索引 0/1/2/4，24 唯一帧。
1 次预热、2 次测量、Latin-square 模式轮换；逐序列重置，完整 Detection/Map/Memory/LLM 每帧执行。
CPU 数据加载/scatter 与诊断拷贝在计时外；M1、认证、cache、同步均在完整 wall 内。
CPU enqueue wall 与 CUDA interval 分开。所有模式都有同样的轻量 profiling wrapper 固定开销。
`original` 是本轮开始前保留的第二轮 M2 代码快照；不是重新实现的模拟基线。
冷缓存表示每段首帧的 cache warmup，不等于进程首次启动；首次 EVA 内容指纹初始化
另测为 {initialization['wall_ms']:.2f} ms（包括 device-to-host 读与 SHA256），不隐去该部署启动成本。
仅按 current-output 的每次重复命中节省估计，需约 {initialization_hits} 次命中才能摊回该额外初始化；
这是基于实测单次节省的摊销推算，不是含模型加载的冷进程 benchmark。

| 模式 | E2E P50/P95 ms | M1 P50 ms | 复用 Camera | 跳过 Window 计算 | E2E speedup |
|---|---:|---:|---:|---:|---:|
{table}

`gate_maintain` 保留完整 M1 历史，只增加拒绝筛查；`gate_reset` 在拒绝时明确丢弃旧 WorldState，
然后执行当前 precheck/commit；`gate_output` 是仅需当前感知的服务，在拒绝时显式 reset 并跳过 M1，
下一次验证从空状态初始化。后两者不是保留完整 Anchor 历史的等价实现，不可混报。
本表 `camera_auto` 同样采用 output_only 服务；默认配置保留完整 M1 的正式相机 Auto，
另在两个真实 Scene 平衡复测：Baseline P50/P95
{real_balanced_summary['baseline']['wall_ms']['p50']:.2f}/{real_balanced_summary['baseline']['wall_ms']['p95']:.2f} ms，
连续 M1 Camera Auto {real_balanced_summary['camera_auto_m1']['wall_ms']['p50']:.2f}/{real_balanced_summary['camera_auto_m1']['wall_ms']['p95']:.2f} ms，
每臂16计时、真实相机复用数 {real_balanced_summary['camera_auto_m1']['reused_cameras']}；该补充不混入六 Scene 的样本数。
所有路径继续当前 pose/time、Detection/Map/StreamPETR Memory 和真实语言生成；从不缓存当前 heads 结果。

单独的真实视觉服务计时从请求进入到实际 LLM 入口（包含 M1、缓存、规划、当前 heads/query 与同步），
两真实 Scene 各4帧、两轮旋转顺序，模型已经过完整 VLM 预热。该 component-only probe 在语言入口退出，
不将其时间当作完整 VLM。主 profile 的 `visual_service` 标签仅为 extract_img_feat，完整视觉前缀以此表为准：

| 模式 | 视觉服务 wall P50/P95 ms | CUDA interval P50/P95 ms |
|---|---:|---:|
{visual_table}

没有把不发生复用的延迟微小波动归因于 GPU 跳算；真实移动相机输入下实际复用数见表。
{real_coverage['unique_frames']} 个唯一真实帧统计 {real_coverage['attention_windows']} 个相机窗口，
M1 完整观测 patch 计数 {real_coverage['complete_observation_patches']}，完整候选窗口
{real_coverage['m1_observation_candidate_windows']}；观测候选不是神经特征等价证明。
真实是否净加速按各模式表格和质量结果判断，重复请求的结果不能当作真实驾驶覆盖率。
CPU RSS 是进程生命周期高水位；逐模式 GPU allocated/reserved peak 见 summary.json，不能混作隔离 CPU 峰值。

## 受控非零复用与连续状态

两个独立 Scene 各 16 个真实时间/pose/任务输入，合成 mixed 局部变化序列，另有完全重复图像请求序列。
mixed 同时覆盖单 patch、完整 window、多个 window、单个/多个相机相同及全变化。
完全重复序列保留真实变化的自车 pose、时间戳与下游 Memory；首帧实际完整计算供缓存。
16 帧是约 7.5 秒的有限状态漂移回归，不是完整场景或长期安全证明。

16帧重复图像请求的 warmed E2E：Baseline P50 {rbase['complete_wall_ms']['p50']:.2f} ms，
Camera Exact P50 {rpt['complete_wall_ms']['p50']:.2f} ms，受控加速 {speed:.3f}×。
这组16帧计时按方法成组，主要用于漂移回归；独立平衡顺序复测以正式 Camera Auto 为比较臂，
两个 Scene 各4帧、1轮预热和2轮计时、每臂16个计时，Baseline P50/P95
{balanced_summary['baseline']['wall_ms']['p50']:.2f}/{balanced_summary['baseline']['wall_ms']['p95']:.2f} ms，
Camera Auto {balanced_summary['camera_auto']['wall_ms']['p50']:.2f}/{balanced_summary['camera_auto']['wall_ms']['p95']:.2f} ms，
受控净收益 {bspeed:.3f}×。这是明确无 WorldState 连续性要求的 current-output 服务：
M1 manager 为空，完整逐像素相机输入等价仍由 M2 认证，heads/Memory/LLM 每次真实执行。
另一个保持完整 M1 连续性的正式 Auto 臂 P50/P95 为
{balanced_summary['camera_auto_m1']['wall_ms']['p50']:.2f}/{balanced_summary['camera_auto_m1']['wall_ms']['p95']:.2f} ms，
受控收益 {m1speed:.3f}×；它计入实际验证与前一帧维护成本估计，只有收益预算为正才批准复用。
上述收益指 P50 中位数；P95 未改善，包含冷缓存/状态维护成本，不声称整个尾部时延分布都更优。
平衡复测质量通过 {sum(x['frozen_targets_pass'] for x in balanced['quality'])}/{len(balanced['quality'])}，
原始精度 Memory、完整输出与 feature 差异也逐项保留。不将该服务与每帧 M1 的服务混为相同契约。
质量结果为 `{quality.get('repeat_camera_exact')}`；只有同输入下任务约束通过才称为受控净收益。
正式 camera_auto 在成本表发布后又实际执行完整模型验证，单轮质量结果单列，不冒充多轮性能。
初次中断留下的 4 个语言文件使最初 mixed Baseline 跳过了语言推理；这些单轮时间标记为无效，
未用于预热性能表。另在全新目录重放完整 16 帧 Baseline，逐帧核对原始精度 Memory 完全一致后，
用实际新生成文本重新比较 A/B/C。修正前原记录保留在 pre_language_oracle_recheck.json，过程见 language_oracle_recheck.json。

原始 dtype Memory NPZ、每帧质量与误差随时间数据见 state_drift_results.json 和 integration_final/；
raw 和同龄 slot 联合匹配同时保留，不能靠重排掩盖当前保留集合不同。
Mixed 强制实验中，后续仅恢复 full backbone 不能自动消除已经带偏的 Temporal Memory，
这也是拒绝未认证部分更新的原因；各独立序列/场景重置后从完整实际输出重新开始。

| 输入/执行模式 | 冻结目标通过 | EVA MaxAbs | 检测中心差 m | Map坐标差 m | 文本一致 |
|---|---:|---:|---:|---:|---:|
{quality_table}
场景、camera 顺序、版本、权重、预算、输入 stride、非有限值与时间倒退在运行时/单测中 fail closed。
相机 cache 保存计算来源 frame/timestamp，复用只更新验证请求上下文，不伪造新的计算来源。

## 世界对齐近似方向

approximate_probe_results.json 单独记录离线必要可行性筛查。先在当前 absolute position 之前
对齐内容表示；不将历史 RoPE 相关上下文 token 直接 warp。几何观测支持不是 Transformer 等价。
若整层 Patch Embedding 的最大节省已无法覆盖认证，Gate D No-Go，停止深层/下游近似实验，
相应字段明确 not_run，不填虚构质量数；不加入在线 Auto。
本次离线累计完整观测支持 patch 计数为 {accepted_patch_count}，重复帧/相机逐项记录；
筛查使用不含 Detection/Map 排除锚点的 geometry-only 乐观条件，不冒充部署候选覆盖率。
整层 patch oracle 节省能覆盖实测 precheck 下界的必要条件：
{approximate['optimistic_patch_saving_exceeds_validation']}。尚无下游任务认证的近似复用结论。

## 十项问题答复

1. Window 上限为 {next(r['saving_upper_bound_wall_ms'] for r in bounds['rows'] if r['scope']=='skip_first_2'):.2f} ms，完整输入认证/缓存后的部分更新成本另见 partial_window_costs.json。
2. 本轮原路径 M1 P50 为 {bounds['current_validation_medians_ms']['original']:.2f} ms（历史 {bounds['m1_previous_p50_ms']:.2f} ms）；同步完整认证超过该局部上限。
3. 全量 B 对照和 Dense microbatch 指向 batch shape 浮点差异；没有通过改阈值规避。
4. Window 正式集合未扩大，局部 sparse 未通过的任务目标仍导致回退。
5. Cheap Gate 仅拒绝；是否减少 M1 成本按上表消融判断，状态 reset/reentry 与连续服务分列。
6. 相机 batch 是更大合法边界，输入必须完整有限且逐值相等，签名、版本和顺序有效。
7. Camera GPU 成本见 camera_reuse_results.json；受控完整请求质量和时延如上，真实覆盖另列。
8. 不声称有获任务认证的世界对齐近似路径；必要条件失败后停止，没有默认启用。
9. 真实非零节省只能由实际计数支持，不将受控重复当成驾驶复用。
10. 完整 VLM 时延与任务约束同时检查；理想上限、受控实际、真实移动场景互相独立。

测试：{tests['tests']} 项，失败 {tests['failures']}、错误 {tests['errors']}、跳过 {tests['skipped']}。
复现见 commands.sh；阶段加载中断/失败候选在 logs/ 留档。原 M1/M2/Refinement 结果逐文件 SHA256 验证。

配置入口：`worldstream_m2_camera_auto.py` 保留 M1；`worldstream_m2_camera_output_only.py`
显式不维护 M1，匹配平衡复测的 current-output 臂；`worldstream_m2_camera_on_demand.py`
在拒绝时丢弃 M1 历史，下一次需要验证时重新初始化。具体契约见 `docs/worldstream_m2_rearchitecture.md`。

![Feasibility](figures/feasibility.png)
![Numerical propagation](figures/numerical_propagation.png)
![Real ablation](figures/real_ablation.png)
![State drift](figures/state_drift.png)
![Camera boundary](figures/camera_boundary.png)

## Profiling instrumentation changelog

| 文件 | 变更 | 用途 | 行号 |
|---|---|---|---|
{instrumentation_table}

Profiler 只由独立实验脚本包装，不写入 EVA Forward。生产代码中的 admission/cache 是功能路径，
不是计时埋点；无需运行 profiling 脚本即可使用。所有原始数据保留，可按需移除独立 profiling 工具。
'''
    (ROOT/'experiment_report.md').write_text(report)
    original=read('original_artifacts_sha256.json')
    failed=[p for p,h in original.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    write(ROOT/'artifact_preservation.json',dict(files=len(original),all_unchanged=not failed,failures=failed))
    if failed:raise AssertionError(failed)
    folders=('worldstream/m2','tools/worldstream_m2/rearchitecture','tests/worldstream_m2')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in folders for p in Path(folder).glob('*.py')}
    for p in [Path('projects/mmdet3d_plugin/models/detectors/petr3d.py'),Path('docs/worldstream_m2_rearchitecture.md'),ROOT/'commands.sh']+list(Path('projects/configs/OmniDrive').glob('worldstream_m2_camera*.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(ROOT/'source_manifest.json',sources)
    print('Final report generated; original artifacts unchanged:',len(original),flush=True)

if __name__=='__main__':main()
