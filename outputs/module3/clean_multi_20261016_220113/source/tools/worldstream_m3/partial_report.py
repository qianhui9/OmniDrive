"""Explicit incomplete-run deliverable; never presents exploratory data as final."""
import argparse
import json
import shutil
from pathlib import Path
from .experiment import write,stats
from .report import read_lines,metric_summary,quality_summary,make_figures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='outputs/module3/run03')
    parser.add_argument('--output',default='outputs/module3');args=parser.parse_args()
    source=Path(args.run);root=Path(args.output);dest=root/'partial_delivery';dest.mkdir(exist_ok=True)
    rows=read_lines(source/'frames.jsonl')
    audit=json.loads((source/'input_audit.json').read_text())
    # Complete, balanced cells only. Still exploratory because allocator growth
    # and interrupted run invalidate the planned final benchmark protocol.
    modes=set(json.loads((source/'protocol.json').read_text())['modes'])
    keys={(r['repeat'],r['index']) for r in rows if not r['warmup']}
    complete={key for key in keys if {r['mode'] for r in rows if (r['repeat'],r['index'])==key}==modes}
    measured=[r for r in rows if not r['warmup'] and (r['repeat'],r['index']) in complete]
    latency=metric_summary(measured)
    available={(r['repeat'],r['index']) for r in rows if r['mode']=='baseline'}
    paired=[r for r in rows if (r['repeat'],r['index']) in available]
    quality=quality_summary(paired)
    status=dict(status='partial_delivery',final_gpu_experiments_complete=False,
        valid_for_final_performance_claims=False,source=str(source),recorded_frames=len(rows),
        complete_timed_frames=len(measured),cuda_failure='experiment wrapper retained old runtime/KV; fixed, clean rerun pending',
        current_blocker='Host exec automatic approval timed out; one retry also timed out',
        auto_approved=False)
    def artifact(name,body):
        value=dict(status,**body)
        write(dest/name,value);shutil.copy2(dest/name,root/name)
    baselines=[r for r in measured if r['mode']=='baseline']
    artifact('baseline_profile.json',dict(summary=latency['baseline'],rows=baselines,
        scope='Exploratory Transformer decoder intervals; full E2E wall includes complete service. Not final benchmark.'))
    artifact('token_statistics.json',dict(rows=audit,default_deployment_questions=1,selected_experiment_questions=5,
        visual_carriers=513,hidden_size=4096,all_question_counts=stats([r['actual_questions'] for r in audit]),
        text_tokens=stats([n for r in audit for n in r['text_tokens']]),
        expanded_tokens=stats([n for r in audit for n in r['expanded_tokens']]),
        same_frame_lcp=stats([n for r in audit for n in r['pairwise_exact_lcp']]),
        cross_frame_lcp=stats([r['cross_frame_first_question_lcp'] for r in audit if r['cross_frame_first_question_lcp'] is not None])))
    artifact('compact_input_results.json',dict(task_candidate_records=[dict(index=r['index'],repeat=r['repeat'],
        requests=r['diagnostics'].get('requests',[]),answers=r['answers']) for r in rows if r['mode']=='compact'],
        topk_random_quality_ablation='not_run: main run interrupted before paired LLM probes',online_enabled=False))
    artifact('prefix_match_results.json',dict(audit_source=str(source/'input_audit.json'),
        executed_suffix_lengths=[dict(index=r['index'],repeat=r['repeat'],mode=r['mode'],
            requests=[{k:d[k] for k in ('input_tokens','matched_prefix','reused_tokens','effective_prefill_tokens','forward_prefill_lengths') if k in d}
                      for d in r['diagnostics'].get('requests',[])]) for r in rows if r['mode'] in ('exact','m123')],
        profiler_kernel_proof='not_run: probes follow complete E2E and were not reached'))
    artifact('kv_correctness.json',dict(actual_gpu_token_comparisons=quality,
        real_gpu_logits_comparison='not_run',same_input_profiler='not_run',sampling_real_model='not_run',
        unit_test_evidence='logs/tests_cpu_final2.log and logs/tests_gpu.log; CPU seeded greedy/sampling and legacy KV correctness pass'))
    artifact('quality_results.json',dict(all_available_paired_gpu_output_checks=quality,
        frozen_target='Exact greedy token IDs; no relaxed threshold',
        gt_semantic_quality='not_run: no fabricated QA accuracy or road-safety certificate',
        all_failures_retained=True,online_compact=False,online_exact=False))
    artifact('latency_results.json',dict(exploratory_balanced_subset=latency,
        warning='OOM/incomplete protocol; these values are observations only, not a final service-speedup conclusion'))
    artifact('memory_profile.json',dict(rows=[dict(mode=r['mode'],index=r['index'],repeat=r['repeat'],
        allocated=r['peak_allocated'],reserved=r['peak_reserved'],cache_bytes=r['diagnostics'].get('cache_bytes',0)) for r in rows],
        diagnosis='The experiment-only instance wrapper formed a reference cycle; no final steady-state memory claim'))
    artifact('ablation_results.json',dict(measured_methods=sorted(modes),balanced_exploratory_subset=latency,
        not_run=['topk/random real-GT comparison','one-question deployment','fixed-RNG real-model sampling',
                 'whole-model prefix logits','profiler trace','clean full 6-scene repeat after lifetime fix']))
    artifact('cost_table.json',dict(approved=[],reason='No complete clean GPU calibration'))
    artifact('quality_certificate.json',dict(approved=[],reason='Observed token mismatches; quality certification not passed'))
    make_figures(dest,measured,audit,latency,quality,[],{},[],
        watermark='EXPLORATORY ONLY — interrupted OOM run; no final speedup claim')
    # Clearly watermark every available real-data plot; absent plots stay absent.
    # Figure titles/data sources remain explicit in the manifest/report.
    manifest=json.loads((dest/'figures/manifest.json').read_text())
    for name in list(manifest):manifest[name]=dict(source=manifest[name],status='exploratory incomplete OOM run; not final')
    manifest['m3c_compression_quality']=dict(status='not_run',reason='No completed paired real-GT LLM probe')
    manifest['m3e_reuse_latency']=dict(status='not_run',reason='No completed isolated prefix ablation')
    write(dest/'figures/manifest.json',manifest)
    shutil.copytree(dest/'figures',root/'figures',dirs_exist_ok=True)
    report='''# M3 实现与未完成 GPU 实验交付

**代码、OmniDrive 集成和回归测试已实现；完整 GPU 实验尚未完成，当前不能宣称满足质量约束的端到端加速。**

已实现 `worldstream/m3/` 的任务条件化保序选择、真实展开 Embedding 的精确 LCP、预算受控的 legacy KV、suffix-only Prefill、原 HF Greedy/Sampling/停止逻辑适配，以及成本/质量准入。Petr3D 在当前 Carrier 组装后调用 M3，关闭时保留原路径；启用时不会因回答文件存在跳过推理。默认 Auto 证据不足时完整回退，不修改权重或 M1/M2/Temporal Memory 规则。

## 已完成的真实审计

- RTX 3090，OmniDrive 环境：PyTorch 1.13.1+cu117、Transformers 4.31.0、PEFT 0.7.0，未升级依赖。
- 实际 Carrier `[1,513,4096]`：256 Detection extra queries、1 Ego/CAN bus、256 Map extra queries。它们不与预测框/车道/M1 Anchor 一一对应，禁止根据 Anchor ID 删除。
- 12 个真实帧、6 个场景，完整原始多问题负载每帧 13–17 问；默认部署配置只有 1 个 planning 问题。昂贵推理每帧固定选取 5 问，选择仅依赖问题文本。
- Image Token 原始位置为 35；实际多模态展开后同帧公共前缀 557–570 Token，覆盖全部视觉 Carrier；跨帧当前视觉变化，首问题仅共享 35 个纯文本 Token。
- 原生 `generate()` 拒绝外部 `inputs_embeds`；4.31 的 `prepare_inputs_for_generation` 会在有 KV 时只保留最后一个 Token。M3 使用独立 suffix forward，避免丢失多 Token 后缀或二次插入图像。

## 实际运行与失败

`run03/` 留下 119 个完整执行帧记录，包括 72 个预热帧以及未完成的计时序列；六方法为 baseline、M1/M2、Compact、Exact、Combined、M1/M2/M3。前 3 个场景有 36 个平衡计时帧，但整个正式协议未完成，且显存随实验模式切换增长，不能将该子集作为最终性能证据。

实际调用记录确认：同帧后续问题复用了约 557 Token，只执行各自后缀；探索中五问题累计 Transformer Prefill 常见约 860→290 ms。这里是探索观察，**不作为最终 Prefill 或端到端加速结论**。完整 wall、CUDA interval、调用次数、输出 Token IDs 和增长的显存均逐帧保存。

夜间场景 index=5489 的短驾驶动作问题出现 Exact Prefix 输出后半段措辞不同。输入依赖相同并不保证不同 FP16 Shape 计算逐位相同；逐层 KV/Logits 诊断尚未运行，不能确认具体数值原因。冻结的 Token 一致性目标维持失败，未放宽阈值。Compact/Combined 也出现输出变化，均不获在线批准。

OOM 原因已定位为**实验脚本**实例包装器的 `runtime → closure → bound runtime` 引用循环，使旧 KV 未及时释放。已改为类方法覆盖、显式 reset 和计时外回收，新增 weakref 生命周期回归已通过；真实 GPU 稳态复测尚待完成。生产 M3 没有这个实验包装器。

修复后准备重跑时，主机命令的自动权限审核连续超时，一次重试仍被拒绝，阻断新的 GPU 运行；这不是系统判定只读 GPU 查询不安全。用户的“不终止其他任务、定期检查空闲”要求始终保留。

## 最终问题逐项答复

1. 513 个视觉 Token；文本和展开长度逐问题保存在 token_statistics.json，不用一个问题的约 180 ms 代替五问题累计 Prefill。
2. 存在，实际展开后的同帧 LCP 为 557–570；跨帧只有 35。
3. suffix-only 实际调用已记录；独立 Profiler/kernel trace 尚未完成，不能宣称已完成全部跳算证明。
4. task 候选对 object_relation 问题把视觉 Token 从 513 减到 385，其他保守任务完整回退。
5. 部分真实回答发生变化；未完成可靠 GT 质量消融，不能宣称 QA 保持。
6. 对象关系只是候选域；地图、安全、全局/未知保留完整。没有获准在线压缩的任务类别。
7. M1 可约束场景/时效，没有可靠的 Anchor-to-carrier 对应。
8. 有探索阶段 Prefill 减少观察；完整干净实验、Decode 与全服务最终统计未完成。
9. 尚未证明满足冻结质量约束的 M3 端到端净加速。
10. 尚未证明满足冻结质量约束的 M1/M2/M3 净加速。
11. Decode 占比、数值一致性、近似质量、M1/Cache 成本和实验包装器显存问题均需区分。真实 Camera 命中以原始诊断为准。

## 交付与复现

代码说明见 `docs/worldstream_m3.md`；命令见 `commands.sh`。所有 JSON 顶层标注 partial_delivery / valid_for_final_performance_claims=false。`run01`、`run02`、`run03` 的失败/探索原始记录保留；既有 M1/M2 交付文件没有修改。Checkpoint SHA 见 checkpoint.sha256。

已有图 M3(a)/(b)/(d)/(f)/(g) 仅展示标记为探索性的真实数据。M3(c) 与 M3(e) 缺少对应完整实验，明确不生成曲线；见 figures/manifest.json。恢复 GPU 主机权限后，用修复后的脚本在全新输出目录重跑，随后运行正式 report 工具；不可从文件存在跳过 LLM。
'''
    (dest/'experiment_report.md').write_text(report)
    shutil.copy2(dest/'experiment_report.md',root/'experiment_report.md')
    write(root/'delivery_status.json',status)


if __name__=='__main__':main()
