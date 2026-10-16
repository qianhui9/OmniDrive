"""Aggregate only executed refinement runs; preserve all original M2 artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .common import write
from .refinement_common import ROOT


def read(name):return json.loads((ROOT/name).read_text())


def pct(values):return np.percentile(values,[50,95]).tolist() if values else None


def main():
    from .validate_quality_policy import main as publish_quality_policy
    publish_quality_policy()
    data=read('e2e_latency.json');runs=data['runs'];quality=data['quality']
    block=read('block_benchmark.json')['rows'];costs=read('break_even_results.json')
    for row in block:
        assert any(record['block']==row['block'] and record['updated']==row['updated'] and
                   json.loads(record['key'])['input_stride']==row['input_stride']
                   for record in costs['records']), 'Block timing stride and published key differ'
    summary={}
    for controlled in (False,True):
        for mode in sorted({r['mode'] for r in runs if r['controlled']==controlled}):
            records=[r for r in runs if r['controlled']==controlled and r['mode']==mode and not r['warmup']]
            frames=[f for r in records for f in r['frames']]
            key=('controlled_' if controlled else 'real_')+mode
            summary[key]={k:pct([f[k] for f in frames]) for k in
                ('complete_wall_ms','backbone_gpu_interval_ms','vision_gpu_interval_ms','m1_ms','m2_extra_interval_ms')}
            summary[key].update(frame_timings=len(frames),unique_frames=len({f['frame_id'] for f in frames}),
                code_stage='before_output_layout_preservation' if controlled else 'current_dense_fallback_path',
                cold_wall_ms=pct([f['complete_wall_ms'] for f in frames if f['cold']]),
                steady_wall_ms=pct([f['complete_wall_ms'] for f in frames if not f['cold']]),
                skipped_window_evaluations=sum(f['m2'].get('actual_skipped_windows',0) for f in frames),
                fallback_reasons=dict(Counter(f['m2'].get('fallback_reason','per_block') for f in frames)),
                cache_saved_frames=sum(f['m2'].get('cache_saved',False) for f in frames),
                peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in records),
                process_peak_cpu_rss_bytes=max(r['peak_cpu_rss_bytes'] for r in records))
    write(ROOT/'summary.json',summary)
    q=[r for r in quality if r['controlled'] and r['mode']=='optimized' and r['repeat']>0 and r['actual_skipped_windows']>0]
    pre_layout_q=list(q)
    latest_audit=read('quality_audit_results.json') if (ROOT/'quality_audit_results.json').exists() else {}
    if latest_audit.get('output_layout_preserved'):
        q=[r for r in latest_audit['quality'] if r.get('input_case')=='changed' and
           r['mode']=='optimized' and r['actual_skipped_windows']>0]
    realq=[r for r in quality if not r['controlled'] and r['mode']!='baseline']
    unique=[f for r in runs if not r['controlled'] and r['mode']=='optimized' and r['repeat']==1 for f in r['frames']]
    windows=[w for f in unique for c in f['m1_windows'].values() for w in c['windows']]
    realdiag=dict(unique_frames=len(unique),attention_windows=len(windows),
        candidate_windows=sum(w['candidate'] for w in windows),
        accepted_windows=sum(f['m2'].get('actual_skipped_windows',0) for f in unique),
        stable_patch_candidates=sum(c['stable_patches'] for f in unique for c in f['m1_windows'].values()),
        supported_and_exact_patches=sum(c.get('supported_and_exact_patches',0) for f in unique for c in f['m1_windows'].values()),
        max_window_stable_fraction=max(w['stable']/w['valid_tokens'] for w in windows),
        largest_stable_patch_component=max(c['largest_stable_component'] for f in unique for c in f['m1_windows'].values()),
        per_frame=unique)
    write(ROOT/'real_m1_diagnosis.json',realdiag)
    task=dict(nonzero_sparse_frame_timings=len(q),unique_nonzero_frames=len({(r['scene'],r['frame']) for r in q}),
        planning_pairs_scored=sum(r['planning'].get('ade_m') is not None and r['baseline_planning'].get('ade_m') is not None for r in q),
        max_backbone_abs=max((r['feature']['max_abs'] for r in q),default=0),
        max_detection_center_m=max((r['detection'].get('max_center_difference_m') or 0 for r in q),default=0),
        max_unmatched_detections=max((r['detection'].get('unmatched',0) for r in q),default=0),
        max_map_raw_abs=max((r['map_numeric']['max_abs'] for r in q),default=0),
        max_memory_raw_abs=max((r['memory_numeric']['max_abs'] for r in q),default=0),
        text_equal=sum(r['text_equal'] for r in q),
        planning_ade_delta_max=max((abs(r['planning']['ade_m']-r['baseline_planning']['ade_m'])
            for r in q if r['planning'].get('ade_m') is not None and r['baseline_planning'].get('ade_m') is not None),default=0),
        all_real_fallback_numeric_equal=all(r['feature']['max_abs']==0 and r['output_numeric']['max_abs']==0
            and r['memory_numeric']['max_abs']==0 and r['text_equal'] for r in realq))
    task['worst_feature_locations']=sorted(q,key=lambda r:r['feature']['max_abs'],reverse=True)[:5]
    write(ROOT/'task_quality.json',task)
    # Device kernel work is separate from event intervals, copies, and host launch gaps.
    profiles=read('packing_profile.json')
    for key,record in profiles.items():
        for name,values in record.items():
            trace=read(f'traces/block{key}_{name}.json')['traceEvents']
            kernels=[e for e in trace if e.get('cat')=='kernel']
            values['kernel_count']=len(kernels)
            values['kernel_active_ms']=sum(e.get('dur',0) for e in kernels)/1000
    write(ROOT/'packing_profile.json',profiles)
    base=summary['real_baseline']['complete_wall_ms'][0]
    rows=[]
    for mode in ('baseline','original','optimized','auto'):
        s=summary['real_'+mode]
        rows.append(f"| {mode} | {s['complete_wall_ms'][0]:.2f}/{s['complete_wall_ms'][1]:.2f} | {s['backbone_gpu_interval_ms'][0]:.2f} | {s['m1_ms'][0]:.2f} | {s['m2_extra_interval_ms'][0]:.3f}/{s['m2_extra_interval_ms'][1]:.3f} | {base/s['complete_wall_ms'][0]:.3f}× | {s['skipped_window_evaluations']} |")
    e2etable='\n'.join(rows)
    blocktable='\n'.join(f"| {b['block']} | {b['updated']}/54 | {b['timing']['full']['gpu_ms']:.3f} | {b['timing']['original']['gpu_ms']:.3f} | {b['timing']['optimized']['gpu_ms']:.3f} | {b['timing']['naive']['gpu_ms']:.3f} | {b['timing']['original']['gpu_ms']/b['timing']['optimized']['gpu_ms']:.3f}× |" for b in block)
    bb=read('backbone_benchmark.json')
    bbtable='\n'.join(f"| {r['updated']}/54 | {r['timing']['full']['gpu_ms']:.2f} | {r['timing']['original']['gpu_ms']:.2f} | {r['timing']['optimized']['gpu_ms']:.2f} | {r['timing']['full']['gpu_ms']/r['timing']['optimized']['gpu_ms']:.3f}× |" for r in bb)
    correct=read('correctness_report.json')
    hardware=json.loads(costs['records'][0]['key'])
    independent=[c for r in correct['independent_inputs'] for c in r['cases']]
    passes=sum(c['within_frozen_envelope'] for c in independent)
    protocol=read('protocol.json')['scenes']
    scene_table='\n'.join(f"| {s['category']} | {s['scene_name']} | {s['turn_rad']:.3f} | {s['max_object_speed_mps']:.2f} | {s['low_visibility_fraction']:.3f} |" for s in protocol)
    targets=read('quality_targets.json')
    target_pass=task['max_unmatched_detections']==0 and task['max_detection_center_m']<=targets['detection']['max_center_difference_m'] and task['text_equal']==len(q) and task['planning_ade_delta_max']==0 and task['planning_pairs_scored']==len(q)
    speedup=base/summary['real_optimized']['complete_wall_ms'][0]
    net_answer=('未获得净加速' if speedup<=1 else '本组样本中位数测得正收益')+f'（{speedup:.3f}×）'
    # Map raw maximum mixes coordinate and score leaves; expose each leaf separately.
    map_leaves={}
    for r in q:
        for leaf in r['map_numeric']['leaves']:
            map_leaves[leaf['path']]=max(map_leaves.get(leaf['path'],0),leaf['max_abs'])
    task['map_leaf_maxima']=map_leaves;task['detection_text_planning_targets_pass']=target_pass
    task['all_frozen_regression_targets_pass']=target_pass and map_leaves.get('root[0][0]',float('inf'))<=targets['map_max_control_point_difference_m']
    task['pre_output_layout_fix']=dict(nonzero_checks=len(pre_layout_q),
        max_detection_center_m=max(r['detection'].get('max_center_difference_m') or 0 for r in pre_layout_q),
        max_map_raw_abs=max(r['map_numeric']['max_abs'] for r in pre_layout_q))
    audit=read('quality_audit_results.json') if (ROOT/'quality_audit_results.json').exists() else None
    aligned={}
    if audit:
        aq=[x for x in audit['quality'] if x['mode']=='optimized' and x['actual_skipped_windows']>0 and x.get('input_case','changed')=='changed']
        for branch in ('det','map'):
            aligned[branch]=dict(max_matched_point_error_m=max(x['slot_alignment'][branch]['max_matched_point_error_m'] for x in aq),
                max_embedding_abs=max(x['slot_alignment'][branch]['leaves']['memory_embedding']['max_abs'] for x in aq),
                every_slot_matched=all(x['slot_alignment'][branch]['matched']==x['slot_alignment'][branch]['total'] for x in aq),
                aligned_timestamp_max_abs=max(x['slot_alignment'][branch]['leaves']['memory_timestamp']['max_abs'] for x in aq))
        task['memory_alignment']=aligned
    write(ROOT/'task_quality.json',task)
    archived=read('archived_m1_spatial_diagnosis.json')
    archive_counts=[sum(c['stable_patches'] for c in r['cameras'].values()) for r in archived['rows']]
    archive_fraction=max(w['stable']/w['valid_tokens'] for r in archived['rows'] for c in r['cameras'].values() for w in c['windows'])
    capture=read('cache_capture_benchmark.json')
    auto=read('auto_runtime_benchmark.json')['rows']
    formal_gate=read('formal_quality_gate.json')
    autotable='\n'.join(f"| {r['updated']}/54 | {r['timing']['gpu_ms']:.2f} | {r['full']['gpu_ms']:.2f} | {r['speedup']:.3f}× | {r['diagnostics']['actual_skipped_windows']} | {r['diagnostics']['stages'][0].get('fallback_reason')} |" for r in auto)
    memtable='\n'.join(f"| {m} | {summary['real_'+m]['peak_gpu_allocated_bytes']/2**30:.2f} | {summary['real_'+m]['process_peak_cpu_rss_bytes']/2**30:.2f} |" for m in ('baseline','original','optimized','auto'))
    worst=[];seen=set()
    for row in sorted(q,key=lambda r:r['feature']['max_abs'],reverse=True):
        key=(row['scene'],row['frame'])
        if key in seen:continue
        seen.add(key);e=row['feature']
        worst.append(f"| {row['scene'][:8]} / {row['ordinal']} | {e['max_abs']:.6g} | {e['index']} | {e['baseline']:.6g} | {e['actual']:.6g} |")
    worst_table='\n'.join(worst)
    tests=ET.parse(ROOT/'test_logs/unit_tests.xml').getroot()[0].attrib
    report=f'''# OmniDrive M2 定向改进与真实 GPU 验证

本轮修复 Full Fallback→Cache→Sparse 状态转换，独立校准 Block 0/1 成本，减少小批量
窗口整理工作，并补充真实完整模型中非零跳算的下游对照。所有新结果在本目录，
原 `outputs/module2/` 由 SHA256 清单逐文件验证，未覆盖。没有修改模型权重、EVA 窗口大小或 M1 判定门限。
另修复最终特征输出 stride 与原始 Forward 不同的问题，避免后续算子因内存布局选择不同实现。
真实四场景全为 dense fallback，其计时不受此后增量返回格式修正影响；完整 EVA/Auto 与质量审计已按最终格式复测。
实测环境：{hardware['gpu'][0]}，PyTorch {hardware['torch']}，CUDA {hardware['cuda']}，
Python 环境 OmniDrive；权重 `ckpts/iter_10548.pth`，原 FP32 激活与 FlashAttention 内部 FP16 Q/K/V。
所有计时均为实际执行，未把理论 FLOPs 或未运行配置填入表格。

主要结果：真实 M1 场景的 M2 附加 interval P50 从
{summary['real_original']['m2_extra_interval_ms'][0]:.3f} ms 降至
{summary['real_optimized']['m2_extra_interval_ms'][0]:.3f} ms；计入全部 M1/M2 与 LLM 后，
完整 VLM {net_answer}。受控完整模型确有非零跳算，但改变输入尚未通过冻结的下游目标，
正式 Auto 只批准更新数 `{formal_gate['approved_counts']}`，未以放宽精度换取加速。

## 1. 缓存连续性与重复计算

原 `_dense()` 确实清空缓存并丢弃当前 full forward 的中间输出，但它本身没有重复运行
backbone；问题是下一帧无法继承当前真实执行结果。现在通过临时 hook 捕获同一次原始
full forward 的 Block 0/1，finally 移除 hook，预算允许后原子发布当前 input/outputs/context。
不兼容历史先失效；仅更新范围太大时仍能保存当前输出。Cache version 单调递增，
camera 顺序独立快照，旧输出不会被原地覆盖。`always/adaptive/never` 明确控制捕获。

`adaptive` 在真实 M1 没有完整窗口或要求强制刷新时不保存，记录
`capture_not_expected_to_pay`；其理由是当前没有后续收益证据。强制捕获的 Full→Sparse→Full→Sparse
测试验证每个 Full 只有一次 Block forward，下一次 Sparse 从前一帧读取版本，且对照原始 EVA 正确。
预算与 OOM 失败继续完整路径，未保留未经完成的缓存。

## 2. 零机会快速回退

优先读取同一 M1Result 对应的 CPU mask，避免重复解压。廉价采样只证明拒绝：每个窗口
只要一个已失效像素，就不能完整复用；所有窗口都被证明失效时不上传 mask、不构造 GPU closure、
不 gather/scatter，也不创建无收益缓存。采样没有发现失效不是通过，仍检查完整 pixel/patch footprint。
M1 precheck/commit、Detection/Map/Temporal Memory/LLM 都保留完整执行。

初次把完整 mask 归约搬到 CPU，实测附加成本约 1.7 ms，比原路径约 1.35 ms 更慢；
该失败候选已停止，部分数据保留在 `integration/`。最终数据来自 `integration_final/`，
不把失败候选混入最终汇总。必要条件的离线诊断在 12/12 次转换排除全部同坐标窗口复用，
但没有因此跳过 M1 更新：绕过 M1 需要额外验证显式 reset 的状态语义，本轮没有启用。

## 3. 每个实际 Block 的成本与 Scheduler

Schema v2 显式记录 block index、checkpoint 内容 SHA256（参数及 RoPE buffer）、输入
shape/stride/dtype、GPU 型号/架构/显存、PyTorch/CUDA、Flash backend、norm epsilon、
attention scale、autocast/TF32。权重替换或原地版本变化使内容指纹重新计算；首次哈希成本属于初始化。
旧 schema 和未知条件失败关闭；不插值，不将 Block 0 的记录套给 Block 1。
Block 1 的 dense 与 packed 输入 stride 可能不同，另行测量 `layout_cost_supplement.json`。

Auto 只接受经过整体独立精度验证的实测更新数，且总 GPU interval 与 wall 都快于 full。
最终输出布局下的改变输入 sparse 仍没有通过下游 1cm 回归目标。最终正式表批准的更新数为
`{formal_gate['approved_counts']}`；其余数量保留成本与 backbone 精度记录，以 downstream 质量原因回退。
完全重复输入也必须先通过完整模型实际测试，不能只凭 backbone 值相同而忽略下游 stride。
Exact 强制路径仍可用于实验，不能把它当作已通过下游任务质量认证的默认路径。
Spatial packing 若没有同时快于 original 的两项测量，就保留 original packing；记录 backend 选择。
Full/Original/Optimized/Naive 各 20 次测量、5 次预热，方法顺序随 Block/数量反转。
表中为 GPU interval 中位数（ms）；CPU/wall/P95 和每次样本都在 JSON 中。
最终 Block 微基准显式保持原始输入 stride，成本 key 从实际计时 Tensor 生成；
旧版重建为 contiguous 却记录原 stride 的测量已移至 `pre_cost_layout/`，不用于正式表。

| Block | 更新 | Full | Original packed | Spatial packed | Naive | Original/Spatial |
|---|---:|---:|---:|---:|---:|---:|
{blocktable}

Spatial gather 仅读取选中窗口，缓存单个 shape 的 padding/坐标元数据，避免全量
`partition(x)` 和每次 `ones_like()`。仍复制完整输出 cache，保留历史引用不可变；没有为了省
几十微秒引入未验证的双缓冲覆盖。Profiler traces 记录实际 linear shape、kernel-active、
kernel launch 与设备拷贝；kernel-active 不等于包含 host gap 的 CUDA Event interval。
沿用已验证普通 CUDA 执行，本轮没有把固定地址 CUDA Graph 微基准当作跨帧运行时收益。
原 Block 的 MLP 只作用于 40×40 有效 token；Packed 在补齐窗口上执行 MLP。
全更新时 48×48 相比 40×40 多 44% token 工作，这也是高更新密度不利的原因，不能只归因于 launch。

## 4. 完整 EVA 与独立数值检查

下表受控随机输入只证明执行器能力，不代表真实 M1 覆盖或完整 VLM 收益。
完整 EVA 7 次测量、3 次预热，每次恢复同一个真实历史 full-forward cache，包含认证、
打包、scatter、缓存发布；没有使用当前帧预先算好的结果冒充复用。

| 更新窗口 | Full EVA ms | Original M2 ms | Optimized M2 ms | EVA speedup |
|---|---:|---:|---:|---:|
{bbtable}

新的 seed=811/1207 是此前未用的独立输入，{passes}/{len(independent)} 个完整 EVA 局部变化条件
通过之前冻结的逐层数值范围。没有按本轮结果扩大 MaxAbs 或 RelL2 容差。
每层和最终 MaxAbs 的 camera/token/channel 位置、baseline/actual 数值均写入
`correctness_report.json`；独立 dense camera-microbatch 差异也保留。大 MaxAbs 不直接等同任务失败，
但也不能凭较小整体 L2 认定任务安全，以下单独比较下游输出。

单次 Full 捕获成本微基准（7 次测量/3 次预热）：原始 full GPU {capture['full']['gpu_ms']:.3f} ms，
带 runtime 但不保存 {capture['no_capture']['gpu_ms']:.3f} ms，冷状态捕获 {capture['capture']['gpu_ms']:.3f} ms。
后两者差包含 hook/缓存发布，不能假定捕获免费。独立 Auto seed=1909 的实际执行：

| 更新 | Auto GPU ms | Full GPU ms | GPU speedup | 跳过窗口计算数 | 第0层回退原因 |
|---|---:|---:|---:|---:|---|
{autotable}

## 5. M1 粒度与真实 Dependency Closure

离线只用元数据选 4 个独立 Scene，每个取索引 0/1/2/4 的时间有序片段（16 个唯一帧），
涵盖约 0.5/1.0 秒间隔；并非完整 Scene 或官方测试集。转弯/速度/低可见度用于描述采样，
不传入在线 M1 决策。低可见度元数据不是精确遮挡分割。

| 条件 | Scene | 片段 yaw 变化 rad | 最大标注目标速度 m/s | 低可见度标注比例 |
|---|---|---:|---:|---:|
{scene_table}

这 16 帧共统计 {realdiag['attention_windows']} 个实际 camera-window，完整 patch 候选累计
{realdiag['stable_patch_candidates']}，同时通过同坐标实际输入相等的支持 patch 为
{realdiag['supported_and_exact_patches']}。完整候选窗口 {realdiag['candidate_windows']}，实际复用窗口
{realdiag['accepted_windows']}。最密窗口的稳定 patch 比例 {realdiag['max_window_stable_fraction']:.3%}，
最大四邻接稳定 patch 连通块 {realdiag['largest_stable_patch_component']} 个。
每窗 stable/unknown/changed patch 数及 40×40 分布保存在 `real_m1_diagnosis.json`。
这批高运动/低可见度片段在完整 patch 级已经没有候选，不能用它代表所有 Scene。
另外只读重分析原模块二停车场 Scene 的 640×640 lossless masks：各帧有 {archive_counts} 个
完整 patch，最密窗口达到 {archive_fraction:.2%} 稳定 patch，仍没有完整窗口。
这证明确有局部集中的支持却不足以覆盖注意力单元。该归档重分析独立标注在
`archived_m1_spatial_diagnosis.json`，不混入本轮四场景计时；归档没有 changed/unknown
逐像素分离证据，所以不伪造这两类计数。

未知/变化像素影响完整 16-pixel patch，任一无效 token 影响完整 16-token window，
Global block 2 再扩展到该相机的 1600 token。本轮仅前两层可独立复用，之后完整执行。
稳定像素与局部集中候选不足以形成完整注意力单元；移动相机下光流对应还不等于相同
normalized pixels/absolute position/RoPE/上游激活，故不能把几何移动后的 token 直接继承。
本轮不缩小 EVA window、不改变刷新区域、不把 `transformer_dependency_certified=False` 忽略。

## 6. 非零真实 GPU 跳算的完整模型质量

另在 turn/straight 两个 Scene 的实际完整 OmniDrive 上构造相同的受控输入序列：
以首帧预处理输入作底，下一帧局部 patch +0.03，再 +0.06，最后回到原 patch 并改变边缘 patch。
其他 pose/时间/任务输入按片段执行，每种模式使用同一序列；这是合成局部变化，不能代表真实移动相机收益。
两个 Scene 各 4 个受控输入，其中 {task['unique_nonzero_frames']} 个唯一输入实际发生部分窗口跳算；
测量重放中有 {task['nonzero_sparse_frame_timings']} 次非零跳算质量对照。LLM 每次真实执行，greedy、seed=0，
每次独立语言输出目录，不读取旧答案跳过推理。
这里以最终 output-stride 修正后的专项审计为质量结论：两 Scene 各 4 帧、一轮实际执行，
不将主计时中旧输出格式的 12 次非零对照混作最终代码的重复证据。旧数据保存在 `pre_output_layout/` 和主计时文件。

- 最终 EVA MaxAbs 最大 {task['max_backbone_abs']:.6g}，具体空间位置和对应值在 `task_quality.json`。
- 同类检测按中心位置 Hungarian 匹配，最大中心差 {task['max_detection_center_m']:.6g} m，未匹配数最大 {task['max_unmatched_detections']}。
- Map 各坐标/分数叶的最大差为 `{map_leaves}`，不把分数和米混为一种单位。
- Temporal Memory 原始张量最大差 {task['max_memory_raw_abs']:.6g}；各 embedding/reference-point/timestamp 叶分别记录。
- greedy 文本完全相同 {task['text_equal']}/{len(q)}，相对同输入 baseline 的 planning ADE 最大变化 {task['planning_ade_delta_max']:.6g} m。
- 可解析且有 GT mask 的 planning 对照为 {task['planning_pairs_scored']}/{len(q)}；未解析/不可评分不会被当成零误差。
- 在看到受控结果前冻结的检测中心 1cm、无未匹配、文本/规划不变目标，检测与文本/规划通过：{target_pass}。
- 连同 Map 坐标 1cm 的全部冻结回归目标通过：{task['all_frozen_regression_targets_pass']}。

最终特征最大误差位置（索引为 B,N,C,H,W，末两项是 40×40 token 坐标）：

| Scene前缀/序号 | MaxAbs | 索引 | Full值 | Incremental值 |
|---|---:|---|---:|---:|
{worst_table}

专项审计保留原始 dtype 的 Memory NPZ，按 timestamp/有效 pose 分组，再对 reference points
作一一匹配，并对全部 embedding/pose/velocity 等字段使用同一个排列，只读诊断，不修改模型状态。
汇总为 `{aligned}`。未匹配原始张量约 90 的差异主要涉及 top-k slot 排序；匹配后仍有厘米级 residual，
不是完全等价。`pre_update_memory` 的前缀截断还可能使 rank 改变影响后续保留集合，因此不能只用
一次重排后的较小误差宣称长期安全。Map 匹配后的控制点差异也记录在专项结果中。
正式 Auto 没有降低 1cm 目标以保留正收益；改变输入的 sparse 默认拒绝，Exact 强制实验继续留档。
完全重复输入的完整模型非零跳算检查 {formal_gate['repeat_nonzero_frame_checks']} 次，
feature/detection/map/native-dtype Memory/text 完全一致：{formal_gate['repeat_full_model_pass']}。
这组结果和其余有限的改变输入回归检查分开，不能外推成驾驶安全认证。

这是短序列的数值/输出/规划回归检查，不是全面驾驶问答语义评分、官方 detection/map AP 或道路安全认证。
尤其 Memory 排列和 Map raw 差异不能单靠全局 L2 掩盖；全部叶数据和逐帧比较都保留，
若目标未通过，就不能宣称任务质量安全。没有实现/启用 Approximate Experimental 特征重采样，
没有经过本轮下游任务验证的近似复用结论；受控 Exact 的结果不能代替 Approximate 的质量证明。

## 7. 真实 M1 + 完整 VLM 性能

四种模式采用按 Scene/replay 轮换的 Latin-square 顺序，1 次预热、2 次测量重放，
每模式 32 个帧计时、16 个唯一帧。CPU 数据预加载与 scatter 在计时前，完整同步 wall
包含 M1 输入/输出适配、验证、commit、M2、所有 heads/memory 和实际 LLM。诊断拷贝在计时后。

| 模式 | E2E P50/P95 ms | EVA P50 ms | M1 P50 ms | M2附加 interval P50/P95 ms | E2E speedup | 跳算计数 |
|---|---:|---:|---:|---:|---:|---:|
{e2etable}

完整回退结果的 feature/detection/map/memory/text 按主计时诊断精度相同：{task['all_real_fallback_numeric_equal']}。
主计时诊断数组转为 FP32；专项 Memory 审计另保存原始 dtype（包括 float64 时间戳），不据前者宣称 double 的逐位一致。
计入 M1/M2 全部成本后，Optimized 相对 Baseline **{net_answer}**。
冷首帧、稳态、fallback reasons、cache-save 和 CPU/GPU 内存见 `summary.json`。
其中 `controlled_*` 条目保留主计时阶段的旧输出格式数据，并标注 code_stage；
最终格式的受控实验来自 `quality_audit_layout/`，只作一轮质量复测，不混成重复预热性能结论。
CPU RSS peak 是进程生命周期高水位（包含模型加载与 CPU oracle），不是隔离每个模式的瞬时峰值。
GPU allocated/reserved 每次运行重置 peak；oracle hook/reference 的固定开销同样用于各模式。

| 模式 | GPU peak allocated GiB | 进程 CPU RSS 高水位 GiB |
|---|---:|---:|
{memtable}

Optimized 与 Original 的 E2E 中位数差不能全归功于 M2：本轮两者 M1 中位数分别为
{summary['real_optimized']['m1_ms'][0]:.2f}/{summary['real_original']['m1_ms'][0]:.2f} ms，存在 CPU 调度/测量波动。
可以明确归因的是 M2 自身额外 interval 的下降；不把约几十毫秒整体差异当作该优化的确定收益。
四 Scene 和重复帧不能当作大量独立统计样本；不声明显著性或未测试场景的收益。

## 8. 九项交付结论

1. 原 M2 存在不必要缓存丢弃；没有为了建缓存重复 full forward。现在按条件捕获当前 full 输出。
2. Full→Sparse 的正确状态转换有实际 EVA 单测、版本/Scene/Camera/权重/OOM 覆盖。
3. 小更新量 Spatial packing 更快；高更新量收益不稳定，保留 original 或 full，不隐去负收益。
4. Auto 使用独立 Block/布局/权重成本；未知数量/不兼容表/无收益/未通过整体数值检查都不批准 sparse。
5. 当前 M1 候选在空间上不完整，且 same-coordinate equality 是额外屏障；窗口与 global closure 限制复用。
6. 没有已获任务质量认证的近似几何特征复用路径；默认也未启用。
7. 真实 M1 接受/跳算数以本轮统计为准；受控完整模型确有非零真实 GPU 跳算，二者分开。
8. 是否净加速以完整 E2E 表为准，M1/M2 所有开销均计入，没有用 kernel 加速替代完整 VLM 结论。
9. 即使减少 M2 整理开销，M1 数百毫秒成本和只可跳早期两层的结构上限仍是主要瓶颈；
   不靠不合法历史 token、训练或改模型语义制造正收益。

测试：{tests['tests']} 项，失败 {tests['failures']}、错误 {tests['errors']}、跳过 {tests['skipped']}。
复现入口、冻结协议、源代码/旧结果 SHA256 在本目录；原结果未修改。

![Packing latency](figures/packing_latency.png)
![Real complete latency](figures/real_latency.png)
![M1 window coverage](figures/m1_window_coverage.png)
![Controlled nonzero quality](figures/controlled_quality.png)
'''
    (ROOT/'refinement_report.md').write_text(report)
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in
            ('worldstream/m2','tools/worldstream_m2','tests/worldstream_m2') for p in Path(folder).glob('*.py')}
    for p in (Path('docs/worldstream_m2.md'),
              Path('projects/configs/OmniDrive/worldstream_m2_refined_auto.py'),ROOT/'commands.sh'):
        source[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(ROOT/'source_manifest.json',source)
    original=read('original_artifacts_sha256.json')
    unchanged=all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original.items())
    write(ROOT/'artifact_preservation.json',dict(original_module2_unchanged=unchanged,files=len(original)))
    if not unchanged:raise AssertionError('Original module2 artifacts changed')
    print('report complete; controlled task targets',target_pass,'original artifacts unchanged',unchanged)


if __name__=='__main__':main()
