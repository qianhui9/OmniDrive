"""Regenerate the report from retained measurements, never edit approvals."""
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from tools.worldstream_m2.window_validate import compare_task_runs

root=Path(__file__).resolve().parent
study=json.loads((root/'continuous40/results.json').read_text())
rows=[json.loads(s) for s in (root/'continuous40/frames.jsonl').read_text().splitlines()]
comparison=compare_task_runs(root/'native_tasks',root/'wsgrr_tasks',root/'task_comparison.json')
for source,name in (('/tmp/wsgrr_project_tests_final_cpu.log','tests_cpu.log'),
                    ('/tmp/wsgrr_continuous40.log','continuous40.log'),
                    ('/tmp/wsgrr_task_queue.log','task_queue.log')):
    shutil.copyfile(source,root/name)

def strategy(row):return row.get('strategy',row['case'].split('_')[0])
data=[r for r in rows if strategy(r)=='D3' and r['continuous']]
events=collections.Counter(e for r in data for region in r['region_refresh'].get('event_regions',[]) for e in region['events'])
event_regions=[e for r in data for e in r['region_refresh'].get('event_regions',[])]
def distribution(values):
    return dict(zip(('min','p50','p90','p99','max'),map(float,np.percentile(values,[0,50,90,99,100])))) if values else None

details=dict(event_counts=dict(events),event_region_records=len(event_regions),
    events_with_nonempty_windows=sum(bool(e['windows']) for e in event_regions),
    planned_event_missed_windows=sum(r['region_refresh'].get('event_missed_windows',0) for r in data),
    drift=distribution([v for r in data for v in r['region_refresh'].get('drift',[])]),
    ages=distribution([v for r in data for v in r['region_refresh'].get('ages',[])]),
    M1_regions=distribution([r['m1_regions'] for r in data]),
    timings={k:distribution([r[k] for r in data]) for k in ('M1_precheck_ms','M1_state_read_ms',
        'region_scoring_host_ms','window_and_suffix_host_ms','cache_publication_host_ms')},
    mandatory_windows=sum(r['region_refresh'].get('mandatory_windows',r['stages'][0]['total_windows'] if r['full_update_reason'] else 0) for r in data),
    ordinary_windows=sum(r['region_refresh'].get('ordinary_windows',0) for r in data),
    full_refresh_frames=[dict(frame_id=r['frame_id'],reason=r['full_update_reason']) for r in data if r['full_update_reason']],
    largest_feature_errors=sorted([dict(frame_id=r['frame_id'],**r['difference']) for r in data],key=lambda r:r['max_abs'],reverse=True)[:5])
(root/'analysis.json').write_text(json.dumps(details,indent=2,allow_nan=False))

source_files=[p for directory in ('worldstream/m1','worldstream/m2','tools/worldstream_m2','tests/worldstream_m2') for p in Path(directory).glob('*.py')]
source_files += [Path('tools/worldstream_m123/benchmark/worker.py'),Path('docs/worldstream_m2_window.md')]
manifest=dict(cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),python=__import__('sys').executable,
    changed_files=subprocess.run(['git','diff','--name-only'],capture_output=True,text=True).stdout.splitlines()+[
        'worldstream/m2/region_refresh.py','tests/worldstream_m2/test_wsgrr.py','outputs/wsgrrr_20261022/'],
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    formal_cost_table_sha256=hashlib.sha256(Path('outputs/final_route/window_costs.json').read_bytes()).hexdigest(),
    formal_cost_table_git_diff=subprocess.run(['git','diff','--','outputs/final_route/window_costs.json'],capture_output=True,text=True).stdout,
    production_approvals_added=0,task_quality_certified=False,
    note='Continuous40 ran the initial WSGRR implementation; final task smoke and cost_profile validate the final code with added state-horizon handling and richer diagnostics. Raw long-run files remain unchanged.')
(root/'implementation_manifest.json').write_text(json.dumps(manifest,indent=2))

lines=['# WSGRR 实现与真实权重实验报告（2026-10-22）','',
'已实现唯一 WSGRR 候选路线，保留原生精确离线基线；正式批准表未修改。窗口复用成立，但本次没有测得净视觉加速，完整任务质量仍未独立认证，不能宣称 M2 达标。','',
'## 修改范围','',
'- M1 state_types/computation_state：在既有 bundle 导出真实 WorldAnchor 区域、来源、可靠性和真实过期事件。未改写跟踪器，未增加检测器。',
'- M2 region_refresh（新增）/m1_adapter/config：区域状态比较、真实卷积映射、有效 Token 池化、漂移/年龄/不确定性评分、必须集合和每相机预算。',
'- dependency_planner/runtime：选定窗口进入原依赖闭包和唯一执行器；离线 A/B/C/D 入口与生产签名隔离；保留成本/质量准入，成本扩展重算时继续传播依赖。',
'- state_cache：输入、输出、区域状态、各层年龄、描述和来源同事务提交；策略进入签名，防止不同预算错误复用。',
'- window_validate/promote_costs/benchmark worker：真实连续帧消融、预算计时、可选块级成本、定量误差准入和真实任务短测。',
'- tests/worldstream_m2 与设计文档：旧精确性测试明确转入离线 A；新增 WSGRR 语义、几何、缓存、准入和 GPU 测试。','',
'## M1 的真实信息与缺失','',
'数据链为 M1 precheck → build_bundle → export_m2_inputs → M2。使用已有 anchor_id、semantic_class、当前相机投影 bbox、观测支持、遮挡假设、关联资格及实际过期 ID。状态摘要只含 anchor_type/class，geometry state_version 仅作来源记录，不把检测抖动当语义突变。世界中心参与已有投影；速度只在 velocity_reliable 时导出。',
'当前帧检测发生在视觉计算后的 commit，故刷新依赖历史检测在当前相机的预测投影。当前检测速度没有可靠认证，物理身份也未独立认证；没有信号灯状态、确认消失和可靠语义突变生产者。缺席不解释为消失。M1 投影无效或过小的对象不能提供可信区域，局部 unknown 优先级和硬时效提供补充，不能据此保证发现所有新对象。','',
'## 公式与默认值','',
'`S = 1.0 * (1 - cosine(mean_valid(x0_W), cached_descriptor_W)) + 0.25 * min(age/8,1) + 0.25 * U`。U 是 dirty 或 unknown 的实际感受野均值再按窗口有效 Token 平均。漂移裁剪至 [0,2]。score_threshold=0.05；ordinary_budget=3/相机；max_reuse_frames=8；region_margin_px=8；movement_threshold_px=16。这些是未校准起点。',
'旧、新 bbox 各膨胀 margin 后取并集，按真实 kernel/stride/padding/dilation 的采样格映射 Token，再映射 EVA 窗口和依赖闭包。中心跨窗口或明显位移必须刷新旧、新区域，不搬移历史输出。同分按年龄，再按 camera-major 索引。必须集合独立于普通预算。描述与年龄只在实际窗口重算时重置；已有 1 秒来源租约常比 8 帧更早生效。','',
'## 连续真实帧：A/B/C/D 与预算','',
f"真实 checkpoint `{study['checkpoint']}`；RTX 3090，原生精度和全形状约束。{study['frames']} 帧，跨度 {study['timestamp_span_s']:.2f} 秒；首帧冷启动，以下统计同场景连续转换。每例预热 1 次、测量 3 次，保留原始样本。",'',
'| 策略 | 精确变化 Token % | 选择 Token % | 窗口更新 % | 窗口复用 % | wall ms | GPU span ms | 原生 wall ms | 原生 GPU ms | MAE | 最大误差 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,r in study['aggregate'].items():
    lines.append(f"| {name} | {100*r['exact_token_update_ratio']:.3f} | {100*r['selected_token_update_ratio']:.3f} | {100*r['window_update_ratio']:.3f} | {100*r['reuse_ratio']:.3f} | {r['wall_ms']:.2f} | {r['gpu_span_ms']:.2f} | {r['native_wall_ms']:.2f} | {r['native_gpu_span_ms']:.2f} | {r['mean_abs']:.6f} | {r['max_abs']:.5f} |")
lines += ['', 'A=精确比较基线；B=漂移、年龄与局部优先级，无对象事件/无数量预算；C=加入对象状态与区域关联，无数量预算；D数字=完整策略及每相机普通预算。B/C 在本数据和初始阈值下常选择全部窗口，不能把 D 的预算收益解释为已证明事件本身降低更新率。',
'逐层窗口执行、复用数以及必须/普通 ID 均在 continuous40/frames.jsonl。Block 0/1 同为 16 窗口，两个 Block 在本模型上的窗口计划相同。精确 Token 比例不是 WSGRR 必须刷新判据；选择 Token 比例按真实未补齐网格计算，因此不必等于窗口比例。',
'视觉时间包含 embedding、M1 来源校验、评分、独立前缀、完整后缀、原子发布和诊断；M1 precheck/read 单列。原生参考不含缓存管理。各策略固定顺序，非随机大规模统计实验。GPU event 为含提交间隙的流跨度，不是活跃 kernel 总时长；host 分段未逐段同步。','',
'## M1 事件覆盖和分段开销','',
f"D3 事件区域记录数：{len(event_regions)}；按标签统计（同一区域可能有多个标签）：`{dict(events)}`。计划中漏刷窗口 {details['planned_event_missed_windows']}；这是对实际计划 ID 的检查，不是独立标注的真实世界事件召回率。完整更新帧自然覆盖所有窗口，但初版长测不会为其逐项登记事件。",'',
f"M1 区域数分布：`{details['M1_regions']}`；漂移分布：`{details['drift']}`；刷新前年龄分布：`{details['ages']}`。",'',
'| 分段 | host 中位数 ms |', '|---|---:|']
for name,r in details['timings'].items():lines.append(f"| {name} | {r['p50']:.3f} |")
lines += ['', '上述 cache_publication_host_ms 包含有限性检查同步时等待前序 CUDA 工作完成的时间，window_and_suffix_host_ms 主要反映提交；不能把前者当成纯复制成本，也不能把这些分段解释为 GPU kernel 耗时。']
publication_path=root/'publication_profile/results.json'
if publication_path.exists():
    profile=json.loads(publication_path.read_text())
    measurements=[r['synchronized_cache_publication'] for r in profile['natural'] if 'synchronized_cache_publication' in r]
    if measurements:
        p=measurements[0]
        lines += ['',f"补充同步后纯缓存发布测量（独立两帧校准，3 次）：wall {p['wall_ms']:.3f} ms，GPU span {p['gpu_span_ms']:.3f} ms。它在前序执行完成后，对真实缓存负载再次执行同一 StateCache.commit，不计入视觉管线时间，不能与不同实验的中位数相加。",'',
            '| Block | 更新窗口 | 增量 wall ms | 原生 wall ms | 增量 GPU ms | 原生 GPU ms |', '|---|---:|---:|---:|---:|---:|']
        costs=json.loads((root/'publication_profile/window_costs_provisional.json').read_text())
        for r in costs['records']:
            lines.append(f"| {r['block']} | {r['updated']} | {r['incremental_wall_ms']:.3f} | {r['full_wall_ms']:.3f} | {r['incremental_gpu_ms']:.3f} | {r['full_gpu_ms']:.3f} |")
lines += ['', 'analysis.json 保留分位数、必须/普通窗口总数、完整刷新原因、关键误差帧。scene-0101 描述包含混凝土搅拌车、锥桶、等红灯、上桥和变道。类别覆盖：'+str(study['coverage']['classes'])+'。行人类别出现不等于独立验证了行人进入事件；遮挡/重现仅为 M1 既有证据。信号灯变化与确认消失事件覆盖不可用。','',
'## 真实下游任务短测','',
f"相同的 {comparison['frames']} 帧及真实语言问题，原生 B0 对照 B2 离线 WSGRR D3（真实 M1、Detection、Map、原生 VLM，禁用 M3 混入）。模型 checkpoint 校验及逐帧结果保存在 native_tasks/ 和 wsgrr_tasks/。",'',
f"检测匹配中心距离均值：{comparison['detection_mean_matched_center_distance_m']} 米；地图匹配线点距离均值：{comparison['map_mean_matched_point_distance_m']} 米；答案完全相同 {comparison['identical_answers']}/{comparison['compared_answers']}。检测先按 score≥0.3 筛选、按类别匈牙利匹配；地图按 score≥0.3 筛选、按正反点序最小平均距离匹配。这些是预测一致性，不是官方 AP。",'',
f"真实参考问答 token F1：原生 {comparison['native_metrics']['quality']['token_f1']:.6f}，候选 {comparison['candidate_metrics']['quality']['token_f1']:.6f}；两组参考答案 exact-match 均为 0/8。不能把两组生成回答相同误写为对参考答案全对。",'',
f"完整任务 request wall 中位数：原生 {comparison['native_metrics']['request_ms']['p50']:.2f} ms，候选 {comparison['candidate_metrics']['request_ms']['p50']:.2f} ms；这是包含 M1 等服务开销的短测，不支持服务加速结论。",'',
'task_comparison.json 保留每帧数量、匹配距离、答案变化及两个 worker 的真实问答质量汇总。短序列不足以报告官方完整验证集 Detection NDS/mAP、Map AP 或完整 VLM 质量结论；独立质量认证仍 PENDING。历史原生检测回放只为长测 M1 提供真实状态，不是候选下游评测。','',
'## 测试与正式准入','']
for name in ('tests_cpu.log','tests_gpu.log'):
    content=(root/name).read_text();last=next((s for s in reversed(content.splitlines()) if ' passed' in s),'NO SUMMARY')
    lines.append(f'- {name}：`{last}`。')
lines += ['- CPU 沙箱的两个跳过项是 CUDA 测试；GPU 完整回归实际执行它们。初始旧路线断言失败已通过明确区分离线精确基线与近似语义修正，最终结果以上述日志为准。',
'- 正式 outputs/final_route/window_costs.json 未修改，records/approved_counts 仍为空；没有执行晋升。',
'- cost_profile/ 使用最终实现额外生成未批准的块级成本候选；全量任务质量、独立特征误差限值及所有覆盖案例的净收益仍是正式准入前提。','',
'## 限制与结论','',
'当前实际窗口更新率确实降低，但没有正的净视觉计算时间收益；评分、来源校验、缓存管理与全形状算子开销超过本次可节省的 Attention。M1 precheck/read 不在表中视觉时间内，另增加完整服务成本。部分窗口更新仍执行全形状 QKV、投影和 MLP，不能声称窗口复用比例对应总 FLOPs 或延迟同比下降。',
'自然关键帧仅约 2 Hz，1 秒来源租约使大量旧窗口周期性必须重算。当前只覆盖一个完整场景与短任务评测，无法量化所有关键场景风险。近似误差已实际记录，未把位级一致当唯一验收标准，也未自动放宽质量容差。',
'结论：当前方案不具备生产采用证据。可作为区域选择研究原型继续分析事件覆盖、关键误差帧和成本分解；在获得可靠净收益和独立完整任务质量前，不推荐正式启用。','',
'设计与复现命令见 docs/worldstream_m2_window.md。长测为初版 WSGRR；最终任务短测与 cost_profile 验证加入状态保留时效/更完整诊断的最终代码，原始长测文件未覆盖。所有执行均使用 OmniDrive Python，避开 CPU 4/5；其他 GPU 进程未被停止。']
(root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
print(root/'experiment_report.md')
