# OmniDrive 正式路线实际评测报告

运行目录：`outputs/single_route_convergence_20261022/service`。本报告可由 `python -m tools.worldstream_m123.benchmark.report --run-dir outputs/single_route_convergence_20261022/service` 重新生成。

## 完成范围与限制

- 本次真实测量 24 帧请求、24 个语言请求（包括标注的重复测量）。
- 官方检测 GT 完整性：`NOT_AVAILABLE`；受影响帧 None，sample.anns 链接 None，反向标注 None。
- GT 链接不一致时不发布官方 mAP/NDS/误差指标；不修改原数据。子集预测一致性不能替代全量官方精度。
- M1 采用原有类别、尺寸、1m 唯一匹配的部分 Oracle 离线评分；不报告未实现的 IDF1/MOTA。
- 生成参数沿用生产设置；以帧 token 确定相同随机种子。模型执行包含原生答案写盘；审计 JSON 序列化和反事实正确性检查在计时之外。
- CUDA 数字是默认流事件跨度，包含 CPU 提交间隙；嵌套阶段不能相加充当 GPU 活跃计算时间。

## 本次环境与权重

- Python：`None`。
- GPU：`None`。
- Checkpoint SHA256：`None`。
- 正式配置、成本表、证书及既有 summary/verification 保持不变：`NOT_VERIFIED`。

## 实际任务与请求延迟

| 条件/任务/范围 | 帧请求/独立帧 | 场景 | 重复 | 平均/P50/P95 ms | 帧/s | 有效GT | EM/F1 |
|---|---:|---:|---:|---|---:|---:|---|
| B0/language/smoke/subset | 12/12 | 1 | 1 | 3077.5405/1757.5645/9235.8097 | 0.3249 | 12 | 0.0000/0.3599 |
| B3/language/smoke/subset | 12/12 | 1 | 1 | 3313.5625/2130.5948/9442.5290 | 0.3018 | 12 | 0.0000/0.3599 |

## 全量官方任务精度

| 指标 | B0 | B3 | B3-B0 | 状态 |
|---|---:|---:|---:|---|
| mAP | null | null | null | NOT_AVAILABLE |
| NDS | null | null | null | NOT_AVAILABLE |
| mATE | null | null | null | NOT_AVAILABLE |
| mASE | null | null | null | NOT_AVAILABLE |
| mAOE | null | null | null | NOT_AVAILABLE |
| mAVE | null | null | null | NOT_AVAILABLE |
| mAAE | null | null | null | NOT_AVAILABLE |

mAP/NDS 的百分点变化为上述原始差值 ×100。地图使用现有 OpenLaneV2 指标，未实现的 IoU/mIoU 不填数。

地图官方结果：`{"B0": null, "B3": null, "delta": null, "status": "NOT_RUN: no completed full-split map evaluation"}`。

## 配对比较与模块增量

- language/smoke/B0->B3：12 对请求，增量 236.0220 ms，延迟变化 -7.6692%，加速比 0.9288。场景置信区间见 JSON。

检测及地图输出哈希完全一致：{'comparisons': 12, 'exact_equal': 12, 'exact_equal_rate': 1.0}。
同问题原生/Auto token 完全一致：{'questions': 12, 'token_comparisons': 12, 'token_equal': 12, 'token_equal_rate': 1.0}。

## M1、M2、M3 实际行为

### B1

NOT_RUN：本次尚无该条件的真实帧记录。
### B2

NOT_RUN：本次尚无该条件的真实帧记录。
### B3

- M1 事件计数：`{'committed': 1140, 'associated': 961, 'created': 179, 'expired': 66, 'valid_history': 45, 'history_candidates': 1036, 'transactions_committed': 12, 'physical_candidates': 0, 'precheck_physical_evaluations': 1082, 'precheck_physical_producer_available': 0}`。有效历史状态比例 0.0434。
- 物理认证：Independent physical producer/evidence missing in precheck; post-perception certification had no qualified candidates；普通关联从不计入物理认证。
- B3_language_final 关联精度 1.0000，错误率 0.0000，有效召回 0.9811，覆盖 0.9376；分子/分母 `{'frames': 12, 'candidates': 1025, 'eligible': 159, 'accepted': 961, 'scored': 156, 'correct': 156, 'wrong': 0, 'unknown': 805}`；连续性 `{'predicted_track_fragments': 1, 'continuous_identity_switches': 3, 'scored_visible': 188, 'visible_known_missed': 253, 'ambiguous_visible': 112, 'bridged_visible_gaps': 1}`。
- M2 窗口级：实际复用 0/1296 个输出窗口，执行 Attention 1296 个，实际 Attention batch 1296，审计 PASS；点积窗口工作量 1296。拒绝原因 `{'m1_full_refresh': 2, 'no_valid_cache': 10}`。
- M3：请求 12，候选 11，准入 0，命中 0，原生生成 0，实际优化 0；原因 `{'short_or_missing_prefix': 1, 'prefix_evidence_unavailable': 11}`。调度额外 host wall 平均 14.5911 ms。

## 阶段瓶颈（独立阶段跨度，不相加推断端到端）

- B0/language/smoke/subset：decode 1965.5402 ms, eva_native 469.7152 ms, prefill 178.7317 ms, map 28.4953 ms, detection 14.1710 ms
- B3/language/smoke/subset：decode 1969.9177 ms, prefill 175.8065 ms, map 28.4330 ms, detection 14.2294 ms

## 四个问题的证据边界

1. 整体精度：只有全量官方评测完成且 GT 完整时才能判断；本次可用范围和配对效率数值如上。
2. M1：上列部分 Oracle 关联与连续性来自真实提交；独立物理认证、对齐分母及标准 MOT 指标分别报告，不互相替代。
3. M2：按真实 Attention 窗口和复用输出窗口计数，完整点积工作量另列。零复用不宣称节省。
4. M3：空批准表使用相同 PrefillExecutor 的完整 Prefill。历史增强与前缀复用独立认证，语言质量仅对有效 GT 评分。
5. 瓶颈：stage_latency.csv 提供本次原生 EVA、检测、地图、prefill/decode 和模块跨度；未取得数据的阶段标记缺失，不使用历史微基准补齐。

## 失败和未完成记录

详细状态见 status.json、plan.json、各实验 failure.log 和 raw_logs；失败实验的部分测量保留。

```json
{
  "experiments": {
    "B0_language": {
      "status": "COMPLETED",
      "measured_frames": 12,
      "full_validation": false,
      "integrity": "PASS",
      "official_status": "NOT_RUN: subset"
    },
    "B3_language": {
      "status": "FAILED",
      "error": "Traceback (most recent call last):\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/benchmark/worker.py\", line 391, in main\n    run(args)\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/benchmark/worker.py\", line 248, in run\n    frame, anchors, audit = m1_record(model, observer.commits[-1])\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/benchmark/observe.py\", line 204, in m1_record\n    commit=copy.deepcopy(manager.last_commit_diagnostics), committed_state=manager.export_state_audit()))\n  File \"/data/qh/phdCode/work4/OmniDrive/worldstream/m1/state_manager.py\", line 576, in export_state_audit\n    return export_state_audit(self)\n  File \"/data/qh/phdCode/work4/OmniDrive/worldstream/m1/provenance.py\", line 63, in export_state_audit\n    anchors=rows, depth_history={cid:{k:v for k,v in h.items() if k not in ('pixels','depths','uncertainty_m')} | {'points':len(h['pixels'])} for cid,h in manager._depth_history.items() if h is not None},\nAttributeError: 'WorldStateManager' object has no attribute '_depth_history'\n",
      "integrity": "NOT_VALIDATED"
    },
    "B3_language_final": {
      "status": "COMPLETED",
      "measured_frames": 12,
      "full_validation": false,
      "integrity": "PASS",
      "official_status": "NOT_RUN: subset"
    }
  },
  "plan": {},
  "protected_unchanged": {},
  "measured_frames": 24,
  "measured_questions": 24,
  "complete": false,
  "official_detection_available": false
}
```
