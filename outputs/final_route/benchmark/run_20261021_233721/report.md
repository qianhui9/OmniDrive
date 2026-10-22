# OmniDrive 正式路线实际评测报告

运行目录：`outputs/final_route/benchmark/run_20261021_233721`。本报告可由 `python -m tools.worldstream_m123.benchmark.report --run-dir outputs/final_route/benchmark/run_20261021_233721` 重新生成。

## 完成范围与限制

- 本次真实测量 6339 帧请求、6339 个语言请求（包括标注的重复测量）。
- 官方检测 GT 完整性：`PASS`；受影响帧 0，sample.anns 链接 192041，反向标注 192041。
- GT 链接不一致时不发布官方 mAP/NDS/误差指标；不修改原数据。子集预测一致性不能替代全量官方精度。
- M1 采用原有类别、尺寸、1m 唯一匹配的部分 Oracle 离线评分；不报告未实现的 IDF1/MOTA。
- 生成参数沿用生产设置；以帧 token 确定相同随机种子。模型执行包含原生答案写盘；审计 JSON 序列化和反事实正确性检查在计时之外。
- CUDA 数字是默认流事件跨度，包含 CPU 提交间隙；嵌套阶段不能相加充当 GPU 活跃计算时间。

## 本次环境与权重

- Python：`/home/qh/miniconda3/envs/OmniDrive/bin/python`。
- GPU：`NVIDIA GeForce RTX 3090`。
- Checkpoint SHA256：`6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`。
- 正式配置、成本表、证书及既有 summary/verification 保持不变：`True`。

## 实际任务与请求延迟

| 条件/任务/范围 | 帧请求/独立帧 | 场景 | 重复 | 平均/P50/P95 ms | 帧/s | 有效GT | EM/F1 |
|---|---:|---:|---:|---|---:|---:|---|
| B0/language/attribution/subset | 320/320 | 8 | 1 | 2578.2161/1991.0368/4456.7869 | 0.3879 | 320 | 0.0063/0.4495 |
| B0/planning/main/full | 6019/6019 | 150 | 1 | 3073.3773/3108.8704/3270.4993 | 0.3254 | 0 | null/null |

## 配对比较与模块增量


检测及地图输出哈希完全一致：{'comparisons': 0, 'exact_equal': 0, 'exact_equal_rate': None}。
同问题原生/Auto token 完全一致：{'questions': 0, 'token_comparisons': 0, 'token_equal': 0, 'token_equal_rate': None}。

## M1、M2、M3 实际行为

### B1

NOT_RUN：本次尚无该条件的真实帧记录。
### B2

NOT_RUN：本次尚无该条件的真实帧记录。
### B3

NOT_RUN：本次尚无该条件的真实帧记录。

## 阶段瓶颈（独立阶段跨度，不相加推断端到端）

- B0/language/attribution/subset：decode 1488.9597 ms, eva_native 479.3714 ms, prefill 190.9744 ms, map 28.6824 ms, detection 14.3929 ms
- B0/planning/main/full：decode 2080.4169 ms, eva_native 480.4847 ms, prefill 182.0240 ms, map 28.7212 ms, detection 14.4316 ms

## 四个问题的证据边界

1. 整体精度：只有全量官方评测完成且 GT 完整时才能判断；本次可用范围和配对效率数值如上。
2. M1：上列部分 Oracle 关联与连续性来自真实提交；独立物理认证、对齐分母及标准 MOT 指标分别报告，不互相替代。
3. M2：只将真实跳过的 EVA 相机计入复用。零复用时跳过计算时间为零；净影响使用配对 B2/B1 实测差值。
4. M3：空批准表走正式 Auto 原生回退。语言质量只对有效 GT 评分；净影响使用配对 B3/B2，联合影响使用 B3/B0。
5. 瓶颈：stage_latency.csv 提供本次原生 EVA、检测、地图、prefill/decode 和模块跨度；未取得数据的阶段标记缺失，不使用历史微基准补齐。

## 失败和未完成记录

详细状态见 status.json、plan.json、各实验 failure.log 和 raw_logs；失败实验的部分测量保留。

```json
{
  "experiments": {
    "B0_attribution_language": {
      "status": "COMPLETED",
      "measured_frames": 320,
      "full_validation": false,
      "integrity": "PASS",
      "official_status": "NOT_RUN: subset"
    },
    "B0_full_planning": {
      "status": "FAILED",
      "error": "Traceback (most recent call last):\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/benchmark/worker.py\", line 379, in main\n    run(args)\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/benchmark/worker.py\", line 341, in run\n    write(dest/'metrics.json', summary)\n  File \"/data/qh/phdCode/work4/OmniDrive/tools/worldstream_m123/stateful_opportunity/common.py\", line 27, in write\n    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert)\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/__init__.py\", line 234, in dumps\n    return cls(\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/encoder.py\", line 201, in encode\n    chunks = list(chunks)\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/encoder.py\", line 431, in _iterencode\n    yield from _iterencode_dict(o, _current_indent_level)\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/encoder.py\", line 405, in _iterencode_dict\n    yield from chunks\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/encoder.py\", line 397, in _iterencode_dict\n    yield _floatstr(value)\n  File \"/home/qh/miniconda3/envs/OmniDrive/lib/python3.9/json/encoder.py\", line 239, in floatstr\n    raise ValueError(\nValueError: Out of range float values are not JSON compliant: nan\n",
      "integrity": "NOT_VALIDATED"
    }
  },
  "plan": {
    "full_validation": {
      "status": "PENDING",
      "reason": null,
      "conditions": [
        "B0",
        "B3"
      ],
      "expected_frames": 6019
    },
    "subset_frames": 320,
    "timing_repeats": 3,
    "warmup_frames": 3,
    "timing_frames": 48,
    "jobs": [
      {
        "name": "B0_full_planning",
        "condition": "B0",
        "workload": "planning",
        "limit": 0,
        "repeat": 0,
        "measurement_set": "main"
      },
      {
        "name": "B3_full_planning",
        "condition": "B3",
        "workload": "planning",
        "limit": 0,
        "repeat": 0,
        "measurement_set": "main"
      },
      {
        "name": "B0_attribution_language",
        "condition": "B0",
        "workload": "language",
        "limit": 320,
        "repeat": 0,
        "measurement_set": "attribution"
      },
      {
        "name": "B1_attribution_language",
        "condition": "B1",
        "workload": "language",
        "limit": 320,
        "repeat": 0,
        "measurement_set": "attribution"
      },
      {
        "name": "B2_attribution_language",
        "condition": "B2",
        "workload": "language",
        "limit": 320,
        "repeat": 0,
        "measurement_set": "attribution"
      },
      {
        "name": "B3_attribution_language",
        "condition": "B3",
        "workload": "language",
        "limit": 320,
        "repeat": 0,
        "measurement_set": "attribution"
      },
      {
        "name": "B0_subset_language_r0",
        "condition": "B0",
        "workload": "language",
        "limit": 48,
        "repeat": 0,
        "measurement_set": "efficiency"
      },
      {
        "name": "B1_subset_language_r0",
        "condition": "B1",
        "workload": "language",
        "limit": 48,
        "repeat": 0,
        "measurement_set": "efficiency"
      },
      {
        "name": "B2_subset_language_r0",
        "condition": "B2",
        "workload": "language",
        "limit": 48,
        "repeat": 0,
        "measurement_set": "efficiency"
      },
      {
        "name": "B3_subset_language_r0",
        "condition": "B3",
        "workload": "language",
        "limit": 48,
        "repeat": 0,
        "measurement_set": "efficiency"
      },
      {
        "name": "B0_subset_language_r1",
        "condition": "B0",
        "workload": "language",
        "limit": 48,
        "repeat": 1,
        "measurement_set": "efficiency"
      },
      {
        "name": "B1_subset_language_r1",
        "condition": "B1",
        "workload": "language",
        "limit": 48,
        "repeat": 1,
        "measurement_set": "efficiency"
      },
      {
        "name": "B2_subset_language_r1",
        "condition": "B2",
        "workload": "language",
        "limit": 48,
        "repeat": 1,
        "measurement_set": "efficiency"
      },
      {
        "name": "B3_subset_language_r1",
        "condition": "B3",
        "workload": "language",
        "limit": 48,
        "repeat": 1,
        "measurement_set": "efficiency"
      },
      {
        "name": "B0_subset_language_r2",
        "condition": "B0",
        "workload": "language",
        "limit": 48,
        "repeat": 2,
        "measurement_set": "efficiency"
      },
      {
        "name": "B1_subset_language_r2",
        "condition": "B1",
        "workload": "language",
        "limit": 48,
        "repeat": 2,
        "measurement_set": "efficiency"
      },
      {
        "name": "B2_subset_language_r2",
        "condition": "B2",
        "workload": "language",
        "limit": 48,
        "repeat": 2,
        "measurement_set": "efficiency"
      },
      {
        "name": "B3_subset_language_r2",
        "condition": "B3",
        "workload": "language",
        "limit": 48,
        "repeat": 2,
        "measurement_set": "efficiency"
      }
    ],
    "task_definition": "Real Conv/VQA question, round-robin question index, native production sampling",
    "caveat": "Sequential first scenes sorted by token; representative coverage is described, not a random statistical sample"
  },
  "protected_unchanged": {
    "projects/configs/OmniDrive/mask_eva_lane_det_vlm.py": true,
    "projects/configs/OmniDrive/worldstream_m123_auto.py": true,
    "configs/worldstream_m1_final.yaml": true,
    "outputs/final_route/camera_costs.json": true,
    "outputs/final_route/m3_cost_table.json": true,
    "outputs/final_route/m3_quality_certificate.json": true,
    "outputs/final_route/summary.json": true,
    "outputs/final_route/verification.json": true
  },
  "measured_frames": 6339,
  "measured_questions": 6339,
  "complete": false,
  "official_detection_available": true
}
```


## 用户停止记录

2026-10-22T09:18:24.178729+08:00：用户明确要求停止评测。本次控制器、GPU 推理及收尾进程均已停止，不会自动继续；全部已有预测与日志保留。此前表格为停止前快照，当前完成范围以 status.json 和 stop_requested.json 为准。
