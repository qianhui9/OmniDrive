# M2 Transformer 窗口级增量执行：实现与验证

正式入口已从 camera Auto 改为 window Auto。实现能够执行 dirty-window gather → 原 EVA Attention/MLP → scatter → 全图重建；仅复用全局注意力之前的独立窗口前缀。当前生产窗口质量批准为空，Auto 保守回退；本报告没有声称真实连续流已获得加速。

## 实现范围

- 当前模型：640×640 图像、16×16 patches、40×40 tokens、16×16 token 窗口。每相机 9 窗口，六相机每 block 54；可缓存 block 0、1，共 108 个 block-window。block 2 起的全局依赖部分继续原生执行，patch embedding 完整执行。
- 修复 padding 的缓存语义；缓存克隆拥有激活并原子发布，验证实际预算及张量修改；拒绝训练/梯度执行，修复零更新单窗口的缓存别名。
- 成本键覆盖内部 norm/Flash/数值策略；非法成本和伪布尔批准不能准入。窗口成本与旧相机成本分开，未转移批准。
- M1→M2 legacy 窗口适配补齐绑定事务时间。计数使用实际 Attention batch hooks，并与 prefix 的 executed/reused windows 核对；避免包装 EVA.forward 破坏严格拓扑识别。
- 普通 M1 配置保持原值，legacy unknown/失效掩码仍保守重算；切换粒度不会自动增加候选。已有 source-linked 快照/来源路径有连续 precheck → execution → commit 测试，临时诊断覆盖方式见文档。

## GPU 受控视觉微基准

真实 checkpoint `ckpts/iter_10548.pth`，通过原 EVA strict load；真实 nuScenes 图像使用当前图像预处理。FP32 权重配原 FlashAttention 路径，RTX 3090。每例预热后 3 次测量。以下是 backbone-only 的受控输入实验，独立统计缓存捕获后的请求；不计入自然流命中率，也不代表 Detection/Map/语言端到端加速。

| 输入 | 跳过 block-window | 输出最大绝对误差 | 位级相同 | 通过 1e-5 atol/rtol | 原生 wall ms | 增量 wall ms |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| identical | 108 | 0 | True | True | 477.232 | 429.828 |
| one_window | 106 | 0.40311813 | False | False | 478.855 | 433.241 |
| padded_edge_window | 106 | 0.3570137 | False | False | 477.381 | 431.624 |
| one_window_per_camera | 96 | 0.7398262 | False | False | 479.098 | 437.935 |

自然连续 4 帧：实际跳过 0 个 block-window，全部输出位级等于原生。

相同输入通过位级检查；局部 packed 重算存在超过严格容差的差异。本次未将其判定为通过，也未放宽容差。形状变化可能影响浮点路径，但尚未通过逐 kernel 对照确定所有差异来源。`exact` 只说明输入/依赖闭包严格比较，不等于任务质量认证。Detection/Map/语言的局部复用质量尚未评估。`window_costs_provisional.json` 全部 `quality_certified=false`，生产表 `approved_counts=[]`。

原始数值、逐帧诊断和 3 次 wall/GPU stream span 样本在 `results.json`；GPU event span 不等于各 kernel active time 之和。

## 完整模型短测

- B3 language：预热 1 帧，测量 2 帧、2 个真实问题。状态 COMPLETED，M1 场景完整性 PASS。
- checkpoint 校验：PASS；期望 1573 个键，missing/unexpected/shape mismatches 均为空；仅忽略已声明的非持久 rotary caches。
- M2 实际跳过 0 个 block-window，实际 prefix Attention batch 合计 216，各帧均与原生执行数匹配。真实连续输入被 cheap gate 拒绝；当前质量空表也不授权局部执行。
- M3 回退原生生成，检测和地图路径照常执行。2 帧不足以报告官方全量任务精度；无 B0 配对，不能据此估计净加速。
- 短测逐帧数据、checkpoint 校验和调用配置保存在 `formal_auto_smoke/`。随后补充的 legacy 时间戳传递另由新增回归测试覆盖，完整修改版本见 implementation_manifest.json。

## 回归与保护

- 全套测试：`951 passed, 2 skipped, 19 warnings in 109.90s (0:01:49)`。
- 最新针对性测试：`35 passed, 19 warnings in 1.96s`。
- GPU 专项：`2 passed, 19 warnings in 2.73s`。
- 路线验证另存在 `route_verification/`，引用本次 tests.log。基线、M1 核心、原相机/M3 校准表与历史受保护文件经原清单验证；旧清理审计未覆盖。
- 所有 Python 执行使用 OmniDrive 环境，CPU 亲和性为 0-3,6-31。此前按用户要求停止的长时评测没有恢复，也未停止其他 GPU 任务。

## 使用

正式入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`。

复现短程窗口验证、诊断 source-linked 接口和 Auto 准入限制见 `docs/worldstream_m2_window.md`。当前下一项实质工作是定位局部 packed 数值差异并验证下游任务质量；在通过前保持 Auto 批准为空。
