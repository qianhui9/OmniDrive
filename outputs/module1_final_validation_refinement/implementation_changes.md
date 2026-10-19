# 本轮实现与采样改动清单

冻结实际 HEAD 3113df211a64a2201c43f3926a6ee422e2db9872；起始工作区干净。baseline/source.tar.gz 保存修改前全部 worldstream 源码，baseline/manifest.json 保存精确 SHA256。上一轮与其他历史输出只读。

## 诊断与单一风险机制

先运行完整 96 帧当前 Optimized 的诊断和 cProfile，14 个失败逐一保存原始测量、掩码与离线 GT。10 个失败没有重叠历史投影，3 个有投影，1 个确有原始 changed 被动态排除清除。几何投影重叠不证明 GT 实例身份。

motion_risk_revocation 默认为 False，最终独立配置启用。仅对具有已 Commit 来源、匹配 Scene/执行签名、相机观测时间、有限历史速度的对象构建有限投影 Hull。Hull 只是可能占用区域，不是前景 Mask；只有实测灰度变化才成为 Dirty；旧源占用冲突落在 Hull 外和失败对应中心仅撤销为 Unknown。没有历史或投影跨越相机平面时不编造范围。除已有局部 Change-First 外，没有扩大膨胀半径，没有整图常驻 Dirty，没有使用当前尚未执行的 Detection。

Precheck 动态排除前的 changed 单独进入风险通道，兼容的原 changed/stable/Identity 输入不改。Observation Link 的真实坐标、残差、来源继续输出；observation_link_valid 与 reuse_candidate_eligible 独立，不因撤销候选删掉物理联系。所有 M2 Exact、版本、质量、成本及强制刷新逻辑保持不变。

历史图像由原 Store 保存为灰度；新风险函数按相同空间尺寸和真实灰度测量比较，未扩大 Store 或伪造 RGB。灰度变化是拒绝证据，不替代 M2 完整 RGB 输入等价验证。该真实存储格式已加入测试。

## Profiling 后的测量复用

baseline.prof 的 96 帧结果显示 temporal_identity_pairs 累计 28.53s，3779 次身份 LK 与 patch 支持测量。身份生产器保留原实现。每次 Precheck/Commit 的只读作用域内，复用相同灰度数组的精确 float32 值与 Sobel 梯度；强引用持有数组 owner，退出/异常后 ContextVar 释放。相机线程各自独立作用域，避免跨线程可变共享。仅在 measurement_reuse=True 时缓存数组；无光流/身份决策/跨帧认证缓存，不改关联边、阈值或匹配算法。

## 离线物理质量协议

候选抽样种子 1907、每层最多 32、各层人口/抽样权重在评分前固定。按实际 Link origin 与离线关键 ROI 上下文分层；background_context 指关键 ROI 外上下文，不是经过证明的静态背景。随机样本与精确注释 LiDAR 像素交集 census 分开，不用更容易评分的交集替代原分母。只有合法双时刻可见的精确投影像素及实例刚体参考评分；没有合法参考保持 Unscorable，不以邻近 2px 点为真值，不做多帧点云简单叠加。保留原 Sparse FAR；物理链接误差与无需刷新的静止假设是不同问题。原始回放 log 回退字段存 Scene ID，派生报告用 nuScenes scene.json 解析真正 Log ID，不将帧或 Scene 数当独立 Log。

## 源码与诊断清单

| 文件 | 变更 | 采样/用途 |
|---|---|---|
| worldstream/m1/observation_refinement.py | modified | 有限历史运动占用、风险理由、撤销范围 |
| worldstream/m1/computation_state.py | modified | 链接有效/候选资格分离及撤销量 |
| worldstream/m1/state_types.py | modified | 两个拒绝通道，可选兼容字段 |
| worldstream/m1/config.py | modified | 两个默认关闭的新标志，旧位置参数兼容 |
| worldstream/m1/measurement_cache.py | modified | 调用内精确灰度/Sobel 重用 |
| worldstream/m1/geometry_support.py | modified | 重用实际图像测量，数学相同 |
| worldstream/m1/dense_support.py | modified | 同次 Sparse/Dense 图像值/梯度共用 |
| worldstream/m1/state_manager.py | modified | 接入拒绝通道和每线程测量作用域 |
| tools/worldstream_m1/final_validation_diagnose.py | created | 修改前 96 帧 cProfile、14 个原始证据 Overlay |
| tools/worldstream_m1/final_validation_replay.py | created | 五方法 96 帧配对、维护开销、状态决策摘要 |
| tools/worldstream_m1/final_validation_sampling.py | created | 完整分层分母、固定抽样、独立参考交集评分 |
| tools/worldstream_m1/final_validation_overlays.py | created | 最终/旧候选与风险 Hull 的离线图 |
| tools/worldstream_m1/final_validation_report.py | created | 统计、四类图、冻结判断 |
| tools/worldstream_m2/final_validation_integration.py | created | 新旧/当前/最终同边界模型比较和统一质量采样 |
| tools/worldstream_m2/final_validation_cuda.py | created | GPU 空闲等待、核见证、新回归与完整模型两个人口 |

这些诊断采样可独立移除，不是模型或生产身份逻辑的一部分。工具行号以当前源码为准，完整源码清单在 delivery_source_manifest.json。

失败/过渡尝试保留在 attempts/：最初 Smoke 的 Geometry JSON 序列化问题，以及旧源灰度/当前 RGB 格式检查不符的未完成回放均未当成最终证据。最终回放、最终 GPU 执行各自保存精确源码哈希。

## 96 帧配对后的工程选择

Motion-only 与 final 没有修复 14 个旧失败，也没有拒绝更多可评分错误；只撤销 118 个候选像素，停止继续叠加运动模型。工程推荐 worldstream_m1_engineering_frozen.yaml 关闭 motion_risk_revocation，仅开启独立获益的 measurement_reuse；模型配置 worldstream_m2_final_validation.py 指向该工程配置。风险开启的 worldstream_m1_final_validation.yaml 保留作研究消融。完整 GPU 比较包含 measurement_only 推荐配置与实验完整配置，质量门相同。

measurement-only 的配对 M1 P50/P95 约为 404.03/1024.29ms，当前 Optimized 为 412.13/1122.03ms；Association P50/P95 约为 158.64/807.87ms 对 174.51/913.64ms。实验完整风险方法为 413.85/1042.92ms，不用其 P95 改善宣称 P50 或端到端加速。

## 性能采样定位

| 文件 | 变更 | 内容 | 当前行号 |
|---|---|---|---:|
| worldstream/m1/observation_refinement.py | modified | def motion_risk_masks | 6 |
| worldstream/m1/computation_state.py | modified | motion_diagnostics={} | 116 |
| worldstream/m1/measurement_cache.py | modified | def gray_measurement | 17 |
| worldstream/m1/state_manager.py | modified | evidence.diagnostics['motion_risk'] | 236 |
| worldstream/m1/geometry_support.py | modified | from .measurement_cache import gray_measurement | 68 |
| worldstream/m1/dense_support.py | modified | from .measurement_cache import gray_measurement | 94 |
| tools/worldstream_m1/final_validation_audit.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m1/final_validation_diagnose.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m1/final_validation_overlays.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m1/final_validation_replay.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m1/final_validation_report.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m1/final_validation_sampling.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m2/final_validation_cuda.py | created | 独立诊断/实验/验收工具 | — |
| tools/worldstream_m2/final_validation_integration.py | created | 独立诊断/实验/验收工具 | — |

身份生产器的密码学协议摘要绑定完整配置与源码；新标志或依赖源码改变会产生新的诊断签名，不沿用旧物理认证。96 帧等价核验针对关联、Object ID、来源帧/版本、几何和关联历史决策摘要；不把不同配置下的签名字符串宣称逐位相同。M2 的实际 Producer/Cache 来源另由完整模型审计核验。

96 帧 cProfile 中 associate 累计 36.74ms，192 次 legal_assignment 合计 4.21ms，association_ambiguities 合计 10.04ms。Commit 记录的 association_ms 含身份候选视觉测量，不能把约 914ms 的 P95 误称为 Hungarian 求解长尾。因此没有改匹配算法或收窄关联人口，而是优化重复的图像测量。完整分帧人口和实际函数计时见 runtime_hotspot_analysis.json。
