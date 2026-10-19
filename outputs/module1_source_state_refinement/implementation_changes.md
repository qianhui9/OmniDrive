# 实现与 profiling 审计

冻结对照 HEAD：a4d27405842a6a7de0e839dd830ef2960fc9a937。开始时工作区干净。上一轮 outputs/module1_state_interface_rearchitecture 原始结果全部保留。baseline/HEAD.tar 和 manifest.json 保存冻结源码；本轮所有新数据仅写入当前目录。

## 接口与安全行为

- m2/m1_adapter.py 新增逐相机来源/Lease/Dirty 强制更新信息，并核对真实缓存的 Producer Frame/Version/Timestamp。场景、签名、Cache Publication Version、顺序、非法事务及 Full Refresh 仍触发全局重算。Window 路径保留原耦合行为。
- m2/runtime.py 不再因为任一相机 Dirty 就无条件 reset 全部 Camera Cache；传入逐相机 force_recompute_cameras。旧全局失效策略由 camera_selective_invalidation=False 保留作同边界对照。
- camera_executor.py 对未受影响相机执行完整有限输入 Exact 比较，强制相机与实际输入变化取并集，再应用既有数量、当前签名、下游质量和成本门。Auto 没有合格的 changed_cameras>0 记录时执行完整 Backbone。六元素 CPU 决策信息同时用于执行与审计，不再为执行来源重复回传同一个 changed 向量；未新增 cuda.synchronize。
- camera_feature_cache.py 继续先构建所有 owned Tensor/元数据后原子发布；每个条目分别保留不可变 Producer Version 与当前 cache_publication_version。中途 OOM 的回归检查旧批次仍完整。
- dependency_planner.py 只增加依赖足迹描述，不改变执行规划。按真实 Conv kernel/stride/padding/dilation 和前序非移位 Window 闭包回溯输入范围；虚拟 Padding 不计入观测人口，Global/Residual 边界仍终止可缓存前缀。
- m1/computation_state.py 输出 dependency_pixel_count、支持数量/比例、冲突数量、未观察比例和 partial_observation_only。部分支持不能声明完整 observation_valid；多摄像头 Head/未知依赖保守标为 scope_fragment。来源帧、实际计算帧和当前观测帧仍分开。
- observation_refinement.py 单独恢复通过双向、几何、局部支持及中心光度检查的 LK 中心；与 Dense 同像素去重，来源坐标冲突时撤销资格，动态/遮挡排除继续生效。不做 Patch/Box 填充。残差 remap 分成最多 16384 点，修复扩展回放发现的 OpenCV 32767 维度上限。
- Change-First 只由在线局部变化及已跟踪光度风险产生拒绝种子，半径沿用已测局部支持足迹。它不使用 Annotation、不声明前景所有权、不把 Unknown 变为 Candidate。严格 refresh_required_mask/exact_invalidation_mask 原语义保留。

## 性能修改依据

先执行真实 nuScenes 基线 cProfile，原始文件在 profile_output/baseline.prof，函数表在 baseline_hotspots.json。主要热点为持续身份维护内的重复变换校验、灰度转换；Bundle 编码/拷贝不是主要瓶颈。因此保留 StateStore Fork、Commit 历史、身份算法与事务验证，不用跳过维护制造收益。

measurement_cache.py 提供单次 Precheck/Commit 内的测量缓存：刚体变换按精确 shape/dtype/bytes 缓存；矩阵修改后键立即变化。灰度只在当前只读测量调用范围复用，退出或异常后释放。dense_support 复用同次 Sparse 检测的灰度图。CPU 输入仍防御复制；CUDA .cpu() 已拥有独立主机存储，numpy_copy 复用该新存储，减少一次重复主机复制。真实 M1 配对消融为 change_first 对 optimized；独立辅助函数微基准在 profile_output/optimization_microbench.json，不能当作模型加速。

## 采样代码改动清单

| 文件 | 类型 | 新增采样 |
|---|---|---|
| worldstream/m1/state_manager.py | modified | Fork、Bundle、Dense 阶段、Commit 历史复制和 Receipt 墙钟 |
| worldstream/m1/computation_state.py | modified | Mask 编码及严格字段复制墙钟 |
| worldstream/m1/omnidrive_adapter.py | modified | GPU→CPU 调用、字节与传输墙钟 |
| projects/mmdet3d_plugin/models/detectors/petr3d.py | modified | 输入/输出传输字节回写 |
| worldstream/m2/runtime.py | modified | M1→M2 Adapter 墙钟 |
| worldstream/m2/camera_executor.py | modified | 验证与执行入队墙钟、逐相机 Exact/强制更新决策 |
| tools/worldstream_m1/source_state_refinement_replay.py | created | 扩大回放、逐阶段时间、支持范围、中心抽样和离线评分 |
| tools/worldstream_m1/source_state_refinement_centre_diagnostics.py | created | 邻近稀疏参考距离/对应诊断，不改变主要评分人口 |
| tools/worldstream_m1/source_state_refinement_spatial.py | created | 独立小型真实观测回放，保存实际空间 Mask |
| tools/worldstream_m2/source_state_refinement_integration.py | created | 原模型同边界五臂比较、EVA CUDA Event、Memory/Head/LLM 质量 |
| tools/worldstream_m2/source_state_refinement_cuda_batch.py | created | GPU 空闲等待、核见证、回归和实验生命周期 |
| tools/worldstream_m1/source_state_refinement_report.py | created | 从原始记录推导 JSON、六类图表和报告 |

这些采样只用于可审查的诊断，未引入新的状态生产器或调度器；需要精简部署采样时可按此清单移除。

## 实验性 Lease 分离

实际 EVAViT.forward(x) 仅接收图像，依赖 Patch/位置编码、固定参数和 eval 状态，不读取时间/Pose 或 Head Memory。experimental_exact_lease_separation 默认为 False，且仅允许 Source-Linked Camera 边界；实验仅解除旧计算来源的 Observation Horizon 上限，不解除当前物理 Lease、显式 Dirty、非法时间/来源、签名/版本或周期刷新。当前 Head、Pose、Memory、LLM 仍重新执行。未开放未合格的非零局部更新。

## 研究边界

普通 Observation Link 不是物理真值或 Transformer 等价证明。全部数据仍是 exploratory_reused_population。新增实测中心可缺少可评分 LiDAR 参考；邻近参考也可能来自另一表面，不能据此伪造 Correct/False。最终 Critical ROI 失败、候选错误与 M1 冻结判断以本轮原始记录为准。独立认证保持 0/null 与 No-Go。

失败保留：attempts/pre_final_instrumentation_opencv_limit 为大人口回放中的 remap 失败；gpu_validation 为协议尚未就绪的启动失败；最终 CUDA 目录为 gpu_validation_v2。执行回放工具的归档哈希见 replay_harness_version.json；当前工具只额外加入计算结束后的 ready 文件，数值逻辑不变。

## 最终来源与范围修订

已知 observation_descriptor 按真实历史输入裁剪统计，来源中心只栅格化去重一次，用积分图查询每个裁剪的支持与冲突；没有给 Box 填充候选。无法确定的多相机/时序 Head 依赖使用 null，单相机片段单列。M1 Feature Slot 超时后，实际 M2 缓存仍在时由已提交回执输出真实 backbone 来源，保持 Producer Version，不创建虚构 FeatureRecord，也不延长物理 Lease。最终真实模型重复输入复查覆盖该路径 24 条来源记录。

新增 M1Config 字段置于全部旧字段之后，保留位置参数兼容性。Camera 缓存发布失败时，新计算输出不再指向尚未存在的下一次发布版本；缓存中途 OOM 保持旧批次原子性。旧源码归档、差异与发布路径审计文件均保留。M2 增加实际 candidate_cameras 计数；长序列在此计数加入前运行，缺失处保持 null，最终短序列实测计数单列。

最终源码另运行 96 帧七方法回放、596/596 CUDA 回归，以及自然和受控重复输入共 40 次完整任务核验。长序列主比较为自然 192 次、重复 384 次质量核验，其源码快照与最终版的差异如实保留。最终版源码精确匹配与所有核验结果见 original_model_parity.json、artifact_validation.json。M1 冻结判定仍为 No-Go，不开放未达标研究候选的跳算。

最终生命周期与交付工具：source_state_refinement_finalize.py 顺序执行最终回放、CUDA 回归和模型复查；source_state_refinement_current_recheck.py 共享一次模型加载运行两种输入人口；source_state_refinement_audit.py 核对当前源码哈希、来源、44 个受保护源码文件、必需交付物和六幅图。
