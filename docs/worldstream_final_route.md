# OmniDrive 单一研究主线

研究目标：**World-Referenced Stateful VLM Serving with Dependency-Aligned GPU Execution and Efficient Semantic Consumption**。

唯一正式入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`，直接继承原始 `mask_eva_lane_det_vlm.py`，保持权重、EVA、Detection/Map、LoRA、Tokenizer 和原始 Decode。其他 worldstream 模型配置及替代执行器已退役。原始 OmniDrive 配置保留作独立参考基线。

执行顺序固定为：M1 Precheck → M2 窗口级视觉执行 → 原始 Detection/Map → M1 Commit → M3 可信历史增强与精确 Prefill → 原始 Decode。M2 与 M3 的历史快照均在当前 Commit 之前冻结；M3 在 Commit 后消费冻结副本，不读取新提交状态冒充历史。

M1 只有普通 A 的几何门控、一次全局一对一分配与实际观测更新。正式配置开启 `temporal_visibility`、`local_support`、`source_linked_state`。场景、时间、epoch、执行签名、来源摘要、过期及原子事务检查保留。DQTrack、Shadow/Hint、身份消歧、深度搜索及替代身份历史策略不再提供配置开关。普通关联及观测支持不签发物理身份或几何真值认证。

M2 只有 `window/source_linked/maintain`。完整像素比较与 Patch 感受野、窗口闭包决定更新集；M1 dirty hints 只能增加更新。缓存未命中、来源失效、强制刷新或准入不足，都将同一个 WindowExecutor 的更新集扩大至全部窗口。没有 camera/conditional 缓存执行器，也没有 Exact 绕过模式。Patch Embedding 与 block 2 起的全局依赖部分完整执行。

局部窗口数值差异已定位至打包改变 GEMM 形状，FP32 误差经 FP16 FlashAttention 放大。当前唯一执行器保留原生 QKV、输出投影、归一化与 MLP 的完整形状，仅增量执行独立窗口 Attention；零更新 block 可整体复用。真实 GPU 单窗口、边缘、多相机验证均达到位级一致。统计严格区分复用输出窗口、跳过 Attention 与仍执行的点积工作量，不把局部复用声称为整 block 全部省去。窗口成本表单独维护；旧相机证据不迁移，未获数值、任务质量与成本准入时全量更新。

M3 只有完整当前视觉载体、经过来源与任务质量门控的历史文本补充，以及精确扩展输入前缀/KV 复用。历史不可用于替代当前视觉；缺失、不适用或未获独立质量批准时保留原问题与完整载体，不添加历史文本。前缀未命中或未获准入时 `reuse_length=0`，继续使用同一个 PrefillExecutor；不调用原生 `lm_head.generate()` 作为运行时回退。不支持的模型/生成契约显式报错。原生 generate 仅在关闭扩展的独立基线和等价性验证工具中作为参考。

完整 Prefill 在真实模型与相同种子下通过 token、logits、RNG 对照；完整相同前缀也通过。受控局部前缀的 token 与 RNG 一致，但 logits 超过严格容差，因此当前不授予正式局部前缀复用。历史文本质量与 KV 匹配分别认证；两者不能相互代替。当前部署批准表保持为空。

本次删除计划、逐文件调用方、删除记录、测试及真实测量保存在 `outputs/single_route_convergence_20261022/`。旧结果及 `outputs/m3_state_guided_verified_decode/` 保留作历史证据，不作为当前路线的测量值。完整验证集任务精度与历史文本增强的独立质量，只有实际测量并通过后才可声明。

运行和数值验证命令见 `docs/worldstream_benchmark.md` 与 `docs/worldstream_m2_window.md`。
