# 性能验收状态：未执行准入实验

ZeroKV 关键语义质量未通过，最终目标域核心覆盖 0/42。遵守先质量后 ExactKV 的条件，未执行 C、固定相同参考 Tokens/Decode 步数、三轮交替单问题/三问题实验、场景配对 Bootstrap CI；这些指标均未建立，不能报告端到端加速或 Prefill CUDA 减少。

B 的 144 请求实际跨帧 Exact Visual KV Tokens=0、实际跳过 Prefill Tokens=0、跨帧复用请求=0。资格回放 18 次 Anchor 命中只说明核心内容连续，不代表 18×128 个 Tokens 已经复用。

## 已记录的描述性开销

资格 CPU 回放的 selector 均值 0.006640 ms、Anchor update 0.028937 ms；只包含真实候选上的选择/生命周期，不包含模型或完整候选提取，不能作总系统开销。

B 自然质量单轮中每帧（3 问）Composer host wall 均值 0.551758 ms，核心提取+Anchor 管理合计 host wall 0.455803 ms，M3 Runtime host wall 2189.902 ms。Composer 时间是 CPU enqueue/控制开销，不是同步隔离 CUDA 成本；GPU 工作已计入完整模型/请求 wall，不能把局部 host 时间当全部额外开销。该早期记录未分开完整提取与 update，最终代码已补充独立字段，未倒填缺失数据。

单轮自然质量请求 wall：Native 2992.566 ms/帧，ZeroKV 3293.223 ms/帧（3 问/帧）；模型 wall 2673.164 与 2955.956 ms。自由生成长度不同、A/B M1/M2 路径不同、没有交替重复，故这些数仅是运行描述，不是配对性能结论。计时包括实际 dataset/preprocess、collate/H2D、完整模型与回答写盘、CPU 输出物化；排除审计序列化和诊断 Teacher。CUDA 事件是流区间，含 host gaps，不等同 active kernel 时间，嵌套计时不得累加。

Peak GPU/Cache、Planner、Prefill/Decode 流区间、观察吞吐及逐帧完整请求数据保存在 descriptive_quality_run_timings.json、quality_frame_summary.jsonl、quality_request_summary.jsonl。未提供不成立的置信区间或净加速。

## 执行合法性与待完成条件

唯一现有 Runtime、Cache、PrefillExecutor；Executor 源码未修改。输入 layout、Adapter/模型/资格/执行版本、Position/Mask、连续因果输入及缓存来源仍由原 Matcher/Cache/Executor 校验。相同有效 Anchor 可允许逐元素相等的稳定前缀；核心撤销/变化即截断到视觉起点，只允许相同固定文本，不能因空核心 Embeddings 相同就复用视觉。动态与问题仍需当前 Prefill，不能跨过变化 Tokens。

606 项 CUDA 回归测试通过，包括空核心视觉拒绝、版本失配、Cache 元数据篡改、固定文本合法保留、ID 不进入语义编码、动态输入梯度及原有执行合同。单元测试没有被当成真实跨帧吞吐量证据。

下次只有在新独立数据上解决空间质量和目标域覆盖后，才校准绑定新 layout/Adapter/资格的策略，并运行 B/C 同输入、真实跳过 Tokens 和完整 Request wall 三轮场景配对验收；原生策略证据不能直接继承。
