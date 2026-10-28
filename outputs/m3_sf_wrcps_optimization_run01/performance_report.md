# 性能结果与停止阶段

没有准入正式A/B/C服务测试、固定tokens/相同步数控制、三轮交替或场景bootstrap。因此ΔT_KV、Prefill CUDA改善、M3 Wall改善、E2E加速比、配对95%CI、FPS/QPS改善均为null。不能从单次数据采集计时或CPU资格回放推断加速。

真实执行的192个六相机连续帧/576个自然问题均为原生布局B3-ZeroKV载体/候选采集；另有旧故障载体8帧24问的Native/SF无核心成对诊断。前者禁止历史复用，后者是同载体表示合同。真实跨帧视觉KV请求0、tokens0、跳过prefill tokens0；这不代表已训练SF方案的性能测量。

原始逐请求prefill/decode CUDA事件、cache成本、完整M3 wall等保留在各capture的per_question_metrics.jsonl，整帧Request/Model统计在metrics.json。采集上下文摘录见 capture_timing_context.json，明确不构成正式A/B/C对比。GPU内存为现有模型采集内存，训练和正式SF效率内存均未测量。

四个计数层次保持分开：19个资格回放Anchor命中（其中独立验证9）只是CPU状态延续；它们没有进入SF PrefixMatcher、Scheduler准许或PrefillExecutor跳过计算。开发训练集没有真实核心延续（开发验证有5次），训练先被拒绝，因而不能把资格命中乘K当作实际复用量。

保持原正式配置。主要瓶颈已从旧的无核心表示损失，进一步定位到核心跨新Log的覆盖不足、单car语义信息量小、以及有效核心空间质量尚无可训练且独立验证的数据证据。
