# 第二轮代码与测量变更

- 身份统计：`identity.py`、`identity_audit.py` 使用 sample_token 连接原始标注与实际 Dataset，并检查 scene/timestamp、重复、间隔、位姿跳变；原 report 的下标连接已替换。17 个历史实际序列连续，旧时延和原结果文件不改。
- M1 调度：`worldstream/overlap.py` 单 Worker 独占 manager/store；`runtime.py` 管理准入、join/drain/reset；`Petr3D.simple_test` 在完整 EVA 之后、heads 之前等待。M2 依赖未满足时不重叠。签名 memo 使用完整参数/Buffer 版本 witness，复用的仍是原 M1 签名内容。
- Decode：`native_decode.py` 只在原生单 Token、已存在 KV、全一 Mask 时跳过零 additive mask；原生 GEMM、LoRA、采样和停止规则保持。检查与同步均真实计费，默认关闭。
- Batch：现有执行器增加公共前缀内存下界预拒绝，预算不足时避免先计算 seed 再失败；没有改变获准 Batch 的数值路径。已有失败 carrier 的逐层算子、KV、批形状、首次生成分歧、Sampling 回退在独立 probes 中测量。
- 证书：仅 frame-gap 性能范围可校准；精确部署/实现/线程配置、形状、问题长度、服务与生成合同不放宽。逐帧完整 E2E 配对上界与独立场景检验决定批准；没有数据或净收益则空批准。真实状态认证不由成本证书替代。
- 测量：复用原 experiment runner，增加定时范围外的前/后审计 hook。完整服务包含预处理、同步、生成 IDs 的 CPU 捕获及原生/运行时 I/O；定时后的大张量审计拷贝除外；M1 状态 digest/离线快照、物理 Oracle、raw Logits 拷贝和 Profiler 单独运行。Kernel 优先用 CUDA correlation 连接 host API 与 CPU scope，External id 作回退，嵌套层/投影只归属一次，kernel sum 与 active union 分开；不推定未归属间隙都是 launch gap。
- 冻结对照：Original Persistent 使用提交 `b4782ab` 的原 `simple_test`，同一加载权重/嵌入 LoRA、相同冻结问题与 RNG。源快照按运行保存；后加载 probe 保存执行源哈希或源码。探索性 smoke 与正式独立场景分开。
- 完整性：原结果关键文件 SHA256 在 `previous_results_manifest.json`，实际 checkpoint 内容 SHA256 在 `model_manifest.json`。各实际 GPU run 检查参数版本和 dtype 未改变。没有训练、合并或改写权重。

验证记录以 `logs/` 为准，实际 GPU 正确性、故障/超时、恢复、场景切换见各 probe 的原始结果。此文描述实现与测量范围，不代替实际 Go/No-Go 结果。

- CPU 故障恢复：所有新执行使用 `taskset -c 0-3,6-31`，启动脚本固定前缀；亲和性进入成本键。初始退出 139 和首次未测量的门检查失败保持独立归档，后续校准与 Auto 使用健康核重新测量。
