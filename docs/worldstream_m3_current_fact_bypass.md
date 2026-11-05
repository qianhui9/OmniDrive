# R0 M3：当前事实直答与目标验证解码

唯一研究配置为完整冻结 C-basic。Runtime 先校验场景、帧、时间、相机和当前 Commit，再按问题选择实际感知事实。达到固定支持度与目标绑定准入条件时，构造 FAST 答案，不执行输入展开、Prefill 或 Decode。

直答不读取完整历史答案，不以缺少目标推断 No；UNKNOWN 保持未知。未准入或不支持的问题仍使用完整当前视觉输入，经 PrefillExecutor 和原目标模型执行。精确 prompt 的历史生成仅作为 K8 草稿，经目标模型验证及 Sampling 校正，不直接当作当前答案。

没有 WSDR 前缀规划器、持久 KV、跨问题语义草稿、视觉前缀 Composer 或旧 temporal/target-completeness 分支。`M3Config` 拒绝这些策略，JSON 中固定空值/关闭字段仅用于完整参数对照。

支持度0.45不是错误率45%，风险上限0.75也不是概率校准。最终平均质量、FAST质量、物理目标身份和严重错误必须分开评价。运行命令与源码删除核验见 [运行指令](../tools/运行文件.md) 和 [benchmark](worldstream_benchmark.md)。
