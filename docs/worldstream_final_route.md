# 唯一改进研究路线：R0

正式入口 `projects/configs/OmniDrive/worldstream_m123_auto.py` 继承原始模型和预处理，并从 `configs/worldstream_r0.json` 读取完整 R0。模型边界校验 M1/M2/M3 全部开启且参数吻合。Native 保留作实验基线，不是另一种改进候选。

M1 使用普通一对一关联、时序可见性、局部支持、来源绑定及原子提交；检测阈值0.3。M2 使用唯一 WindowExecutor、普通刷新预算3、全局间隔2和漂移阈值0.7721027880907059。两模块数学执行、来源与缓存安全检查保持。

M3 使用 C-basic：current_only、basic、置信支持阈值0.45、绑定风险上限0.75。支持且准入的问题由当前帧实际感知事实构造回答，零 LLM Forward。其他问题保留完整 object→ego→map 载体，执行完整 Prefill 与 World-Guided K8 目标模型验证；只使用精确 prompt 历史生成草稿。请求 RNG 隔离和 Sampling 校正保留。

`reuse_strategy=exact_prefix` 是完整历史配置清单保留的字段；在本路线中没有持久 Prefill KV 复用。旧 ReusePlanner、PrefixKVCache、WSDR-KV、视觉前缀 Composer 和旧策略入口已删除。其配置值不能通过修改字段重新启用，M3Config 会拒绝非 R0 组合。

完整字段记录保留为对照旧收据的兼容清单，包括固定关闭/空值字段；它们不再对应可选择的实现。全部正式参数见 JSON，不能仅按摘要重建。

这次整理不训练、不修改权重，不重新选择参数。历史暴露标记、UNKNOWN、覆盖损失及严重错误仍需报告。复现和测试命令见 [运行指令](../tools/运行文件.md)。
