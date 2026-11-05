# R0 M1

R0 完整参数位于 `configs/worldstream_r0.json` 的 `m1_config`，与历史权威策略全部字段一致，检测阈值0.3。`configs/worldstream_m1_final.yaml` 保留为同机制的测试输入；正式入口使用完整 JSON 参数。

Precheck 验证历史相机观测与来源，M2 读取绑定快照并计算增量视觉特征。Detection/Map 完成后，Commit 原子发布实际输出。M3 从本帧已经提交的实际观测提取当前事实，不使用旧 WSDR 的 pending 历史文本增强接口。

普通几何关联不代表物理身份认证，当前检测支持不代表视觉特征可复用。未知和缺失不能推导不存在。状态事务、来源、几何和观测正确性测试保留。
