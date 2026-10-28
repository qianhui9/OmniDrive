# Composer 实际训练结果

RTX 3090 24 GB 上完成真实 Teacher/Student 训练，共 3 epoch、288 次优化更新。按独立 Adapter 验证 Loss 选取 epoch 0（96 次更新）作为 adapter.pt；没有用最终 VAL 验收选择 checkpoint。训练完成不等于质量准入，当前 deployment_qualified=false。

实际 8 个其他官方 TRAIN 场景/Log、64 个连续帧，4 个训练 Log 与 4 个验证 Log；每帧原始有序 3 个自然问答，训练 96 QA、验证 96 QA。与资格校准/验证及最终验收的场景、Log 均不交叉。真实非零稳定延续仅训练 4 次、验证 1 次；冷启动空核心帧也训练，但不计为命中。这是明显的数据覆盖限制。

K=128，rank=128，固定 513 视觉 Tokens；所有当前对象和地图进入动态 Resampler，Ego 保留。仅 Composer 参数更新；原始 LLM、视觉 Backbone、检测/地图 Heads、M1/M2 参数和基础权重冻结。Answer KL + Task CE，关键问题权重 2，没有新增辅助损失。稳定内容相同则稳定 Embeddings 按构造相同；随机 Anchor ID 不进入语义编码。

|Epoch|训练 Loss|验证 Loss|验证 KL|
|---|---:|---:|---:|
|0|4.373756|1.183139|0.034976|
|1|1.038471|1.245019|0.055603|
|2|0.942608|1.431241|0.120186|

微批次 1，FP32 Composer/当前载体在原生 LLaVA 输入边界转换到 FP16 LLM；冻结 LLM 仍真实反传到 Composer。梯度检查点保留原生子模块 eval 行为，非 LLM 模块在离线训练时移至 CPU。第一条真实 backward 峰值 allocated 14,645,022,208 bytes（13.64 GiB）、reserved 14,768,144,384 bytes，输入 595 Tokens、回答 22 Tokens；全程日志最大 allocated **13.7173 GiB**。不是仅导入或假定显存足够。

权重及基础模型 SHA 见 model_artifacts.json；adapter SHA256：949d93aae435f1f9ef1c085ee657651c21d9ecc9902744035bf25fda55b89ebd。完整逐步 Loss、训练配置和真实显存凭据见 adapter_training/。最佳验证 Loss 为 1.183139120，后两轮验证退化，故未使用最后一轮权重。

本轮只训练 K=128。K=64 代码接口存在但未训练/验收；最终 VAL 上没有合格核心且表示出现空间回归，继续在同一验收集搜索 K 无法修复证据独立性。不能报告 K=64 有效。
