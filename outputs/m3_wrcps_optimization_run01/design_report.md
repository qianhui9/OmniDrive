# WRCPS run01：实现与实际可行性结论

本轮代码从 `268e0d2b` 创建分支 `m3-wrcps-20261028-run01`，保留 `c6cd3ced` 及原有正式配置。实现了实验性 WRCPS，但真实第一阶段未通过，**不推荐部署，未训练 Adapter，也未宣称质量或加速达标**。

## 实现

- `worldstream/m3/stable_world.py` 从当前已完成的 M1 Commit 导出只读稳定快照。只接受 `get_current_certifications()` 发布且带独立资格证据的身份连续性声明；不接纳普通几何匹配或 RGB 候选。当前支持身份、类型、观测类别，不声称已有认证地图几何/拓扑。速度、位置、自车状态和交通灯状态留在当前视觉中。
- StableWorldAnchor 的内容摘要只含稳定事实；Frame、Timestamp、Epoch、State Version 和执行来源单独验证。常规递增版本不改变内容摘要；回滚、场景/来源/epoch变化、时间跳变、内容变化、8帧上限、证据缺失会换 Anchor。没有证据时每帧更新 Anchor。
- `world_prefix_composer.py` 显式定义 `wrcps_stable_dynamic_v1`。固定 513 tokens：128 Stable + 385 Dynamic。稳定编码使用事实的 UTF-8 字节、位置与学习查询；同一 Anchor 直接返回缓存 Tensor。动态部分用两个低秩 resampler 读取全部当前 256 Object 和256 Map tokens，保留当前 Ego token，输出192 Object摘要+1 Ego+192 Map摘要。压缩存在信息损失风险，必须通过训练和 ZeroKV 实测质量判断，梯度覆盖不是语义能力证明。
- Stable 编码器只读 Anchor；不从当前/历史视觉中冻结前128个 Object tokens。大状态超过声明的编码容量时报错，禁止静默截断。缓存 Tensor 修改、加载后权重修改均拒绝。
- `runtime.py` 在唯一 M3 路径、原有 Planner / Matcher / Scheduler / PrefillExecutor 中使用新布局。旧 native 布局约束保留；新布局有独立 Group Spans 和签名。模型、Adapter文件Hash、代码实现签名、布局、Position与Attention检查均保留。失效 Anchor 清除旧 KV，完整计算仍由同一 Executor 执行。PrefillExecutor 本身无改动。
- 原生模型权重、六相机、M1/M2算法和正式配置未修改。新增 Adapter 只可用 `exact_prefix`；旧近似 policy 不能套用。新policy还必须绑定 Adapter Hash、执行/布局签名和新成本质量证据。
- `wrcps_train.py` 提供冻结 Teacher/LLM 的答案 KL + task CE 训练，关键问题提高CE权重。相同稳定事实的时间一致性由结构保证，不另造伪监督。训练输入必须为带Hash的真实连续训练集 capture 和原始问答标签；验证/训练场景分离，校准/验收不得进入训练。训练快照必须等于实际捕获的当前 Commit 快照。

## 第一阶段真实实验

8个未进入历史 M3 语言实验的场景，每场景6个连续六相机帧，共48帧。2场景开发检查，6场景独立检查。每帧按冻结清单执行最前3个真实原生自然问题，共144请求。未使用重复测量来增加独立样本数。场景对原模型既有训练/既有完整感知评估并非全新，这里“独立”仅指历史 M3 语言开发/校准记录外、且与本轮训练/校准不重叠。

可用稳定事实：0/48帧；Anchor Hit：0/40跨帧转换；有效Anchor平均持续0帧。保守无效Anchor每帧更新，生命周期1帧；两者不能混报。6131条已提交Anchor记录均为 `unqualified_geometric_track`，全部 identity_certified=False。这是逐帧记录数，不是6131个独立对象。

实际原因：正式 `M1.commit()` 中物理结果列表始终未填充，`get_current_certifications()` 未返回证书；配置 `physical_qualification_path` / SHA 未提供独立验证来源。历史 precheck 只能报告几何/RGB观察支持，不证明身份/前景归属或地图拓扑。不能将多次几何关联、置信分数、M2窗口稳定或同类别观察提升为世界状态认证。导出稳定快照是本轮最小必要接口补充；在缺少真实测量生产者和独立校准证据时，仅增加字段不能补出可靠稳定性。

首次尝试去掉语言问题以减少审计开销，被正式路径的“至少一个语言问题”合同拒绝。保留失败日志 `anchor_audit.log`；成功运行使用冻结的3个原生问题，没有修改正式执行要求。

## 核心问题回答

1. 当前M1足够稳定吗？存在持久候选ID，但没有足够的已认证世界语义支持合法WRCPS。
2. 有效Anchor平均持续？0帧；无效Anchor每帧刷新，平均1帧。
3. Composer能产生严格相同Prefix吗？合成合同测试在CPU和CUDA检查同Anchor严格相同；真实数据无合法同Anchor，不能声称真实成功。
4. 平均真实视觉KV复用？未运行WRCPS请求，故该均值不适用；实际WRCPS视觉复用总量为0。
5. 相对35文本tokens增加多少？未取得任何真实世界稳定前缀带来的增加；不能把原生路径同帧问题复用算作WRCPS跨帧收益。
6. 新布局精度变化？未训练、未评估，未知；四项预算未被认定通过。
7. ExactKV额外答案变化？小模型合同测试无差异；真实任务未知。
8. Prefill/M3/Request加速？均未测得；没有合法WRCPS A/B/C对照，不填伪造加速比。
9. 新模块开销？只记录真实快照CPU重放Anchor管理开销（见performance_report），不是完整在线成本。未运行Composer服务开销。
10. 相对Native稳定E2E加速？没有证据。
11. 最大障碍？当前首先卡在稳定世界状态的证据和生命周期，不是已测得的Decode或Composer性能瓶颈。
12. 能支撑系统论文贡献吗？目前是有测试的实验实现及负面可行性证据，尚不支持方法质量或性能贡献的正面结论。

## 后续门槛与局限

不通过阈值放松或冻结未认证视觉来制造成功。需要先补齐现有M1候选连续性的真实测量生产者及场景独立资格评估，再取得真实训练集稳定连续帧。之后才训练并验证Native vs ZeroKV四项质量及关键语义，重新校准Exact策略，再按validation_plan.json执行单/三问题、自由/固定token、三轮交替的A/B/C实验和场景级配对CI。该资格补齐涉及M1证据生产和独立验证，超出一个轻量M3输入接口可以解决的范围，本轮未扩大M1架构。

当前仅完成第一阶段真实审计。训练/质量/真实跨帧KV/E2E阶段明确未完成，保留所有实现和未达标证据，不替换现有推荐配置。

## 状态版本补充审计

`m1_version_audit.json`逐场景记录Epoch、State Version、时间戳与执行签名。成功审计中各场景内部Epoch和执行签名保持不变，State Version正常递增；完整快照摘要48帧各不相同。不能拿完整快照摘要当作稳定内容身份，但去掉这些变化字段也不能将空事实集证明为稳定世界状态。

最终CUDA全量回归595项通过、0失败、0跳过，见`tests_gpu_final.log`；M1/M2源码、原有正式配置和PrefillExecutor保护检查通过。
