# 本轮修改范围与原因

基准 HEAD：`169c2969a5daa566990583db72bf0288845774f0`。实际目录为 `/data/qh/phdCode/work4/OmniDrive`（需求中的 `OmniDriv` 按实际仓库解析）。本轮未修改 M2/M3 源码、模型结构、参数、LoRA、Planning 输入、Sampling 参数或冻结质量门限。

| 文件 | 实施内容与原因 |
|---|---|
| `worldstream/m1/state_manager.py` | 同 Epoch 全部已提交帧 ID 去重；无效身份、位姿、相机和时间不进入历史；绑定历史相机快照；被外部改写的来源不能作为回滚目标，直接清空并记录中止；新检测仍产生新版本；导出当前已提交审计。前景生产者收到实际竞争数。 |
| `worldstream/m1/state_store.py` | 事务来源摘要覆盖完整 Anchor/Feature/Provenance、场景、执行签名和帧 ID 集合；新增明确 Tensor 可用状态和 `network_reuse_authorized=false`。 |
| `worldstream/m1/input_contract.py` | 将元数据也绑定到精确帧摘要，包括可供资格审查的日志来源，防止提交期间改写。原有在线 GT 拒绝保留。 |
| `worldstream/m1/physical_certifier.py` | 保留 95% 单侧上界 ≤5% 门限；加强校准日志完整性、当前评估日志隔离、重复测量和相机空间支持检查；重新计算并保留场景级风险统计。 |
| `worldstream/m1/foreground_correspondence.py` | 拒绝重复、非法或坐标不匹配的实测轨迹；回执保留逐相机 UV 与世界点；协议绑定 M1 全部传递实现摘要，避免修改辅助几何代码后复用旧资格。 |
| `worldstream/m1/identity_continuity.py` | 候选阶段拒绝重复轨迹；协议绑定全部 M1 源码；保留资格风险来源。身份声明不扩展为静止或度量几何。 |
| `worldstream/m1/identity_temporal.py` | 在有限身份认证回执中增加场景级资格风险来源。未放宽 LK、双向、空间、竞争或运动一致性门限。 |
| `worldstream/m1/provenance.py` | 导出带版本来源、历史验证来源、当前身份拒绝原因、相机观测、资格和事务状态的防御性副本。物理/度量几何认证始终单列。 |
| `worldstream/m1/omnidrive_adapter.py` | 日志支持独立的当前提交状态快照，历史 precheck 不被当前证据反向回填。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 最小运行时接线：记录提交快照；backbone/head/adapter 异常及时 abort_pending。保留原计算顺序及全部网络计算。感知已经成功提交后 LLM 异常不会撤销真实感知提交。 |
| `configs/worldstream_m1_effective.yaml` | 新建真实 RGB 测量、有限身份和前景证据配置；保持 observe_only、shadow 身份关联、无资格文件、无神经 Tensor 捕获。 |
| `projects/configs/OmniDrive/worldstream_m1_effective.py` | 原始 Planning 配置继承链；显式 M1 开启、M2/M3 关闭。 |
| `tools/worldstream_m1/qualification_inventory.py` | 离线盘点真实验证场景/日志与标注文件；保存日志互斥的可行性划分和最乐观风险界。数据不足只输出 BLOCKED，绝不生成资格文件。 |
| `tools/worldstream_m1/effective_acceptance.py` | 两次真实图像与历史实际模型检测回放；独立核查逐帧提交、来源/时间/Epoch、候选和认证；保留拒绝原因、覆盖与实测对应点统计。无 GT 输入。 |
| `tools/worldstream_m1/effective_gpu_validation.py` | 调用已有 model_smoke 完整原生 Planning 入口，增加源码/权重/数据固定、实际生成文件一致性和独立日志验收。没有创建推理执行器。 |
| `tests/worldstream_m1/test_effective_acceptance.py` | 38 项新增对抗测试：非法帧、相隔多帧重复 ID、完整来源篡改、资格泄漏/风险、重复对应点、模型异常、不可用 Tensor、独立验收器拒绝未来来源。合成样本仅用于契约测试。 |
| `tests/worldstream_m1/test_evidence_contract.py` | 原有合成资格夹具补齐明确校准与评估日志来源，匹配更严格协议。不是实测资格文件。 |
| `tests/worldstream_m1/test_identity_surface.py` | 合成夹具补齐日志；来源改写现在要求事务中止，替代继续提交未认证状态。 |
| `tests/worldstream_m1/test_independent_contract.py` | 外部 Epoch 改写要求丢弃不可信状态；其他异常仍验证原子回滚。 |

# 实际调用链与生命周期

`Petr3D.forward_test → simple_test → frame_from_model → WorldStateManager.precheck → extract_img_feat(完整 EVA) → simple_test_pts(完整 Detection/Map) → m1_observer → outputs_from_model → WorldStateManager.commit → 原生 Planning/LLM`。

precheck 在精确帧绑定和输入检查后，检查重置条件，并在私有 Store 上验证历史观测、计算实际 LK/DIS 对应、投影候选和失效区域。当前完整模型输出此时尚不存在。其结果中的 `valid_anchor_ids` 只表示历史观测支持；`physically_certified_anchor_ids` 不被 commit 回填。

commit 在待提交事务未超时、帧摘要、Epoch、来源 Store 和历史相机快照均一致后，执行候选关联、实际前景/身份测量、资格门、新版本状态和 Feature 来源构造。全部完成后只有一个 Store 发布点；半成品不暴露。实际图像仅来自成功的感知 commit，precheck 不延长历史来源的实际观测时间。

新检测清除上一几何版本的证书。`get_current_certifications()` 只返回与当前已提交 Frame/Scene/Timestamp/Epoch/Execution/Version 一致的有限声明。`export_state_audit()` 分别保留 observed、candidate、历史 observation_supported、identity_certified、physical_or_geometry_certified、身份 unknown/rejected 和失效/事务原因。被替换的新状态附带的历史验证来源另有旧版本与源帧，不能将它视为新几何已认证。

场景、相机布局、执行条件、时间倒退或长间隔建立新 Epoch 并全量刷新；非法帧不建立历史。过期记录在私有事务中裁剪并记录 ID。常规执行异常保留最后可信提交；已提交来源或 Epoch 被外部改写则清空来源，记录 aborted/source_state_discarded，再从完整计算恢复。

所有神经 Feature 默认仅存元数据，`tensor_captured=false`、`tensor_availability=unavailable`、`network_reuse_authorized=false`。没有因此少算任何 EVA、Head、Prefill 或 Decode。
