# 本次必要工程补齐

上传需求的主体已在 HEAD `0e78092` 实现，历史实验保持原状。本次在不改方法、参数、模型或数据划分的范围内补齐来源验证。

- `identity_continuity.py`：来源/当前轨迹时间必须有限且绑定实际 AnchorObservation；来源相机ID必须相同。认证端检查真实来源裁剪，重算 distinct track 的空间象限，拒绝负 FB 误差和未通过的共同视图。独立身份声明的证书增加来源/当前图像 digest、执行签名、资格文件 SHA256 和明确 claim scope。
- `physical_certifier.py`：资格文件的单位、计数、scene/log 和 calibration 列表按类型验证。bool 不作为事件计数、非整数不作为实体数量、畸形对象不导致认证链崩溃。已有 scene/log 独立性、单侧95%上界和5%风险要求全部保留。风险文件哈希只作来源绑定，不替代独立质量。
- `state_manager.py`：只读 Post-Commit 查询增加与已提交 Store execution_signature 的一致性检查。保留原 commit() 返回契约；不回填 Precheck，不改变事务发布顺序。
- `surface_visibility.py`：第三视图必须是同一相机、同一坐标空间、有效且可靠定时的相机。跨相机/坐标历史继续 Unknown；正常几何/光流/可见性门槛不变。
- `test_identity_surface_bindings.py`：新增18项受控边界测试。修复前13失败、5已通过；修复后全部通过。该统计只属于工程验证，真实物理认证成功仍为0。
- `revalidate_identity_surface.py`：复用原始数据和评价函数，强制新输出目录，限定已曝光 Development/Calibration 的复现；保存原始trace/真实图片并比对身份测量、区域指标与保守 Mask。不会选新算法或改写旧归档。
- `profile_identity_surface_bindings.py`：在固定8个已曝光预加载驾驶帧上，与 HEAD 完整源码快照交替计时；排除图像IO、模型推理、输出adapter、Oracle及日志。逐相机Mask与身份测量/判定必须一致。只是成本诊断，没有P1算法优化或部署加速声明。

默认配置和原严格前景生产器保持；未修改 M2/M3、模型、Checkpoint、LoRA、图像预处理或原 Detection/Map 推理逻辑。历史 Heldout/Final 不用于重新调参/选择。本次不签发任何新的独立质量资格。
