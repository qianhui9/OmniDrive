# 真实视觉桥接审计

结论：投影区域接口的有界预训练检查通过；不存在 WorldAnchor 与原生 VLM token 的一一对应。正式部署资格未获认证。

六相机经实际输入 metadata 顺序排列，RGB归一化、resize/crop与K同步更新，本轮处理后输入640×640。EVA/M2输出 `[1,6,1024,40,40]`，原生patch stride16，camera-major、随后row-major。M2复用的是前部窗口及近似全局残差，其后仍有当前视觉计算；不能把窗口复用当作最终LLM输入相等。

`petr3d.extract_img_feat`将M2输出整理为上述网格，Object/Map Head先投影1024→256并结合当前相机几何与历史时序状态。`streampetr_head.py:533–562`把256额外学习查询放在检测查询之前，额外query的reference/query_pos置零，检测框来自去掉额外查询后的分支。最终256个额外输出投影至4096，再追加1个CANbus/ego token。`petr_head_map.py:404–430`同样把256额外查询与1800个lane查询分开，lane由control points表达，额外map输出不构成道路栅格。Native视觉顺序固定为object256、ego1、map256，即 `[1,513,4096]`，LLM扩展时转其计算dtype。Head输出含当前ego/历史变换和全局上下文，通常跨帧不同。

M1通过`omnidrive_adapter.outputs_from_model`把当前检测的ego坐标gravity center/corners经ego_to_world转换为世界框，按类别/大小/距离合法门加一次全局分配关联。普通关联不等于物理身份认证。`state_manager.py:519–528`将整份backbone/object_carrier/map_carrier来源绑定到多个anchor，`associated_feature_ids`没有token级归属或框索引映射。每个anchor的observation support则保存真实相机时间、投影xyxy区域和RGB描述子；描述子也不是前景mask。

新增`worldstream/m3/visual_feature_bridge.py`只读当前提交的M1对象与已有EVA/M2网格。用已校验的camera-time外参、预处理K重新投影选中框的8角，取投影凸包内原生网格中心；按padding前真实image bounds裁剪，并排除其他当前对象support AABB，选择剩余位置最多的相机。无需新编码器、检测器或tracker。读取的是实际观测区域的上下文化特征，既非独立纯对象特征，也不宣称不含背景/全局影响。其他框排除保守，不构成遮挡/前景证明。

160帧/5个日志的富集诊断产生262个合格身份观测，233个可提取区域；29个因支持不足拒绝。独立离线GT只在提取完成后评分，233/233可评分，222/233=95.2790%区域满足“至少80%网格中心落在对应GT 3D框投影凸包”，通过预设95%门；137次相邻身份延续正确。11个几何失败例保存在`bridge_analysis.json`，最差支持率50%。GT框投影包含背景，95%门是区域适用性预检，并非95%真实前景纯度；两个开发日志中一个仅有2个区域且1个失败，不外推单日志保证。

原始网格位置中位数52，K64/128/256分别仅95/34/10个区域有足够位置。独立自然24帧中6个合格对象区域均只有15个位置。旧car Core资格没有自动授予64个真实视觉观测的资格。

生成帧字段表示当前M2输出的读取帧，不表示该张量的每个上游窗口都首次在此帧计算；本轮未额外导出每个EVA输出位置的完整M2计算谱系。内容hash绑定实际读取的张量。所有新增桥接代码保持研究接口，未写入正式detector路径。
