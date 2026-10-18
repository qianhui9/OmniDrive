# 实现改动与验收边界

- 新增严格接口语义：historical projection candidate、observation supported、physically certified；保留 valid_anchor_ids 兼容，物理证据不足时逐项说明缺失条件。
- Precheck 的验证、过期清理和 last_verified 写入在私有 store 中进行；模型未成功时不会发布。Commit 在私有工作对象完成全部 anchor/feature/descriptor/provenance 写入，仅在最后发布一次；公开 store 在计算期间也保持旧快照。
- SHA256 绑定实际图像、标定、位姿、相机顺序和支持数组；pending epoch/单调超时、重复/旧提交、显式 abort 和下一帧恢复均验证。
- 补充 track birth、actual observation、last verified source、association count、有界历史、certified count 与 anchor provenance；不同状态来源和计数不混用。
- 在关联前按确定性优先级选择容量内预测，保留原始 prediction index。新增双端歧义拒绝和缺帧后拒绝直接延续。SciPy 为必要依赖，不再使用不等价 greedy。
- 旋转框采用经过角点结构校验的局部坐标 membership；保留三角化/视差/重投影等拒绝条件，不将框内点当作实例前景。检测头有限速度不标为可靠物理运动。
- 输出 Stable/Changed/Unknown/Refresh 四张可逆精确 Mask，逐相机保留 coordinate space 与 raw transform。保留 observation partition，单列强制刷新原因及最终刷新比例。框描述失配为 Unknown，不能伪造 Changed。
- 离线工具修复排序/截断后的 GT 评分索引映射；GT 只由 tools 评分读取。新增 M1 独立冻结协议、回放、异常注入、Oracle、成本和报告工具，以及 26 项针对性测试。
- 旧测试全部保留；其中 Precheck timestamp 和 missing-camera 的断言改为新事务契约；补正两个合成相机的 camera_id，以真正测试多视图而非输入标识错误。

没有修改 projects/、M2/M3 算法或 checkpoint，没有训练，也没有在质量未通过时开展 P1 加速。首轮确认保留在 attempt_before_atomic_publication/，最终实现重新执行完整回放及原模型对照。第一次最终 GPU 数值对照配置关闭日志，故再次使用同样输入强制保存成本诊断，最终以 gpu_perception_profiled/ 为准。

明确限制：所有本地验证 scene 均有旧记录曝光，没有新的独立 Final Test；前景/身份生产者和全人口定位/运动误差界仍缺失。Physical Certification 和 Region Validation 均 No-Go；成本测量不是完整系统加速证明。完整 API 说明见 ../../docs/worldstream_m1_independent.md。
