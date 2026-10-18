# 在线证据与方向认证审计

{
  "frames": 72,
  "current_rgb_frames": 72,
  "ego_pose_frames": 72,
  "historical_rgb_frames": 54,
  "current_detection_frames": 70,
  "calibrated_camera_frames": 72,
  "temporal_correspondence_frames": 72,
  "independent_foreground_mask_frames": 0,
  "tracked_correspondence_points": 428001,
  "geometry_valid_correspondence_points": 102351,
  "supported_correspondence_points": 45312,
  "object_candidates": 1783,
  "candidate_source_frames": 1783,
  "observation_consistent_candidates": 8,
  "history_association_candidates": 812
}

历史 Source Frame、当前 RGB、标定和相机时间 Pose、历史三维几何、当前 Head、历史关联、双向对应点及残差均可获得。它们来自实际传感器/预测接口；历史关联和 Head 一致性仍是模型自身证据。当前 adapter 从未提供独立实例前景 mask，也没有问题指称到实例的独立映射。背景可穿过同一 ROI 并满足双向、极线与纹理检查，因此无法由 ROI 相似性确定物理身份。现有三角化对动态对象主动拒绝，对静态对象仍不证明前景归属。

已实现独立认证分支；当前数据缺少必要证据而拒绝，区别于此前认证算法本身不存在。候选正确但缺证明、候选错误、候选无值、不可评分分别计数。6 个校准场景不足以得到 95% 场景单位 Conformal 有限界；2m 离线关联还存在截断选择，不能外推。未将 Detector Score 转为米级误差，未使用测试 GT 构造在线区间。零接受时 FAR 和上界为 null，不能报告零风险或部署通过。
