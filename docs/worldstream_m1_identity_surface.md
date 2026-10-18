# M1 身份连续性和三帧局部表面实验

`identity_continuity` 默认关闭；`dense_region_policy` 默认保持原保守策略。显式实验配置为 `configs/worldstream_m1_identity_surface.yaml`。未发放任何真实生产器资格，不能将候选直接用于后续模块。

## 身份声明

`object_identity_continuity` 只声明指定来源和当前实际检测对应同一物理实体。它不声明静止、精确中心/速度、前景像素或可复用的完整 Box 表面。原 `static_candidate_identity_continuity_with_local_ray_consistency` 生产器和声明继续保留，分别测量。

`identity_continuity.py` 定义一种有限人口：实际上一帧来源、同类对象、无门内竞争的双向候选关联、源/当前检测位移不超过固定 0.5m，以及实际共有相机中的双向轨迹和裁剪观测支持。采用现有光流/纹理/裁剪门槛，没有加权分数。检测分数不被解释为身份概率。静态射线轨迹支持只是选择条件；真实实例关系必须通过独立标注评价，框内点不等于实例归属。

证据仍通过 `PhysicalEvidence` 的来源/时间/场景/epoch/版本/digest/执行路径回执检查。身份生产器协议绑定配置和传递依赖源码。资格 JSON 必须使用 canonical SHA256 固定并经独立审计，且所有接受均可评分；任何错误使场景/log 记为失败，单侧 CP95 上界必须≤5%。哈希与执行回执只绑定协议/执行，不能替代独立质量证明。受控 60-scene 正例只用于接口测试。

## Commit 后读取

`commit()` 保持原返回类型；`M1Result.physically_certified_anchor_ids` 仍是 Precheck 时刻的历史验证字段，不得反填当前实际检测的后验结果。

完成原子 Commit 后调用：

```python
precheck = manager.precheck(frame)
manager.commit(frame, actual_outputs, precheck)
published = manager.get_current_certifications()
for certificate in published.certifications:
    assert certificate.current_frame_id == published.frame_id
    assert certificate.timestamp == published.timestamp
    print(certificate.anchor_id, certificate.certified_claim)
```

查询返回 frozen `M1CommitCertification` 和防御性复制的证书。来源/当前 frame/version、scene、timestamp、epoch 和 claim scope 在证书内。嵌套 provenance 是防御性副本，调用方修改它不会影响 Store。查询只读取已发布状态：pending 时仍看到上次 Commit，提交异常不会暴露准备中的证书；reset、几何替换、时间/epoch/当前版本不匹配时旧证书不可返回。查询不改变任何 Mask，也不提供米制几何或 Transformer 依赖相等证明。

## 三帧表面判据

两条射线可为错误的沿极线匹配重建出自洽深度。实验策略在已有逐像素光度/纹理/FB/几何和动态扫过保护之后，要求可观测视差，并要求当前—上一帧三角化的点解释另一次实际测得的当前—前两帧长基线对应；同时检查直接与两段复合对应闭合、双向流和局部映射方向。它是一个三视图闭合判据，没有新半径搜索或额外网络。

缺历史、低视差、时间不一致、可见性/闭合不足均为 Unknown。新判据只拒绝 Stable，不将无法判定伪装成 Changed。已有局部光度 Changed 保持原含义；强制 Refresh 不修改观测分类。额外历史仅保留一组不可写相机图像和 source map，随原子 Commit 发布并随 reset 清空，不随序列无限增长。

重复纹理和遮挡仍可能同时骗过多视图 RGB。该判据没有无条件表面归属保证；只有独立风险评价与有效覆盖收益同时达标，才可能批准有限区域用途。消融 `surface_observable_ablation` 只使用两帧可观测性，不能当完整表面验证。

## 评价边界

运行时仅消费实际 RGB、已知相机/ego pose、历史状态和 Commit 后的检测。`tools/worldstream_m1` 才能读取 annotation、LiDAR、实例 ID 和离线 Oracle。评价沿用唯一身份/2m 竞争门限与 annotation-owned 稀疏 LiDAR，明确保留截断与不可评分人口；无返回像素不记作正确背景，GT Box 体积不当作像素 Mask。

新场景与采集 log 均未在本地旧 M1 质量记录中曝光，但仍属于模型 Train Population。只有 12 个新 log（4/4/4），远少于零错误部署资格至少需要的 59 个独立接受单位。可行性结果与正式身份/区域资格分别报告。全部配置、源码哈希、原始命令、真实 trace、模型数值一致性和成本见 `outputs/module1_identity_surface_refinement/`。
