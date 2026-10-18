# M1 实现与验收边界

本轮只改动 `worldstream/m1` 及其独立配置、测试、工具和文档。保留原模型权重、Checkpoint、LoRA、感知预处理/推理参数以及 M2/M3。此前结果目录没有修改。

## 代码

- `physical_certifier.py`：固定拒绝替换为声明级验证。校验实际数值、来源/当前帧、场景、epoch、执行签名、图像 digest、几何版本、时效和受信生产路径回执。独立质量资格采用 canonical JSON 哈希固定；重复实体/帧不增加独立 scene/log 样本数。没有批准资格时仍拒绝。最小声明仅限静态候选身份连续性和局部射线一致性，不批准中心、距离或速度。
- `foreground_correspondence.py`：显式实验开关下使用实际双向 RGB 对应、相机姿态、三角化、OBB 内点、空间分布和环深度对比。背景与目标不可区分、视差不足、遮挡或视图冲突时 Unknown。框仅作候选约束，没有生成实例前景 Mask，没有额外网络或在线 GT。
- `state_manager.py`：新增生产和认证只发生在感知完成后的私有 Commit；不向已经执行的 Precheck 回流。新检测替换几何时更新版本并清除旧认证。新增源 digest 检查，保留原子提交和 abort 契约。
- `state_store.py` / `geometry.py`：几何、描述子和历史灰度图采用独立的不可写 bytes-backed 数组；fork 隔离包装和可变元数据，feature acquisition 防止可写别名。Tensor snapshot 隔离，默认不捕获大 tensor。
- `dense_support.py` / `temporal_visibility.py`：实验局部策略围绕视差突变、折叠、双向错误、显著光度残差拒绝稳定支持并进入 Unknown。比较固定半径 4/8/12；保留动态扫过保护与原 Mask 语义，默认仍为原 uniform 保守方案。
- `association.py`：新增离线排边重求全局 assignment 对照，区分匹配数损失与总代价变化。在线 local margin 仍为 0，未额外加入在线求解或调阈值。

## 实际验证

保留旧 346 项通过的用例，本轮新增 33 项。CPU 沙箱运行 379 passed、1 skipped，另在真实 CUDA 可访问环境完整重跑，最终 **380 passed、0 skipped**；原 CPU 记录保存在 regression_cpu.xml/log。受控测量正例验证认证器能够消费完整输入，但不计作现实物理成功。另有原模型实际 GPU 8 帧输出、backbone 和检测/map head memory 完全一致验证。数值对照与计时见回归 XML 和 JSON。

原 26 个已曝光场景共 368 帧仅作 Development/Reproduction，复现原 Heldout 97、Final 52、开发长序列 367 个关键点漏刷，逐相机精确刷新 Mask 不变。新增 38 个对本地旧 M1 记录新的 train 场景/互异 log，按 20/6/12 划分，每场景 12 帧；456 帧检测来自本轮原 checkpoint 实际 GPU 运行。新旧合计 64 场景、824 唯一回放帧，每帧比较六种真实策略及全刷新参考。新增场景仍属于模型训练人口，不是原模型独立测试。

算法和阈值在质量评分前冻结；Calibration 无合格局部候选，Heldout/Final 没有重选。局部方案提高覆盖，但 Calibration 最保守的 local_r12 稀疏参考错误稳定率仍为 169/1336 = 12.65%，超过 5%。不将覆盖增长报告为成功。精确旧漏检、实际像素图、新失败和全部分母保存在 trace。

生产器有非零候选，但稀疏 LiDAR + GT 体积参考不能给出完整实例像素归属。没有签发真实质量资格；严格物理认证接受为 0，FAR/上界均为 null。20-scene 定位校准仍受唯一身份/2m 筛选截断，只报告条件结果，不接入在线几何认证。

## 冻结后的指纹加固

质量运行后的唯一核心源码差异是资格协议指纹：补充绑定全部证据相关配置和生产器源码。原始冻结协议不覆盖、不重写；实测原型保存在 `measured_prototype_source/`。所有数值算法函数 AST 不变；所有实验资格均为 null，资格检查在读取协议 digest 前已拒绝，故观测、Mask、状态和接受决定不变，仅记录的诊断 digest 改变。详见 `evidence_protocol_binding_audit.json`。最终回归和独立 CPU/GPU 复测采用最终源码。

## 默认与成本

生产器默认关闭；显式研究配置 `configs/worldstream_m1_evidence.yaml` 开启实验观测但仍没有批准资格。局部策略没有成为默认。独立单进程 CPU 对照和空闲 GPU 协议复测分别记录，先前与分析并发的运行保留但不用于加速结论。新增计算没有获得独立批准覆盖，Runtime 为 No-Go；按任务要求未进行质量未通过前的 P1 优化。

接口、复现路径和适用边界见 `docs/worldstream_m1_evidence.md`；最终数据与四项独立决策见 `experiment_report.md` 和 `go_nogo_decisions.json`。
