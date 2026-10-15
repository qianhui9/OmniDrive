# M1 第二轮：时序可见性与局部支持

实现结果与原始数据见 [第二轮报告](../outputs/module1_refinement/refinement_report.md)。第一轮目录保持不变。Python 环境为 `OmniDrive`，本机解释器 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。

本轮配置 `configs/worldstream_m1_refinement.yaml` 保持 `require_visibility_evidence=true`、`trust_detection_velocity=false`、`execution_mode=observe_only`，启用同源 RGB 双向 LK 和局部观测支持。原配置关闭新功能，可复现原有判定。完整模型配置入口为 `projects/configs/OmniDrive/worldstream_m1_refinement_observe.py`。

## 在线边界

`WorldStateManager.precheck(frame)` 只读当前 RGB、相机内外参/时间和历史 commit 状态。`commit(frame, full_outputs, result)` 才能接收完整感知输出并更新实际计算版本。最多保留每个相机上一实际计算帧的一张灰度图；当前 precheck 不刷新历史观测来源。

`OnlineVisibilityEvidence` 包含来源/当前像素坐标、FB误差、几何合法性、局部支持、稳定/变化掩码和来源时间。它来自同一 RGB 传感器，不能表述成独立传感器可见性或目标前景分割。未知动态运动继续拒绝，未提供速度可靠性的新假设。

局部支持由实际纹理点附近7×7采样、光度/梯度检验和标定几何构成。不会填充Box投影矩形。匹配失败/未知遮挡/未覆盖区域为Unknown，实际有合法对应但出现显著局部残差的像素为Changed。Anchor仍是Valid/Invalid二元；Stable/Changed/Unknown只是像素证据。

## M2输出契约

`M1Result.invalidation_regions[camera_id]` 增加：

- `verified_stable_area_ratio`、`changed_area_ratio`、`unknown_area_ratio`、`invalidation_area_ratio`。
- `refresh_override`：周期/场景重置请求完整更新，可能覆盖已有图像支持。
- `evidence_source`、`transformer_dependency_certified=False`。
- `exact_invalidation_mask`：无损packbits/little-endian、zlib、base64编码，带`shape`。可用`worldstream.m1.mask_codec.decode_mask()`解码。
- 既有`regions`仍是32×32保守包络，可能覆盖整图；局部调度必须使用精确mask，不能把图像空隙直接映射成可跳过的Transformer Token。

```python
from worldstream.m1.mask_codec import decode_mask
result = manager.export_m2_inputs()
invalid = decode_mask(result.invalidation_regions['CAM_FRONT']['exact_invalidation_mask'])
```

最新真实导出例子：`outputs/module1_refinement/m2_example.json`。保存的NPZ、可视化二值PNG和导出的精确mask已逐像素比较。

## 执行

```bash
conda activate OmniDrive
export MPLCONFIGDIR=/tmp/omnidrive-mpl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest tests/worldstream_m1 -q
python -m tools.worldstream_m1.run_refinement --output outputs/module1_refinement_new
python -m tools.worldstream_m1.benchmark_refinement --output outputs/module1_refinement_new/latency_profile.json
python -m tools.worldstream_m1.gpu_refinement --output outputs/module1_refinement_new/gpu
```

真实数据入口及预测路径见脚本默认值和`protocol.json`。脚本保护已有完整/部分回放；复现请使用新目录。`commands.sh`记录本轮实际运行命令；`reproducibility/before_optimization/`保存与优化前SHA256完全一致的源文件，用于重新比较性能。诊断脚本依赖已有第一轮原始日志。

GPU进程必须能访问宿主机`/dev/nvidia*`。本机3090和驱动可用；Codex受限执行上下文可能隐藏设备节点，不能将其当作驱动故障。GPU实验包含原LLM执行，Feature Oracle单列，未进行任何跳算。

## 评价边界

原始完整检测回放、OpenLane标注辅助Map回放与新GPU完整推理分别记录。GT只进入离线Oracle，不进入当前precheck。校准与评价按Scene分离。当前校准为`insufficient_evidence`，没有选择部署阈值。

图像支持改善不代表低风险物理状态或精确特征复用。主评价只有5个唯一GT实例提供了可评分接受记录，7/25的物理错误均来自历史定位误差；新增区域召回也未保持100%。少量场景的Feature Oracle支持接受组具有更小平均误差，但最终深层误差与图像残差相关性弱。应继续observe_only并验证真实依赖闭包，不能直接启用M2跳算。
