# M2 唯一窗口执行器

正式入口 `projects/configs/OmniDrive/worldstream_m123_auto.py` 固定 `reuse_boundary=window`、`state_interface=source_linked`、`m1_service=maintain`、`mode=auto`。没有相机缓存、条件形变和 Exact 强制准入。

模型图像为 640×640，经 16×16 Patch 产生 40×40 tokens，窗口为 16×16 tokens，需要补到 48×48。六相机每 block 54 个窗口；仅 block 0、1 是独立前缀，共 108 个输出窗口。Patch Embedding 和 block 2 起的全局依赖部分始终完整计算。

执行器先验证 M1 冻结来源，再进行完整像素比较、Patch 感受野传播与窗口闭包。M1 dirty 区域只扩大更新；unknown 区域也必须由 M2 独立比较，不靠关联、warp 或像素抽样取得复用许可。局部来源过期只扩大对应依赖的更新范围；场景/时间/epoch/签名变化、缓存修改、未校准或质量不足会要求全窗口更新，始终使用同一个执行器。

数值诊断发现，原先将整个 Block 缩成 dirty-window batch 会改变 Q/K/V GEMM 的归约形状；微小 FP32 差异经 FP16 FlashAttention 放大。简单填充若干窗口仍不可靠。修复后的唯一策略保留原生 QKV、输出投影与完整网格 MLP 的形状，仅 gather/scatter 独立窗口 Attention。真实模型中，局部窗口、填充边缘、六相机变化均位级等于原 EVA。

**代价边界：**局部更新时 QKV、投影和 MLP 仍完整执行；仅 Attention 省计算。整个 block 零更新时才省去所有点积工作。`actual_skipped_windows` 表示复用输出窗口，`executed_windows`/`attention_windows` 表示实际 Attention 窗口，`pointwise_window_work` 单独计入完整点积工作。不能把前者当作整 block 加速比。

缓存独立拥有输入和激活，预算检查后原子发布；张量版本及上下文/来源元数据修改触发全量更新。窗口成本键包含 block 权重、输入布局和数值策略，独立于旧相机表。新 schema 拒绝旧打包证据。当前生产批准为空，新的测量不会自动签发任务质量证书。

复现真实权重的算子与连续图像验证（输出目录必须为新目录）：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m2.window_validate \
  --output outputs/window_validation_new --frames 4 --repeats 3
```

该工具直接验证唯一 WindowExecutor，不设置 Exact、不更改 Auto、不写生产批准。自然帧与受控输入分开记录，耗时是视觉算子范围，排除缓存发布与完整服务；不能作为端到端加速证据。

当前修复结果：`outputs/single_route_convergence_20261022/window_validation/`。差异定位原始证据为同目录上一级的 `window_numeric_diagnosis.json`；此前有误差的结果保留在旧 run 目录，不覆盖历史记录。
