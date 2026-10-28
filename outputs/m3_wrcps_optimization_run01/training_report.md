# 训练结果

状态：NOT_STARTED_STAGE_ONE_FAILED。真实48帧均无可用认证稳定事实；未用虚假稳定标注或重复理想场景启动训练。Adapter checkpoint与权重Hash不存在，训练步数为0，训练/验证数据范围为空。`training_config.json`是预定配置，不是已执行训练记录。

已实现 `tools/worldstream_m123/wrcps_train.py`：冻结原生视觉/Heads/LLM，只训练Composer。使用答案分布KL + 原始答案CE，运动/地图/全场景关键问题CE权重为2。所有Object/Map当前特征参与动态压缩，Ego原样保留。稳定时间一致性由Anchor事实独立编码保证。未增加LoRA，未修改Teacher权重。

输入契约：manifest包含 scene_splits、base_weights_sha256、nuscenes_scene_table、samples。scene_splits显式区分train/validation/calibration/independent_acceptance；只有官方nuScenes训练场景可用于Adapter train/validation。每个sample包含scene_id、frame_id、capture及SHA、当前stable_snapshot、原始question_source及SHA、pairs（question_index/question/answer）。Capture必须由真实当前帧生成，包含carriers/ids/questions/stable_snapshot/stable_context；不可把验收问题答案作为训练标签。预检拒绝无真实跨帧稳定Anchor的数据，训练时再次核对快照与capture一致。

`benchmark.worker --dataset-split train --capture-m3-inputs --capture-stable-world --m3-zero-reuse --m3-strategy exact_prefix --condition B3`提供原始布局训练capture接口；必须提供连续训练帧sequence-file，禁止把该运行当作验收结果。当前没有足够稳定状态生成合法训练manifest，故没有运行训练或伪造manifest。

完成稳定性证据之后的训练入口：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.wrcps_train --manifest <真实训练清单.json> --output <新的训练目录> --preflight-only
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.wrcps_train --manifest <真实训练清单.json> --output <另一个新的训练目录> --stable-tokens 128 --rank 128 --seed 20261028
```

以上是带前置条件的接口说明，本轮未执行，不是成功训练的复现命令。7B模型冻结梯度训练仍有激活显存成本；本轮未验证完整训练在24GB上的峰值显存或吞吐量。
