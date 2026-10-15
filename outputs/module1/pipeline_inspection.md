# 本地 Pipeline 检查（2026-10-15，commit d641ebd）

工作区开始时 clean；README.md 仅 4 bytes，保留。没有发现适用的 AGENTS.md。环境 `/home/qh/miniconda3/envs/OmniDrive/bin/python`：torch 1.13.1+cu117、numpy 1.23.4、scipy 1.8.0、opencv 4.8.1、mmcv 1.6.2、mmdet 2.28.2、mmdet3d 1.0.0rc6，pytest/yaml 可导入。不升级依赖。初次在受限执行上下文中，`nvidia-smi` 失败，`torch.cuda.is_available() == False`，该上下文无 /dev/nvidia*。补查确认宿主驱动正常：驱动 580.173.02，RTX 3090 24GB，OmniDrive CUDA 张量测试通过。前述现象是设备隔离，不是宿主驱动不可用。权重存在：ckpts/iter_10548.pth（19,155,008,764 bytes）、eva02_petr_proj.pth、pretrain_qformer。已有 test/mask_eva_lane_det_vlm/*/pts_bbox/results_nusc.json 仅可作为历史输出回放，不能作为本次模型推理。

## 调用与形状（初始记录为代码/配置推导；末尾补充本次 GPU 实测）

`forward_test` 首次进入重置 Detection/Map Memory，解包 test augmentation → `simple_test` → `extract_img_feat` → `simple_test_pts`（prepare_location/position embedding → Detection forward/get_bboxes → Map forward/get_bboxes → 原 LLM generate）。simple_test_pts 返回 bbox_results, generated_text, lane_results。EVA 将 B×N×3×H×W 展平送入 Backbone，position_level 特征 reshape B×N×C×h×w。默认 stride=16；本地已部署 mask_eva_lane_det_vlm.py 为 EVA 1024 channels / 24 blocks，640×640 输入推导为 40×40 特征；tinyllama 配置为 EVA 768 channels，不存在配套 tinyllama 权重，故优先集成本地已有大模型配置。

Detection Memory：embedding B×memory_len×256；reference_point B×memory_len×3；timestamp B×memory_len×1；egopose B×memory_len×4×4；velo B×memory_len×2；canbus B×can_bus_len×14。pre_update 将历史世界参考点变到当前 Ego；post_update TopK（配置 300）detach 后前插并转回世界坐标。post_update 后长度可为 memory_len+topk，下一次 pre_update 截断。timestamp post_update 减当前绝对秒，pre_update 加当前绝对秒，使用时是年龄；sample_time 类似，间隔>=2s时 refresh。Map 对应参考点 B×memory_len×n_control×3，n_control=11，另有 memory_mask，无 velo/canbus。

Carrier 是独立 num_extra queries 经 decoder/projection：Object 配置 256 个加 canbus token，Map 256 个；不与检测框/地图线逐个对应。部署配置 out_dims=4096，tiny 配置=2048。M1 只记录实际运行时 shape，Carrier 来源为分支级，Transformer dependency unknown/conservative。M1 不读写或重复调用 Head Memory 更新。

## 坐标与输入

CustomNuScenesDataset.get_data_info：ego_pose 明确由 ego2global 四元数/平移生成，为 T_world_from_ego，timestamp 和 img_timestamp 由微秒除 1e6，保留绝对秒（float64 format）。数据按 timestamp 排序，QA/OpenLane 子集有间隔，不能称高频视频。val Info 共 6019 条。

**关键：文件/变量沿用 LiDAR 命名，实际检测/地图参考为 Ego。** update_coords.py 重建 GT boxes/velocity 到 Ego，CustomNuScenesDataset 的 global exporter 已注释 lidar2ego；get_data_info 的 `extrinsics` 用 inv(T_ego_from_camera)，而不是 sensor2lidar。Box wrapper 虽是 LiDARInstance3DBoxes，值的物理参考仍为 Ego，get_bboxes 返回 bottom-center，需要 gravity_center/corners 转世界。Map get_bboxes 输出的是 Ego 下 control points（未调用 Bezier 采样）。

lidar2img 在此路径实际 K×T_camera_from_ego。ResizeCropFlipRotImage 左乘 ida_mat 更新 K；ResizeMultiview3D 再更新 K 后重算投影，故 Collect3D 中对应预处理后的图像。原始 K 可以从 cam_infos 获取；可靠原始→预处理映射为 K_processed×inv(K_raw)，不是经验缩放。Normalize mean/std、to_rgb 由 img_norm_cfg 决定，Pad 不改变 K。各相机分辨率独立处理。

Info 字典顺序为 CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT；代码迭代字典而非固定名称排序。相机各自提供 timestamp、ego2global 和 sensor2ego，可以用相机实际 pose 投影；原模型使用 sample pose+camera extrinsics 近似同步。M1 的 raw replay 使用逐相机 pose，在线集成在扩展配置中通过 Collect3D meta_keys 保留 cam_infos；缺失异步 pose 时超容忍间隔保守失效。

## 数据与命令入口

`data/nuscenes` 是现存真实数据软链；samples、v1.0-trainval 元数据、train/val ego temporal infos 可用。sample.json 可提供 sample→annotation→instance 身份；annotation JSON 用流式读取避免同时加载全部元数据。

原项目入口需要 distributed launcher，tools/test.py 单进程分支 assert False：
`bash tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ckpts/iter_10548.pth 1 --format-only`
原先受限上下文无法访问 CUDA；现在已获准在隔离环境外运行独立 smoke runner，限制帧数、保存感知数值摘要与 M1 日志，不修改原 LLM 的 prompt/生成参数。完整对照状态见 model_smoke/status.json。离线回放仍明确记录历史输出来源和 Oracle 定义。

## 实现完成后的数值检查补充

实际 CPU 图像 Pipeline 已跑通：真实六相机输入经原项目 Load/ResizeCropFlip/Resize/Normalize/Pad 后为 `[1,6,3,640,640]`，没有模型权重计算。最终 `raw_to_preprocessed` 实测为 x×0.4、y×0.8−80（完整矩阵见 real_pipeline_smoke.json）。实际 get_data_info 及其原始变换 helper 的投影与 M1 世界投影通过数值交叉验证；误差只代表代数实现一致性，不代表标定精度。

当前配置继承 Detection `can_bus_len=2`，故 pre_update 截断后为 B×2×14，post_update 前插后可为 B×3×14；主 Memory pre_update 长 600，post_update 可长 900。Object Carrier 配置推导 B×257×4096，Map Carrier B×256×4096，EVA B×6×1024×40×40；上述形状已由后续 RTX 3090 完整模型运行确认，见 model_smoke/m1_state.jsonl；代码捕获接口读取实际运行 shape。

重要数据异常：本地 sample.json 的 anns 列表曾被过滤，首样本列 3 项；sample_annotation.json 通过 sample_token 反查恢复 43 项，和 Ego Info 一致。因此实验 Oracle 使用 annotation reverse links 获取真实 instance_token，不依赖过滤过的 anns 列表。对应回归测试已加入。

新增 tools/worldstream_m1/model_smoke.py 已实现有 CUDA 时的受限连续推理与关闭/开启 M1 的数值摘要比较；初次沙箱阻碍记录保存于 model_smoke/initial_sandbox_status.json；宿主 GPU 已确认可用，当前完整对照状态记录于 model_smoke/status.json。CPU 的实际 Petr3D 方法结合测试替身已验证视觉/感知/Memory/生成输入一致，不冒充完整 GPU 集成验证。

## 宿主 GPU 完整验证补充

2026-10-15 在设备可见的宿主上下文完成三帧 Baseline/Observe-Only：Backbone、Detection/Map 输出及 Temporal Memory 数值摘要完全一致；两次峰值 allocated 显存均为约 15.119 GiB。GPU 访问事实见 gpu_access_check.json，原沙箱观测另存 environment.initial_sandbox.json，不再将设备隔离写成宿主驱动故障。完整结果见 model_smoke/results.json，运行日志 gpu_run.log。
