#!/usr/bin/env bash
# 已执行命令记录。回放目录已存在，重新运行请换 --output，脚本会拒绝覆盖状态。
M1_PY=/home/qh/miniconda3/envs/OmniDrive/bin/python
M1_PRED=test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json
$M1_PY -m tools.worldstream_m1.inspect_pipeline
$M1_PY -m tools.worldstream_m1.pipeline_smoke
$M1_PY -m tools.worldstream_m1.model_smoke --compare
# 初次沙箱执行因设备不可见退出；后续在获准的宿主上下文成功运行相同三帧对照。
# 结果见 model_smoke/results.json；复跑请为 model_smoke 指定新的 --output。
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --output outputs/module1/replay_strict
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --experimental-appearance --output outputs/module1/replay_appearance_ablation
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --descriptor gradient --output outputs/module1/replay_gradient
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --stride 2 --output outputs/module1/replay_interval_1s
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 12 --scene-index 1 --output outputs/module1/replay_second_scene
$M1_PY -m tools.worldstream_m1.run_streaming --infos data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl --source annotations --frames 8 --split calibration --output outputs/module1/calibration_train
$M1_PY -m tools.worldstream_m1.map_demo --frames 6
$M1_PY -m tools.worldstream_m1.calibrate --run-dir outputs/module1/calibration_train --output outputs/module1/calibration.json
$M1_PY -m tools.worldstream_m1.evaluate --run-dir outputs/module1/replay_strict
$M1_PY -m tools.worldstream_m1.visualize --run-dir outputs/module1/replay_strict
$M1_PY -m tools.worldstream_m1.profile --predictions "$M1_PRED" --repeats 5
$M1_PY -m tools.worldstream_m1.plot_results
$M1_PY -m pytest -q tests/worldstream_m1 --disable-warnings --junitxml=outputs/module1/unit_tests.xml
$M1_PY -m tools.worldstream_m1.summarize
