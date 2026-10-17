# 插桩与代码变更清单

旧模块实验不改写；profile_stages=False 关闭详细插桩；不配置 stream_config 保留原入口。
时间类型与开销边界详见 docs/worldstream_system_optimization.md；实际运行版本见 delivery_source_audit.json。

| 文件 | 类型 | 首个修改行 |
|---|---|---:|
| [docs/worldstream_system_optimization.md](../../docs/worldstream_system_optimization.md) | created | 1 |
| [projects/mmdet3d_plugin/models/detectors/petr3d.py](../../projects/mmdet3d_plugin/models/detectors/petr3d.py) | modified | 76 |
| [tests/worldstream_m2/test_cheap_admission.py](../../tests/worldstream_m2/test_cheap_admission.py) | modified | 27 |
| [tests/worldstream_m2/test_computation_source.py](../../tests/worldstream_m2/test_computation_source.py) | created | 1 |
| [tests/worldstream_m3/test_adaptive_batch.py](../../tests/worldstream_m3/test_adaptive_batch.py) | created | 1 |
| [tests/worldstream_m3/test_refinement.py](../../tests/worldstream_m3/test_refinement.py) | modified | 85 |
| [tests/worldstream_m3/test_system_reporting.py](../../tests/worldstream_m3/test_system_reporting.py) | created | 1 |
| [tests/worldstream_m3/test_system_runtime.py](../../tests/worldstream_m3/test_system_runtime.py) | created | 1 |
| [tools/system_optimization/__init__.py](../../tools/system_optimization/__init__.py) | created | 1 |
| [tools/system_optimization/audit.py](../../tools/system_optimization/audit.py) | created | 1 |
| [tools/system_optimization/decode_probe.py](../../tools/system_optimization/decode_probe.py) | created | 1 |
| [tools/system_optimization/delivery.py](../../tools/system_optimization/delivery.py) | created | 1 |
| [tools/system_optimization/experiment.py](../../tools/system_optimization/experiment.py) | created | 1 |
| [tools/system_optimization/gate_benchmark.py](../../tools/system_optimization/gate_benchmark.py) | created | 1 |
| [tools/system_optimization/lifecycle_probe.py](../../tools/system_optimization/lifecycle_probe.py) | created | 1 |
| [tools/system_optimization/report.py](../../tools/system_optimization/report.py) | created | 1 |
| [tools/system_optimization/run.sh](../../tools/system_optimization/run.sh) | created | 1 |
| [tools/system_optimization/wait_gpu.py](../../tools/system_optimization/wait_gpu.py) | created | 1 |
| [worldstream/execution.py](../../worldstream/execution.py) | created | 1 |
| [worldstream/m1/provenance.py](../../worldstream/m1/provenance.py) | modified | 1 |
| [worldstream/m1/state_manager.py](../../worldstream/m1/state_manager.py) | modified | 296 |
| [worldstream/m1/state_store.py](../../worldstream/m1/state_store.py) | modified | 55 |
| [worldstream/m1/state_types.py](../../worldstream/m1/state_types.py) | modified | 89 |
| [worldstream/m2/camera_executor.py](../../worldstream/m2/camera_executor.py) | modified | 26 |
| [worldstream/m2/camera_feature_cache.py](../../worldstream/m2/camera_feature_cache.py) | modified | 13 |
| [worldstream/m2/cheap_admission.py](../../worldstream/m2/cheap_admission.py) | modified | 14 |
| [worldstream/m2/runtime.py](../../worldstream/m2/runtime.py) | modified | 41 |
| [worldstream/m3/batched_executor.py](../../worldstream/m3/batched_executor.py) | modified | 90 |
| [worldstream/m3/config.py](../../worldstream/m3/config.py) | modified | 21 |
| [worldstream/m3/runtime.py](../../worldstream/m3/runtime.py) | modified | 120 |
| [worldstream/m3/world_state.py](../../worldstream/m3/world_state.py) | modified | 41 |
| [worldstream/runtime.py](../../worldstream/runtime.py) | created | 1 |
