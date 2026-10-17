# 待原 formal GPU 进程退出后执行

当前 formal 进程已加载的 `worldstream/runtime.py` 不得中途改动，以保持所有运行源码快照与已加载实现一致。

原正式全流程（包括 late-import 的 v2_probes、long_state_supply、batch_probes）以及物理 Oracle/report/GPU 回归完成后：

1. 在 `UnifiedRuntime.workload_key` 的 strategy.runtime 中增加真实 policy：profile_stages、m1_timeout_s、capture_cost_key、runtime trace logging 开关和 filesystem device。这些影响成本/故障契约，不能静默继承证书。同时将 identity、m1_signature_suffix 和 model.simple_test_pts 的 compiled code 加入实现指纹（可用 getattr，兼容测试 doubles），保护生成调用/签名依赖。
2. 修正 experiment.py protocol 的 audit-copy 文字：只有共同 Token-ID CPU 审计捕获在 E2E 内；停止计时后的 output/feature/memory NumPy 拷贝、M1 digest/snapshots、Oracle、Profiler 不在 E2E 内。实际时延和旧 protocol 保留，已有 timing_scope_correction.json 说明真实范围。
3. 使用新实现运行 `python -m tools.system_optimization.v2_cost_refresh`，16 个同样预先选定主场景帧比较 m1/opt_serial/overlap（另有 native refs），新 Auto 场景固定 114–115 各 4 帧。老证书自动归档到 initial_cost_certificates，不覆盖旧源运行。不能用旧 implementation hash 声称新版已测。
4. v2_report 需要读 cost_refresh / cost_refresh_auto 两个完成目录，将 initial auto_validation 的证书标为 retired_initial，仅新精确键匹配者 production_auto_certificate=True。配对/状态质量/Oracle仍保留所有真实数据，重复主帧不增加独立样本数。
5. 重新物理 Oracle/report、最终 GPU 回归、delivery_v2 完整重哈希、diff/compile/bash 检查。模型未修改，旧九份结果的 SHA 必须保持。

新增补充协议的场景和检查点在得到相关结果前已固定。long_state_supply 使用原四个长场景第 8/16/32 帧的真实五问题，主流保持完整 M1 历史；额外 LM 请求属于独立混合负载，不计入常规完整帧速度表。native_decode probes 已增加实际 KV 形状/bytes/storage-growth 记录、独立 resident-memory 采样控制。所有新 helper 均在 formal 结束时 late import，不修改已完成的测量。
