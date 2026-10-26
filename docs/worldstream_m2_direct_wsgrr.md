# M2 正式 WSGRR 执行契约

正式配置：`projects/configs/OmniDrive/worldstream_m123_auto.py`。
M1/M2/M3 均开启，M2 普通刷新预算为每相机 3 个窗口，最大复用年龄为 8 帧。
M3 继续使用现有 `wsdr_kv`、质量预算和策略文件。

## 一条计算路线

`M1 Precheck → forward_from_m1 → Block 0 输入 → WSGRR → DependencyPlanner.plan_windows → WindowExecutor → StateCache.commit → Detection/Map → M1 Commit → M3`。

首帧和场景切换使用同一个 WindowExecutor 更新全部窗口并建立缓存。连续帧直接执行依赖闭包中的窗口；未执行窗口保留实际计算来源、版本和年龄。全局依赖后的 EVA Block 正常完整执行。

M2Runtime 不读取、不实例化 CostModel。正式配置和 M2Config 已移除 `cost_table` 与 `cache_capture`。无论是否存在成本记录、批准数量或质量报告，成功计算都提交缓存。旧字段会作为未知参数报错，使用旧配置的调用方应删除它们。

成本/批准工具仅用于历史分析。`worker --m2-cost-table` 只绑定可选分析元数据，不注入运行时配置。正式验收无需成本文件，核查实际窗口、缓存发布、M1 快照和计划执行一致性。

D3 验证入口是正式算法的别名，共享缓存签名和版本。A/B/C/NoM1 消融选择代码位于 `tools/worldstream_m2/offline_variants.py`，所有 GPU 执行仍经过同一 `_forward/_execute`。不要将消融缓存用于正式服务。

## 错误语义

保留模型拓扑、参数/数值策略、预处理几何、相机顺序、场景/帧/时间、M1 epoch 与 source-linked 计算来源检查。
场景切换重置历史；同场景不兼容状态、篡改缓存、缺失 M1 快照、非有限输入/输出、发布失败或显存不足直接抛异常，没有备用执行器或静默重试。

M1 明确全帧刷新以及局部计算来源过期仍构成必须刷新输入。强制刷新不能掩盖错误来源或非法当前状态。StateCache 在全部所有权拷贝、容量和有限值检查成功后才原子替换，失败时不将请求标记为完成。

## 诊断与测量

运行时输出 `total_windows/planned_windows/executed_windows/reused_windows/actual_skipped_windows`、`cache_saved/cache_version/full_refresh_reason`、必须/普通窗口、按相机与 Block 统计，以及 `M2_execution_ms/M2_cache_publication_ms/M2_total_ms`。

窗口分母是可缓存前缀 Block 的真实输出窗口；正式 EVA 为 Block 0、1。全局复用率为实际复用数之和除以总窗口数之和。另行报告发生任何复用的帧比例。二者均不能代表视觉或端到端加速。

运行时毫秒字段为 CPU wall/提交时间，不是独立 CUDA kernel 时间。基准使用同步 wall 和 CUDA event span；CUDA span 包含主机提交间隙，不能标为 GPU kernel active time。

## 本次可复现证据

`outputs/m2_unified_wsgrr_20261026_run01/experiment_report.md` 包含正式 40 帧重放、旧 D3 配对、完整模型 40 帧任务比较、测试、原始命令和性能瓶颈。原始历史结果未覆盖。

该短场景提供联合 M1/M2/M3 的实际任务输出比较，不能作为全验证集精度认证。M2 改变视觉输入后，旧 M3 在全量 M2 输入上的独立质量结论不能直接沿用；部署质量还需要多场景联合验证。
