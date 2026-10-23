最终状态：本轮代码修改、测试、一次最终完整验证和重复请求计时全部完成。没有待运行的本任务实验。

核心代码仍与 final_source_manifest.json 一致；final_evidence.json 为 PASS，固定 checkpoint SHA 为 6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d。生产批准表、正式配置和历史输出未修改。

GPU 回归 333 passed（tests_gpu_final_boundaries.log）；真实 EVA 10 个执行器数值案例与历史完全一致；最终 40 帧含 39 个连续帧，486 项缓存审计全部通过。

最终正式评估 main_full/official_eval 已完成 6019 帧、150 场景：mAP 0.375755，NDS 0.470092，相对 Native 下降 0.4351%、0.2652%。DET_l 0.132247，3D Lane Fscore 0.080970。

全量窗口更新率88.0028%，复用11.9972%。完整集已识别必刷集合无执行遗漏、普通预算无超限、缓存发布无失败、无未来来源时间戳。以上不是真实世界变化漏检率或身份认证。

最终视觉 P50：Native486.8778ms、主方案492.5912ms；完整请求40帧两次重复 pooled P50：Native1714.2446ms、主方案2060.3634ms。视觉慢约1.17%，完整请求慢约20.19%；没有净加速，生产质量接受性未认证。

最终报告 experiment_report.md；机器汇总 summary.json；逐场景 paired_scenes.json；异常窗口 anomaly_window_trace.json；实际预测和GT异常 anomaly_prediction_diagnostics.json；四类残余案例 residual_detection_cases.json 及全部GT状态变化 residual_detection_gt_changes.json。

开发阶段未完成的1220帧没有正式评分，未用于最终结果；相关开发目录已按用户后续清理要求删除，见 cleanup_manifest.json。开发失败和中断的简要记录见 attempts.json。

运行约束：所有命令 taskset -c 0-3,6-31；Python /home/qh/miniconda3/envs/OmniDrive/bin/python；其他用户GPU任务只等待不停止。后续用户授权清理其他实验输出，清理范围与保留项见 cleanup_manifest.json。
