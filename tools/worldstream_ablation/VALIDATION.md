# 启动验证记录

最终版本验证完成于 2026-11-16（Asia/Shanghai）。本次执行启动测试和数据检查；完整消融任务由用户随后使用 `run.sh` 启动。

## CPU 与正式 R0 回归

50项测试全部通过：21项新增机制检查，29项已有正式R0回归。新增检查包括真实小型Transformer的KV复用/当前Decode、独立候选验证、答案缓存、三种状态维护、双向光流和局部残差、连续窗口刷新、硬预算/最大年龄、查询对象限定的否定答案评分，以及**实际执行MACs与PyTorch Profiler算子记录精确对照**。

命令和结果保存在 [cpu_validation.json](cpu_validation.json)。Python语法及Bash语法检查通过。正式 `worldstream/`、`projects/`、R0参数和权重没有修改。

## 全部14组真实GPU推理

最终命令：

```bash
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --smoke --output outputs/worldstream_ablation_smoke_run02
```

使用 OmniDrive 环境和 RTX 3090，严格核验并加载项目真实checkpoint。每组首个Scene的3个连续帧、每帧5个问题；合计 **14组 / 42次真实帧推理 / 210个问答请求**，全部完成。Smoke只共享一次严格权重加载，各组重置模块缓存及原生任务头；正式完整任务仍为每个配置启动独立进程。

- 最终总完成标记：[completion.json](../../outputs/worldstream_ablation_smoke_run02/completion.json)，`status=COMPLETED`、`smoke=true`。
- 四张表的导出验证：[ABLATION_REPORT.md](../../outputs/worldstream_ablation_smoke_run02/ABLATION_REPORT.md)；另有module/ogsm/davp/qcse四个CSV。
- 每组原始回答、实际计算量、CUDA时延、逐问题评分、有效配置与独立完成标记，均在对应子目录。
- 源码、配置、原始图像、QA、标注和预训练资产哈希均通过运行后核验；原始推理收据哈希核验通过。
- `without_ogsm` 的全部15个请求均未采用事实直答，保留当前目标模型执行与候选验证。
- Periodic与Feature-Drift的两次热帧，每个摄像头均真实执行3个前缀窗口；最大复用年龄另由持续50帧的机制测试检查。
- Prompt/KV共出现6次重复固定查询的文本前缀KV命中，全部仍执行当前Decode，未复用视觉KV。
- CPU亲和性为0–3、6–31，未包含4/5。

**这些仅证明启动、机制及汇总路径可运行，不能作为论文性能结论。** 首个Scene仅3帧，没有可评分GT答案语义变化机会，因此Stale Answer Rate为N/A，未伪填0%。Smoke的共享权重加载与执行顺序也不适合作为正式时延对比。

开发期 `worldstream_ablation_debug_module01`、`worldstream_ablation_smoke_run01` 仅保留排错记录，不是最终验证结果。最终证据统一以上述run02为准。

## 完整任务数据入口

已独立检查默认18个完整Scene的全部723帧、六相机图像、conv/vqa文件及精确问题负载，时间链接连续，输入文件齐全。每配置3615个查询；14组完整任务预计执行 **10,122次视觉帧调用 / 50,610个问答请求**。

数据检查结果见 [full_data_validation.json](full_data_validation.json)。完整任务尚未执行，不能按这些计划数量推定最终成功；运行后以新输出根目录的`completion.json`为准。

## 任务包

`worldstream_ablation_task_bundle.tar.gz` 打包本目录的代码、启动脚本、实验协议、配置矩阵和验证说明；不包含模型权重、大型数据或旧实验回答。解压到项目的`tools/`目录即可恢复本任务包。`FILES.sha256`列出包内源文件与说明的校验值。
