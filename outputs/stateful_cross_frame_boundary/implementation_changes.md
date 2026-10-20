# 本轮代码边界与原始文件保护

实际起点 HEAD：`57ddef9c28c69a8ad32f3e78d38aaaca00331b92`。开始时只有 `outputs/m3_state_guided_verified_decode/` 为未跟踪内容；该目录保留，不读取、不恢复其机制。

修改 `tools/worldstream_m123/stateful_opportunity/report.py`：

- 增加独立 `opportunity_gate`，逐项检查历史来源、完整数学闭包、对应 Claim 的状态证据、可避免的原生工作、覆盖状态/验证开销的上界、完整服务意义与可执行验收路径。局部测量正收益不再自动触发 P0 Go；生产资格独立保持关闭。
- 写明 `original_task=planning`、`task_classifier_label=global_safety`、`native_multi_cross_frame_requalification=Not Run` 和 Exploratory 人口。
- 将局部数值差异、Sampling 未认证、Active 任务质量/服务未运行分别记录。
- 拒绝重建到原始旧研究目录。旧报告修订写入本轮 `p0_refinement/`，历史 658 项通过、失败运行及成功回放均保留。

新增独立分析包 `tools/worldstream_m123/large_boundary/`：

- `run.py`：复用现有 `load_native`、冻结 M1、真实数据和原生采样。先执行完整场景的 Native Original/Persistent Full 配对，轮换两种服务的先后顺序；再对同一已曝光人口的两场景短回放进行只读依赖观察和两个详细 Profiler。原生完整任务仍运行，未建立复用分支。
- `profile.py`：临时方法作用域、CUDA Event 与 CPU Wall、逐 Token Forward 记录、EVA/Head Attention/FFN/Projection/Temporal Memory 与 LLM 输出投影的独立明细。按 Trace 中真实设备活动逐条计数，使用唯一 CPU runtime correlation 归属阶段，并与独立设备事件核对；无法可靠归因的真实活动保留为 unattributed，不修补、不缩放。共享 Head 位置模块只包装一次。
- `report.py`：依实际热点排序三个大边界，保留必要依赖反例、旧数值证据、逐请求的宽松零成本 Oracle 和真实状态维护成本。CPU/GPU envelope 与 Kernel/完整 E2E 分开；未运行的 Active、资格成本或 Shape 实验不填零。纯必要输入相同也不能自动升级为完整闭包。
- `tests/worldstream_m123/test_large_boundary.py`：资格缺项/假阳性、未知和非有限上界、状态成本覆盖、嵌套作用域不重复记账、设备/CPU 关联差额、Memcpy 分类、未知关联保留和不可省输入成本测试。

原 `worldstream/m1/`、`worldstream/m2/`、`worldstream/m3/`、`projects/`、配置及模型/LoRA/Tokenizer 不修改。保护文件前后 SHA、参数指针/版本和旧结果 SHA 随每个运行保存。全部临时方法与 Hook 通过退出栈恢复；异常中止时不生成完成或生产通过结论。

服务计时包括输入加载/预处理/Collate/Scatter、完整模型、原生回答写盘和最终同步；轻量粗粒度 Event 观察在两臂一致保留。输出/Memory 摘要与依赖快照在计时外。两个详细 Profiler 不混入服务分位数，Profiler 自身开销单列。

`pilot01/` 为工具验证短回放，CPU FunctionEvent 的关联核对失败原样保留；其原始 Trace 的后处理诊断另存。最终结论只采用 `gpu_run01/` 的完整服务和独立详细 Trace，不把 Pilot 加进人口。

本轮没有新 Runtime、缓存执行器、Kernel、置信度网络或第二套 WorldState。是否进入 P2，以依赖和资格实际证据为准；未经认证 Production/Auto 持续原生完整执行。

最终 `gpu_run01/` 完成 192 个服务请求、2 个详细 Profile 和 8 帧依赖见证；96/96 Native/Persistent 与 8/8 诊断回放的输出/Token/Memory 守卫均逐位一致。设备 Kernel/Transfer 计数与时间核对通过，约 0.0092% 未可靠归属的设备时间明确保留。受保护源码、模型参数版本与旧报告 SHA 均通过前后检查。

GPU 结束后仅完善分析器：将 M1 内部 Wall（可能等待既有 GPU）与逐帧服务增量分开，避免重复收费；另列固定 Host 的 GPU 消除条件估计和更宽松 CPU+GPU Oracle；分析完整性记录执行源码快照与当前报告版本。没有更改已执行的 `run.py`、`profile.py` 或原生加载代码，没有为分析修订重跑或替换实测结果。

完整 GPU 可见回归：**678 passed，0 failed，0 skipped**，保留旧 658 项记录，新增 20 项。原生任务/Stateful Active 质量不是这些单元回归的验收范围；后者继续为 Not Run。
