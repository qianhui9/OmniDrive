# WSDR-KV：任务与 group 变化感知的严格 KV 候选规划

正式入口仍为 `projects/configs/OmniDrive/worldstream_m123_auto.py`，默认 `M3Config.reuse_strategy='exact_prefix'`。WSDR-KV 通过显式实验参数启用，不修改部署批准文件，也不授权未经验证的生产复用。

## 当前接口能够证明的内容

`Petr3D.simple_test_pts` 将 Detection head 的 `vlm_memory` 与 Map head 的 `vlm_memory` 拼接。真实模型的边界是 object 256、ego 1、map 256，总计 513 个 carrier token。object 是额外学习查询的输出，并非 256 个逐对象检测槽位；ego 是 `can_bus_embed`，包含当前/历史运动与位姿信息；map 也是独立的额外查询输出。

EVA 窗口经过 Block 2 起的全局后缀，以及 Detection/Map 稠密注意力和时序记忆，才形成这些 carrier。现有接口没有可靠的窗口到 carrier、anchor ID 到 object query 或地图区域到 map query 的对应表。因此本实现始终报告 `m2_mapping_available=False`、`m2_mapping_reason='m2_region_mapping_unavailable'`，不推导虚假的区域级 KV 区段。M2 只提供当前 scene/frame/time 的执行上下文一致性检查；组是否稳定最终通过实际扩展 Embedding 精确比较判断。

M1 使用 Commit 之前冻结的 `VerifiedWorldState` 和现有 `query_world_state()`：检查快照摘要、来源、帧、时间、epoch、状态版本和执行签名。有效空快照不表示场景完整；普通观测支持不等于物理身份认证。历史状态只用于任务候选与失效判断，不能签发 KV 复用。动态历史文本仍须独立质量准入，插入位置保持在 image marker 之后，没有将动态 JSON 移到视觉前缀之前。

## 规划、布局与实际执行

`reuse_planner.py` 使用 `task_conditioner.task_profile()` 对既有分类结果细化依赖优先级：object 属性、map 关系、object+ego 运动，以及完整视觉范围。全局安全、未知、计数与缺失问题使用完整范围。优先级不删除任何当前 carrier。

规划器对缓存中拥有的 ExpandedInput 分组视图与当前实际展开结果进行精确比较，给出 stable/changed/mandatory groups，以及稳定组优先、任务相关性和原始组顺序作为 tie-break 的候选顺序。它不保存另一份大型 carrier 缓冲。

当前 OmniDrive 使用学习过的固定顺序，没有经过验证的组可交换契约，所以实际 `layout_permutation` 保持 object→ego→map，并报告 `native_group_order_unverified`。`candidate_group_order` 只是建议，不能声称已执行重排或提高匹配率。对于另一个明确声明 `CarrierLayout.order_contract='group_permutation_v1'` 的载体生产者，代码支持在 carrier 组织阶段完整重排后重新调用 native expansion；该声明不是质量批准，OmniDrive 的正式调用不提供它。受控测试会检测这种排列对完整模型 logits 的改变。

`ExpandedInput` 增加 layout signature 和 visual group spans。Token ID、Embedding、position ID、attention mask、visual mask 均仍由实际 native expansion 确定。布局、策略版本、模型/LoRA/后端和场景作用域纳入缓存兼容；缓存元数据修改会清空旧缓存。严格 LCP 不允许跳过已变化的前缀，再复用后方不连续的 KV。

仍只调用原有 `PrefillExecutor`、不可变 PrefixKVCache 和原生 Decode。匹配长度、批准长度与实际使用长度分别记录；短输入命中旧长缓存时仍重新计算最后一个 token 的 logits，批准证据必须绑定调整后的实际复用长度。

## 独立准入与诊断

新策略只接受 strategy=`wsdr_kv`、version=`wsdr_kv_group_v1` 的批准。不会迁移 `exact_prefill` 的批准。证据绑定模型/权重 SHA、部署键、策略版本、任务类型/profile、实际布局签名、输入长度、候选与实际复用长度、dtype 和生成参数哈希，并要求真实数值一致性、独立任务质量和逐样本净成本。外部 evidence/protocol 文件需要真实哈希；`wsdr_kv_evidence_v1` 的 binding、数值检查和测量内容也与批准记录一致。缺任何证据均执行完整 Prefill。

新增请求诊断包括策略、任务、布局、各组状态、M1 有效性、M2 映射缺失、匹配/批准/实际复用、缓存失效与布局退化原因，以及准备、匹配、分支、Prefill、Decode 时延。普通推理不额外调用 `torch.cuda.synchronize()`；WSDR 规划需要的三个组稳定性布尔值一次回传 CPU。未采集的 GPU 时延为 null，已有 CPU 入队时间保留明确名称。`end_to_end_ms` 是 M3 单问题 host wall；完整系统请求时延读取 benchmark 的 `request_ms`，不可混同。GPU Event span 含 CPU 提交间隙，嵌套范围不可相加。

## 最小真实验证

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -u \
  -m tools.worldstream_m123.benchmark.prefill_validate --wsdr-comparison \
  --frames 4 --questions-per-frame 3 --warmup 1 \
  --checkpoint ckpts/iter_10548.pth --output outputs/wsdr_comparison_new
```

该模式复用现有 worker，只运行 native_full、exact_prefix、wsdr_kv 三组。M1/M2、连续帧、真实问题、预处理、权重和生成参数相同；native_full 显式关闭自定义 M3 并保留原生完整 Prefill。每帧三个语言问题，整帧 request_ms 包含三次生成。正式入口本身不切换策略。

WSDR worker 在全部服务时延及显存样本之后运行两个独立数值探针：基于真实最后一帧 carrier，分别改变 map 或首个 object token。它们明确标记 controlled/natural_frame=False，通过已有执行器直接检查候选 KV 的数值性质，不能冒充自然帧复用或任务质量。`wsdr_candidates.json` 始终 production_ready=False、approved=[]，不会写正式批准文件。

自然连续帧如果没有“部分组精确稳定”的样本，报告必须说明覆盖缺口；不能人为冻结组后将它当作自然推理。详细本轮结果见 [实验报告](../outputs/m3_wsdr_kv_20261025_run01/experiment_report.md)。

worker 的 `--question-offset N` 可从已有带标签的 Conv/VQA 问题序列中确定性选择另一个起点，默认 0 保持原选择；不读取答案来筛选输入。本轮额外以 `--limit 2 --questions-per-frame 2 --question-offset 10` 对三种配置补测真实数字参考问题，结果单独保存于 `numeric_service/`，不与四帧主比较混算。
