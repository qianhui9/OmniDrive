# 原始任务依赖与收益可行性审计

实际读取验证集 6019 帧、150 个场景；原始 conv/VQA 文件无替换。
共 84227 条任务出现（含逐帧 Planning/短动作）。完整问题证据见 task_dependency_records.jsonl。

**Gate A：原始 Planning 历史状态替代 No-Go；VQA 局部候选未获得输入完备性证据，B/C 在线开发不启动。D/E 可独立推进。**

以下分类为可复查的词法筛查，不构成任务依赖证明或线上准入。潜在局部候选不能直接计为命中。

| 类别 | 数量 | 比例 | 完整覆盖 | 不完整历史 | 物理事实 / 精度 / 新鲜度 |
|---|---:|---:|---|---|---|
| global_planning | 6019 | 7.1462% | yes | no | complete current agents, free space, signals, route, ego dynamics; trajectory quality unchanged; no local surrogate; current request; dynamic agents need current motion |
| global_safety_scene | 49258 | 58.4824% | yes | no | all relevant hazards, scene semantics, signs, lighting, intent; complete safety/semantic evidence absent from local anchors; current request |
| count_absence_existence | 14580 | 17.3104% | yes (for arbitrary yes/no and exact counts) | no | complete requested spatial domain and identity deduplication; no false negatives; partial history cannot prove absence; current request |
| local_spatial_relation | 1551 | 1.8415% | object grounding required; full global scene not intrinsically required | conditional | unique referent, identities and signed relative direction interval; signed interval must exclude zero; verify at request time; age plus motion error bounded |
| local_distance_position | 209 | 0.2481% | object grounding required; nearest selection needs domain coverage | conditional | unique referent, independent metric localization error bound, ego pose; frozen prior direct numeric error <= 0.5 m; detector score is insufficient; verify at request time |
| local_motion | 404 | 0.4797% | unique object grounding and temporal evidence required | conditional | unique referent, time baseline, independently supported object motion and ego motion; independent velocity interval excludes decision boundary; no new tolerance frozen; current velocity relation with bounded lag |
| local_map_candidate | 2717 | 3.2258% | local domain and map semantic type required | conditional | local lane geometry, topology/marking semantics when requested; lane_control_points alone do not certify topology, markings or traffic rules; static map plus current ego pose and current closure observations |
| other_visual_semantics | 9489 | 11.2660% | question specific; unresolved | no established substitution | appearance, text, buildings, weather or unconstrained language semantics; not represented in current state interface; current request |

原始 Planning 6019 条：可合法替代覆盖 0%，强制 Full Visual 100%，可合法跳过 Prefill Token 上限 0。
conv/VQA 筛查候选 4881/72189；已建立完整依赖映射 0。不能把未解决的问题报告为已证明不可能。
当前状态只有类别、几何、未认证速度和 lane_control_points；没有问题指称到唯一目标的独立映射、完整搜索域、外观/标志语义或独立测距误差界。当前 Head 一致性也不是物理真值。
局部方向在有独立前景归属、身份、投影符号证据时可能无需完整 3D；本轮没有这样的独立认证证据。
人工复核确认 2 条原始距离问题有条件的局部事实依赖（原文与 sample_token 保存在 JSON），允许最小离线 B 诊断。它们仍缺少唯一目标映射与测距误差界，不能在线替代，也不能作为 Planning 主路径。
理论收益文件保留原生 Prefill/Decode 实测逐请求值，并单列删除全部语言工作的不可达上限；所有上限均不是实际加速。
按用户要求不改写 M3 Prompt/Carrier，不强制制造状态命中；保留严格回退与旧风险协议。

复现：`taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.task_dependency_audit`
