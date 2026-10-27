"""Measured M3 recovery report; no unobserved speedup or policy promotion."""
from pathlib import Path
import re
from mmcv import Config
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m2.global_residual_benchmark import rows,read
from tools.worldstream_m123.final_speed_report import table,dist,number


def report(root,r):
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    decision=dict(policy_unchanged=r['policy_unchanged'],formal_m3=dict(cfg.model.m3_config),formal_m2=dict(cfg.model.m2_config),
        new_rules_admitted=False,single_formal_route=True,system_quality_pass=r['system_quality_pass'],m3_quality_pass=r['m3_quality_pass'],
        cross_scene_e2e_gain_proven=r['cross_scene_e2e_gain_proven'],
        reason='Retain one frozen policy and existing AR2. Workload/evidence restrictions explain zero KV; no relaxation, new executor or online selection.')
    if r.get('manual_semantic_review'):
        decision.update(candidate_rejected_semantic_review=True,reason='Automatic lexical proxies missed new unsupported road-rule claims; disable the implicated approximate rule and verify a restricted policy on fresh scenes. This frozen candidate is not the final qualified deployment.')
    write(root/'frozen_candidate_deployment_decision.json',decision)
    if not (root/'qualification/deployment_decision.json').exists():write(root/'deployment_decision.json',decision)
    original=r['rejection']['original48'];hist=r['historical_exact_inputs'];before=r['historical_current_pipeline']
    out=['# M3 WSDR-KV 冻结历史策略诊断与验收（最终部署见 FINAL_REPORT.md）',
        f'完成真实 `ckpts/iter_10548.pth` 实验。当前正式代码在历史工作负载重现 `{before["counts"]["skipped_tokens"]}` Tokens 复用；旧输入隔离回放为 `{hist["reused_tokens"]}` Tokens、`{hist["answers_equal"]}/{hist["requests"]}` 回答与相同视觉的 Native 一致。最新单问题流的零复用主要由工作负载与原有质量/成本证据解释，未发现 PrefillExecutor 丢弃获准前缀的回归。',
        '## 真实基线与代码差异',
        '`baseline_evidence.json` 对历史 4522d3b6 与任务开始 main 09e258d3 逐文件比较，`historical_source.diff` 保存实际差异。Runtime、Planner、Scheduler、Matcher、VisualState、TaskConditioner、PrefixKVCache 和策略字节一致；PrefillExecutor 仅改变 Decode position_ids 的准备方式：预计算 prompt unmasked count，加上生成步偏移。没有替换 Prefill、KV 分支、因果 Mask 或最后 Token logits。',
        f'历史连续 24 帧×3 个原标注问题、顺序、生成设置、checkpoint 和策略保持，当前整个 M1/M2/M3 管线复用分解为 `{before["counts"]["groups"]}`。当前整个管线与旧管线答案相同 `{before["answers_equal_to_old_pipeline"]}/72`，不能把不同 M1/M2 视觉输出的答案差异归因于 M3。为隔离这一点，另使用历史 native_reference 的实际 Carriers、IDs、逐问题 RNG、M1 Snapshot、M2 Receipt、原生布局和生成设置，只运行当前 M3；全部 Native 原始 Token 已逐项复现。',
        f'旧输入隔离回放：平均 GT F1 下降 `{hist["mean_f1_loss"]:.6f}`，新增关键代理错误 `{hist["critical_regressions"]}`。隔离回放不含当前图像读取、Detection/Map 或 M1/M2，绝不作为完整系统 E2E 时延。',
        '## 为什么原始 48 问的 42 个热缓存请求仍复用零 Token',
        table(['原始任务 Profile','问题数'],original['profiles'].items()),
        table(['逐请求首个 Exact 候选拒绝条件','请求数'],original['first_candidate_reasons'].items()),
        table(['请求至少一个候选遇到的条件（允许重叠）','请求数'],original['requests_with_each_candidate_rejection'].items()),
        '6 次冷启动之外，42 个请求确有至少 35 Tokens 的精确因果前缀。37 个 Full Visual 的 Exact 首先因原策略 `critical_regressions=1` 被拒，其近似规则没有质量校准记录；5 个 Object Attributes 的 Exact 因 `quality_loss_upper=0.025974025974025983 > 0.02` 被拒。Object Attributes 跨帧近似候选还会遇到风险、573–576 输入长度范围或最少 558 复用 Tokens 等限制。普通新帧最多复用前 128 个视觉 Tokens（加 35 个文本 Tokens，共 163），不能满足 558 的测量下限；已有规则主要覆盖同帧或继承近似的长前缀。Full Visual 的 35 Token 文本前缀仅代入尚未获准的旧成本式，35×.261811−16.048986−实际开销也为负；有 Exact 前缀本身不等于有净收益。',
        '逐候选首次拒绝数值、完整策略规则、未获准的成本预测、每视觉 Block RMS/最大变化、实际 Prefix/Forward 长度与源帧见 `original48_rejection_details.jsonl`。这是原 Risk0 已记录实际统计的 CPU 重放，不执行 KV，不构造虚假命中。`single_before_rejection_details.jsonl` 是当前正式 AR2 下相同问题的真实重跑；`single_after_diagnostic_rejection_details.jsonl` 另记录真实 KV Tensor _version、存储 revision SHA、来源帧/帧索引/epoch。所有被拒请求仍执行完整 Prefill，没有 Executor 违背规划结果的证据。',
        '## 本次正式代码修改',
        'Scheduler 的同一套证据谓词现在同时支持选择与离线首次拒绝解释。正式 choose 不分配候选日志列表。Planner 只在跨帧、M1 连续性有效且近似规则有合格质量证据时计算逐视觉 Block RMS；同帧、Exact-only 或缺失/不合格近似规则只计算原 Exact/Legal Prefix。Prefix 的 Embedding、Token、Position、Mask、Finite 与缓存生命周期验证保留。阈值、成本长度区间、风险上限、深度和年龄约束均未放宽。',
        'TaskConditioner 仅保守收紧：整体区域描述、通行空间、停车态势和驾驶动作响应不再仅因出现 vehicle 等词就归为 Object Attributes。没有把安全、数量、运动或复杂场景问题提升为 Map。原单问题流本身缺少可批准 Map Relations；零 Map 不是通过错误忽略问题制造的。策略未重新拟合，未新增放行规则，因而不使用失败的验证场景调阈值。',
        f'真实相同输入/缓存/M1/M2/任务 Profile 控制：`{r["same_input_control"]}`。`planner_equivalence.json` 使用实际观测调度开销固定成本项，只隔离统计删除，比较冻结旧 Planner 与当前决策、风险、来源和年龄。统计微基准见下表。',
        table(['真实相邻输入 Prefix 统计','Wall mean/P50/P95 ms','CUDA span mean/P50/P95 ms'],[(k,dist(v['wall_ms']),dist(v['cuda_ms'])) for k,v in r['microbenchmark']['summary'].items()]),
        f'Profiler 实际操作计数：`{r["microbenchmark"]["kernel_counts"]}`。active_kernel_total 为单次 Profiler 采样的实际 CUDA Kernel 总时间；微基准 CUDA span 还含 host 提交空隙，两者分别解释。额外 GPU 标签、旧 Planner 检查和日志探针只在离线诊断执行，整段 `same_input_control` 不参与 headline 性能；不将统计的微秒级收益夸大为系统加速。',
        '## 自然单问题与多问题流的机会和质量',
        '单问题保留上一轮 6 场景×8 帧、48 个原问题及顺序；多问题使用未参与历史 M3 拟合与此前测试的 4 个新场景×6 帧，每帧 3 个原生标注问题。场景按输入自车位移分位冻结；多问题选择沿用预先冻结的任务分布平衡规则，问题名称显式写入 Manifest，不人工重复问题，也不改变原生 object→ego→map 布局。多问题收益不能外推到单问题流。',
        table(['工作负载','实际 Profiles','KV 分类：请求数/Token','总 Prefill Token','实际 Prefill Token','跳过比例'],[(w,str(v['summaries']['free_reuse']['counts']['profiles']),str(v['summaries']['free_reuse']['counts']['groups']),v['summaries']['free_reuse']['counts']['input_tokens'],v['summaries']['free_reuse']['counts']['effective_prefill_tokens'],f'{v["summaries"]["free_reuse"]["counts"]["skipped_fraction"]:.3%}') for w,v in r['workloads'].items()]),
        table(['M3 相对相同 M1/M2 零复用','自然答案相同','平均 F1 绝对下降','最大单请求下降','新增关键代理错误'],[(w,f'{v["m3_quality"]["answers_equal"]}/{v["m3_quality"]["requests"]}',number(v['m3_quality']['mean_f1_loss']),number(v['m3_quality']['max_request_f1_loss']),v['m3_quality']['critical_regressions']) for w,v in r['workloads'].items()]),
        table(['仅实际复用请求','请求数','平均 F1 绝对下降','质量上界','最大请求下降','新增关键代理错误'],[(w,v['m3_quality']['reused_requests'],number(v['m3_quality']['reused_mean_f1_loss']),number(v['m3_quality']['reused_quality_upper']),number(v['m3_quality']['reused_max_request_f1_loss']),v['m3_quality']['reused_critical_regressions']) for w,v in r['workloads'].items()]),
        '数字/空间/运动/安全检查使用原有词法代理，不能替代人工语义认证；每个实际复用请求的原问题、GT、两个自然答案及逐项差异在 `*_m3_natural_quality_pairs.json`。同样输入下前三个增量步及 Prefill 末 Token 的未强制选择 logits 误差保存在 `same_input_controls.jsonl`，包括精确/近似复用、真实来源与实际 Forward 步数；强制答案绝不参与质量评分。',
        table(['完整系统相对 Native','指标','Native','M123','绝对下降','相对下降'],[(w,k,number(x['native']),number(x['candidate']),number(x['loss']),number(x['relative_loss'])) for w,v in r['workloads'].items() for k,x in v['system_quality']['metrics'].items()]),
        table(['工作负载/场景','mAP loss','NDS loss','Lane DET_l loss','Token F1 loss'],[(w+'/'+scene,*[number(x['metrics'][k]['loss']) for k in ('mAP','NDS','DET_l','Token_F1')]) for w,v in r['workloads'].items() for scene,x in v['system_quality']['per_scene'].items()]),
        'M3 内部 .02 和完整系统每项 .10 预算分开判断。Detection/Map 质量来自真实预测与官方 GT 评价；零复用与有复用的感知逐帧摘要完全一致。完整系统与 Native 的差异包含固定 M1/M2，不能把它们全部归给 KV。全部数字仅对声明的小子集成立。',
        table(['自然生成（单轮观测）','请求 mean/P50/P95 ms/帧','M3 Wall mean/P50/P95 ms/帧','实际生成 Tokens','实际 Prefill Tokens'],[(w+'/'+kind,dist(s['request_ms']),dist(s['m3_ms']),s['generated_tokens'],s['counts']['effective_prefill_tokens']) for w,v in r['workloads'].items() for kind in ('native','reuse') for s in [v['summaries']['free_'+kind]]]),
        '自然生成表仅为一次未截短的真实服务观测；输出长度可能不同，不替代三轮固定 Token 计算收益验收。带输入采集的 free_zero 时延不作为性能对照。',
        f'人工式图像核查结果：`{r.get("manual_semantic_review")}`。自动关键词代理零错误不等于语义通过；出现新增双黄线/禁超车/车道数断言，本阶段冻结 Map Approx 候选按语义风险拒绝。原 F1 和自动代理分数完整保留，后续限制性策略在新场景验证，未利用这些已观察场景重调阈值。' if r.get('manual_semantic_review') else '没有新增人工式语义失败记录。',
        '## 相同 Token 的 Prefill、完整 M3 和 E2E',
        '三个必要条件为 Native 完整计算、相同 M1/M2 下正式 Planner/Cache/Executor 固定复用长度零、有真实 WSDR-KV 的完整管线。零对照仅为 Benchmark 的决策覆盖，不是第二个 Runtime。固定序列来自每个原问题的 Native 自然输出，所有条件仍执行真实 HF Forward、Attention、Position 和 KV 更新，逐请求验证相同输出 ID、增量输入历史与实际 Forward 次数。选择开销计入请求，未扣除估算 Token 时间。另有相同视觉、原始 IDs、RNG 和初始真实 KV 存储的三条件隔离回放；为逐请求探针拆开批次时，保持原三问题真实批次测得的调度开销，并校验获准长度与原批次相同，不将拆分探针时延混入主表。',
        '正式自然生成仍使用原 max_new_tokens=320、temperature=.1、top_p=.75。固定 Token 只在 Harness 中应用统一选择处理器并禁用提前 EOS，正常答案和 M3 generation binding 保持。自由生成 zero 条件包含输入采集，只用于质量/诊断，不能与未采集条件作时延结论。所有 headline 时延来自无 trace/无采集的三轮固定 Token 完整请求，轮次顺序 Native/Zero/Reuse、Reuse/Zero/Native、Zero/Native/Reuse。',
        table(['负载/条件（三轮）','Prefill CUDA mean/P50/P95 ms/帧','完整 M3 Wall mean/P50/P95 ms/帧','模型 Wall mean/P50/P95','请求 Wall mean/P50/P95','实际生成 Tokens','实际 Decode Forward'],[(w+'/'+kind,dist(v['summaries']['fixed_'+kind]['prefill_ms']),dist(v['summaries']['fixed_'+kind]['m3_ms']),dist(v['summaries']['fixed_'+kind]['model_ms']),dist(v['summaries']['fixed_'+kind]['request_ms']),v['summaries']['fixed_'+kind]['generated_tokens'],v['summaries']['fixed_'+kind]['actual_decode_forwards']) for w,v in r['workloads'].items() for kind in ('native','zero','reuse')]),
        table(['负载/对照','阶段','加速比','配对节省 mean ms','场景 CI95 ms','稳定净收益'],[(w+'/'+comparison,metric,number(x['speedup']),number(x['saved_ms']['mean']),f'[{x["scene_bootstrap"]["lower"]:.3f}, {x["scene_bootstrap"]["upper"]:.3f}]',x['stable_net_speedup']) for w,v in r['workloads'].items() for comparison,c in v['fixed_comparisons'].items() for metric,x in c.items()]),
        'Bootstrap 固定 seed=20261027、10,000 次，先对场景内全部帧和三轮重复的配对节省求均值，再重采样场景。独立场景只有单问题 6 个、多问题 4 个，不能把重复帧当独立场景。稳定判断要求场景区间下界>0、三轮平均与每场景平均节省均为正；否则只报告观察值与不确定性。',
        table(['负载/对照/阶段','第0/1/2轮加速比','场景平均节省 ms'],[(w+'/'+comparison+'/'+metric,' / '.join(f'{x["speedup"]:.4f}' for x in v['rounds']),str({s:round(a,3) for s,a in v['scene_bootstrap']['scene_means'].items()})) for w,a in r['workloads'].items() for comparison,c in a['fixed_comparisons'].items() for metric,v in c.items()]),
        table(['负载/条件','视觉 CUDA mean/P50/P95 ms','Decode CUDA mean/P50/P95 ms','吞吐帧/s','峰值 allocated/reserved GiB','固定选择 CUDA mean ms'],[(w+'/'+kind,dist(s['visual_cuda_ms']),dist(s['decode_ms']),number(s['throughput_fps']),f'{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}',number(s['selector_overhead_ms']['mean'])) for w,v in r['workloads'].items() for kind in ('native','zero','reuse') for s in [v['summaries']['fixed_'+kind]]]),
        table(['相对零 KV 的收益归因','请求观测稳定','请求节省 ms','模型节省 ms','模型之外节省 ms','其中预处理节省 ms','M3/模型 CI 均为正','E2E 计算净收益已证明'],[(w,a['request_wall_observed_stable'],number(a['request_saved_ms']),number(a['model_saved_ms']),number(a['non_model_saved_ms']),number(a['preprocessing_saved_ms']),a['m3_compute_ci_positive'] and a['model_compute_ci_positive'],a['e2e_compute_gain_proven']) for w,a in r['attribution'].items()]),
        '各阶段 CUDA span 包含 host 提交间隙，并且存在嵌套，不能直接相加重建请求 Wall。请求均值改善若主要来自预处理等模型之外时间，而 M3/模型区间跨零，不能把整段请求改善归功于 KV。完整保留所有正负样本，不扣除估算时间，也不把局部 Prefill 的显著收益替代整个模型的净收益。',
        '## 唯一正式配置、回归与复现',
        f'本阶段实验策略与开始 main 的文件逐字节一致，内部 budget=.02；没有新增放行规则、删除 critical_regressions 或修改成本区间。最终正式策略如经收紧，以 qualification/deployment_decision.json 为准。M2 保持 AR2/.7721027880907059、ordinary_budget=3、max_reuse_frames=8。完整系统预算通过：`{r["system_quality_pass"]}`；M3 自然质量通过：`{r["m3_quality_pass"]}`。相对零 KV 的跨场景稳定 E2E 判断：`{r["cross_scene_e2e_gain_proven"]}`。当前仍只有 `M1→M2→Detection/Map→M1 Commit→M3 ReusePlanner→PrefillExecutor→PrefixKVCache→Decode` 正式路线。',
    ]
    test=root/('tests_gpu_delivery.log' if (root/'tests_gpu_delivery.log').exists() else 'tests_gpu_final.log')
    matches=re.findall(r'\d+ passed[^\n]*',test.read_text());out.append(f'GPU 回归：`{matches[-1] if matches else "见原始日志"}`，日志 `{test.name}`。所有命令使用 OmniDrive、RTX3090 和 `taskset -c 0-3,6-31`，未停止其他任务。历史输出不覆盖。复现入口见 REPRODUCE.md，完整 argv 与退出状态在 commands.json/control_commands.json。')
    files=[('worldstream/m3/scheduler.py','修改','统一证据谓词与离线首次拒绝解释'),('worldstream/m3/reuse_planner.py','修改','删除无资格近似候选的 Block 归约'),('worldstream/m3/prefix_matcher.py','修改','保留精确/合法前缀，按需统计视觉 Block'),('worldstream/m3/task_conditioner.py','修改','保守收紧整体场景与驾驶动作分类'),('tools/worldstream_m123/benchmark/worker.py','修改','显式零复用控制与逐问题 RNG 采集'),('tools/worldstream_m123/benchmark/observe.py','修改','保存实际每个 Prefill 请求开始时 RNG'),('tools/worldstream_m123/benchmark/decision_replay.py','修改','缓存实际版本与诊断 Block/候选解释'),('tools/worldstream_m123/benchmark/m3_recovery.py','新增','仅 Harness 固定零长度决策'),('tools/worldstream_m123/m3_recovery_experiment.py','新增','冻结对照调度'),('tools/worldstream_m123/m3_recovery_control.py','新增','原历史输入、真实同初始状态、logits 和归约控制'),('tools/worldstream_m123/m3_recovery_analysis.py','新增','质量、计数与场景级配对统计'),('tools/worldstream_m123/m3_recovery_report.py','新增','原始数据生成本报告与图'),('tools/worldstream_m123/m3_recovery_reproduce.py','新增','在新目录复现冻结的完整协议'),('tools/worldstream_m123/m3_recovery_qualification.py','新增','只收紧失败规则并在新场景验证'),('tools/worldstream_m123/m3_recovery_final_report.py','新增','区分失败候选与最终部署结果')]
    out.extend(['### 插桩变更记录',table(['文件','类型','用途','行数'],[(p,k,d,len(Path(p).read_text().splitlines())) for p,k,d in files]),
        '正式运行不挂诊断回放或额外标签执行器。新增采集、强制选择与零长度控制均由 Benchmark flags 显式启用；可按上述工具文件单独移除，不影响正式单一路径。',
        '单问题流仍可能天然没有获准长前缀；本轮没有为非零 Token 修改问题或安全边界。多问题场景证明适用任务仍可发生真实跨帧复用；是否构成稳定 M3/E2E 净收益，以配对表和场景区间为准，不能用命中数代替速度结论。'])
    (root/'experiment_report.md').write_text('\n\n'.join(out)+'\n');figures(root,r)


def figures(root,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for axis,(workload,v) in zip(axes,r['workloads'].items()):
        comparison=v['fixed_comparisons']['reuse_vs_zero'];keys=('prefill_ms','m3_ms','model_ms','request_ms','non_model_ms')
        y=[comparison[k]['scene_bootstrap']['mean_saved_ms'] for k in keys]
        low=[a-comparison[k]['scene_bootstrap']['lower'] for a,k in zip(y,keys)];high=[comparison[k]['scene_bootstrap']['upper']-a for a,k in zip(y,keys)]
        axis.errorbar(range(5),y,yerr=[low,high],fmt='o',capsize=4);axis.axhline(0,color='black',lw=.7)
        axis.set_xticks(range(5));axis.set_xticklabels(['Prefill','M3','Model','E2E','Outside model'],rotation=15);axis.set(title=workload,ylabel='Scene-paired saving (ms), 95% bootstrap');axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'m3_fixed_compute.png',dpi=180);fig.savefig(root/'m3_fixed_compute.pdf');plt.close(fig)
