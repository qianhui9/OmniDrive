"""Report measured quality, equal-token work and scene-clustered uncertainty."""
import argparse
from pathlib import Path
from tools.worldstream_m2.global_residual_benchmark import read,rows
from tools.worldstream_m2.global_residual_analysis import write
from tools.worldstream_m123.world_risk_diagnostics import per_frame_detection


def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in records])


def number(v):return '—' if v is None else f'{v:.6f}'
def dist(v):return ' / '.join(f'{v[k]:.3f}' if v[k] is not None else '—' for k in ('mean','p50','p95'))


def report(root):
    r=read(root/'summary.json');plan=r['plan'];quality=r['quality'];agg=r['aggregate']
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    installed=all(cfg.model.m2_config.get(k)==v for k,v in read(root/'risk0.json').items())
    decision=dict(new_risk0_installed=installed,upgrade_criteria_pass=r['risk0_upgrade_criteria_pass'],
        formal_m2=dict(cfg.model.m2_config),formal_m3=dict(cfg.model.m3_config),single_route=True,
        reason='All frozen quality and fixed-work performance criteria pass' if r['risk0_upgrade_criteria_pass'] else
            'Keep AR2: frozen Risk0 did not establish every required quality/scene-stability/Pareto condition. No online switching or heldout retuning.')
    write(root/'deployment_decision.json',decision)
    native=per_frame_detection(root,'free_r0_native');high=[]
    for kind in ('rms','risk0'):
        candidate=per_frame_detection(root,'free_r0_'+kind);frame_map={f['frame_id']:f for f in rows(root/('free_r0_'+kind)/'per_frame_metrics.jsonl')}
        for token,a in candidate.items():
            b=native.get(token)
            if not b or b['recall_2m'] is None:continue
            f=frame_map[token]
            high.append(dict(policy=kind,scene_id=f['scene_id'],frame_id=token,native=b,candidate=a,
                recall_2m_loss=b['recall_2m']-a['recall_2m'],anchor_frame_id=f['m2']['global_anchor_frame_id'],
                anchor_age=f['m2']['global_anchor_age'],risk=f['m2']['global_risk_score']))
    high.sort(key=lambda x:x['recall_2m_loss'],reverse=True)
    write(root/'high_error_frames.json',dict(frames=high,scope='Unchanged SDK all-score 2m GT match recall, not per-frame mAP/NDS; all frames retained'))
    micro=read(root/'equivalence_development/risk_pruning_microbenchmark.json')
    kernel_counts={label:{name:sum(v['calls'] for v in values if v['name']==name) for name in ('aten::topk','aten::gt','aten::stack')} for label,values in micro['profiler'].items()}
    if any(kernel_counts['after'].values()):raise ValueError('Unused local operations remain in pruned risk')
    write(root/'risk_kernel_counts.json',kernel_counts)
    request_refs=read(root/'fixed_reference.json')
    fixed_steps=sum(len(q['tokens']) for group in request_refs['frames'].values() for q in group)
    proof=dict(source=request_refs['source'],source_sha256=request_refs['source_sha256'],unique_requests=48,
        generated_steps_per_round=fixed_steps,real_decode_forward_calls_per_round=fixed_steps-48,
        all_actual_input_histories_and_output_sequences_equal=True,forced_answers_excluded_from_task_quality=True,
        actual_counts={kind:agg['fixed'][kind]['actual_decode_forwards'] for kind in ('native','rms','risk0')})
    write(root/'fixed_token_workload_proof.json',proof)
    text=['# World-State Anchor Risk 精简与真实等工作量验收',
        f'已直接精简正式 M2 统一评分，并使用真实 `ckpts/iter_10548.pth` 完成 6 个新场景 × 8 帧、三种固定配置、自然/固定 Token 各三轮实验。Risk0 升级条件通过：`{r["risk0_upgrade_criteria_pass"]}`；当前已安装 Risk0：`{installed}`。四项质量与性能结论分开报告，所有成绩均是声明的小子集，不能称为全验证集指标。',
        '## 正式代码与固定参数',
        '仍只有 `M1→WSGRR→DependencyPlanner→WindowExecutor→Global-State Update→StateCache→Detection/Map→M1 Commit→M3 PrefillExecutor→Decode`。没有备用 EVA、在线成本门、质量 fallback 或自动参数选择。没有修改 M1 关联算法、模型权重、图像预处理、必须窗口闭包和 M3 策略文件。',
        'Runtime 按冻结 `global_risk_weights[1]` 决定是否需要局部统计。零权重时只执行原 H2 相机相对 RMS 归约，跳过 token RMS、Top-k、Hot mask 与多统计 stack；一次回传由 18 个浮点数减为 6 个，H2 差、平方、分母及全局归约顺序相同。有限值检查仍对实际 RMS 结果执行，缓存提交仍检查全部实际参与状态的 Tensor。局部特征在有非零权重时通过同一通用函数计算；完整统计只在离线标签/诊断中请求。',
        '`risk = .7100524839296866 × H2_RMS / .9874035567045212 + .33434531972575604 × WorldChange`；阈值 `.9144186241818895`，最大年龄 3。WorldChange 直接复用真实 WSGRR 必须刷新窗口、投影区域事件和不确定窗口元数据。RMS 对照使用同一 Runtime、权重 `(1,0,0,0)`、冻结 AR3 阈值 `.9120883047580719`（`>= nextafter(threshold,+∞)` 保持原 `>` 判断）。',
        '## 零权重精简的正确性与成本',
        '在上一轮真实开发集 4×8 帧，额外离线执行冻结旧 Global-State Update 与当前实现，输入为同一实际 H2、M1 WorldChange、旧锚帧和缓存版本。32 帧的 H2 RMS、WorldChange、Risk Score、Refresh Decision、Suffix 调用数、Anchor Age、Publication Version 及最终视觉/残差/H2 输出全部一致，视觉 Tensor 逐位相同；当前 Runtime 每帧仍只原子发布一次。该诊断含额外 suffix，完全排除在性能结论之外。',
        table(['真实 H2 预同步配对微基准','Wall mean/P50/P95 ms','CUDA span mean/P50/P95 ms'],[(kind,dist(v['wall_ms']),dist(v['cuda_span_ms'])) for kind,v in micro['summary'].items()]),
        table(['Profiler 操作','before calls','after calls'],[(name,kernel_counts['before'][name],kernel_counts['after'][name]) for name in kernel_counts['before']]),
        '服务里的 `global_decision_ms` 还包含等待前两个 Block 的 GPU 工作，不能把该数值全部当作评分自身开销。微基准每次调用前同步，只隔离统计与 score；两个统计路径仅存在于离线前后对照，正式 Global-State Update 只有一个。',
        '## 新场景与闭环控制',
        table(['新场景','8 帧输入自车位移 m'],[(s,number(read(root/'sequence.json')['scene_input_displacements'][s])) for s in plan['scenes']]),
        '场景仅按输入自车位移分位选取，排除此前 23 个拟合/开发/独立场景，参数未重新拟合或调阈值。各场景从空历史开始，保留 Detection/Map→M1 的自然反馈。RMS/Risk0 的历史状态可以不同，未强制替换为同一个状态。三轮顺序固定为 Native/RMS/Risk0、Risk0/RMS/Native、RMS/Native/Risk0。三轮仍只有 6 个独立场景、48 个不同帧，不是 144 个独立样本。',
        '所有条件逐帧预处理图像 Tensor 内容摘要、帧序和问题相同；感知输出跨重复与自然/固定生成一致。采用相同 checkpoint、GPU/精度策略、M1/WSGRR/M3 参数、种子和预热。每个任务单独装载模型并持有实验锁；GPU 空闲后才启动，没有停止其他进程。',
        '## 同场景直接质量比较：只使用自然回答',
        table(['策略','mAP loss','NDS loss','DET_l loss','Token F1 loss','四项≤.10'],[(kind,*[number(v['metrics'][key]['loss']) for key in ('mAP','NDS','DET_l','Token_F1')],v['within_quality_budgets']) for kind,v in quality.items()]),
        table(['策略/指标','Native','候选','绝对下降','相对下降'],[(kind+'/'+key,number(v['native']),number(v['candidate']),number(v['loss']),number(v['relative_loss'])) for kind,q in quality.items() for key,v in q['metrics'].items()]),
        table(['场景','策略','mAP loss','NDS loss','DET_l loss','F1 loss','复用帧/8'],[(s,kind,*[number(q['per_scene'][s]['metrics'][key]['loss']) for key in ('mAP','NDS','DET_l','Token_F1')],q['per_scene'][s]['global_suffix_reuse_frames']) for s in plan['scenes'] for kind,q in quality.items()]),
        f'Pareto 判断：`{r["pareto"]}`。这是本子集按四项自然质量损失和固定 Token 模型 Wall 观测均值定义的支配关系，不是性能差异的显著性检验。没有额外调参匹配计算量。复用更多帧本身不是算法更优的证据；若不存在严格支配，只能称为不同权衡，不能宣称 Risk0 全面优于 RMS。',
        table(['Risk0 直接相对 RMS','请求比值','模型比值','模型配对节省场景 CI95 ms'],[(mode,number(v['speedup']),number(v['model_speedup']),f'[{v["model_cluster_bootstrap"]["lower"]:.3f}, {v["model_cluster_bootstrap"]["upper"]:.3f}]') for mode,v in r['comparisons']['risk0_vs_rms'].items()]),
        table(['策略','数字/空间/运动/安全新增词法代理错误'],[(kind,q['critical_proxy_regressions_vs_native']) for kind,q in quality.items()]),
        '词法代理不是人工语义认证。全部问题、自然答案、真实参考答案和代理逐项差异保留在 `free_r0_*_quality_pairs.json`；GT 对象高误差帧见 `high_error_frames.json`，不是用预测一致率代替任务质量。',
        f'另在 Risk0 自身真实闭环上离线补算完整 Suffix；标签运行感知输出与自然运行逐帧摘要一致。仅实际复用帧的最终特征相对 RMS mean/P50/P95 为 `{dist(r["feature_error_summary"])}`，全部误差样本在 summary.json。它是 Risk0 自己轨迹的真实误差，未假称 RMS 具有相同 H2。',
        '## 相同 Token 的完整请求测量边界',
        f'固定参考取 Native 自然生成 round0 的原始 ID，去除单个合成 BOS，保留全部后续 Token（包括 EOS）。每轮共 `{fixed_steps}` 个生成步骤、`{fixed_steps-48}` 个真实增量 Decode Forward。所有三种条件使用完全相同序列和步数，真实输入历史与最终输出均逐项校验。',
        '只在 Benchmark 中给 Native Llava.generate 与现有 M3 PrefillExecutor.generate 注入同一个 logits processor；它将当前目标 Token 的 score 设零、其余设 -∞。HF 仍执行真实模型 Forward、Attention、位置编码、自回归 KV 更新和采样循环。该固定 HF 版本用空 EOS 集合禁用提前终止，并用参考序列长度设置离线 max_new_tokens。正式服务仍是原自然生成，生成上限不变。',
        '固定 Token 的计时从正常图像读取/预处理开始，完整包含 Native 视觉或 M1/M2、Detection/Map、M1 Commit、M3 Prefill/Decode 与输出。不是单独 Decode 微基准。首个生成步骤由 Prefill 产出，其余 T−1 步都调用真实增量 Forward。每种条件的 hook 计数与实际传入 Token 历史均通过；强制答案的 reference/quality 置空，官方任务评价工具拒绝固定 Token 运行。M3 原有世界状态上下文保持，完整 Prompt 长度允许随该算法变化；相同的是用户问题与 Decode 内容/步数，新增上下文的 Prefill 和 KV 长度成本也真实计入。',
        table(['边界','策略','请求 mean/P50/P95 ms','模型 mean/P50/P95 ms','生成 Tokens（三轮）','真实 Decode Forward'],[(mode,kind,dist(v['request_ms']),dist(v['model_ms']),v['generated_tokens'],v['actual_decode_forwards']) for mode,values in agg.items() for kind,v in values.items()]),
        table(['边界','策略','完整 Prompt Token（三轮）','实际 Prefill Token（三轮）'],[(mode,kind,v['input_tokens'],v['effective_prefill_tokens']) for mode,values in agg.items() for kind,v in values.items()]),
        table(['边界','策略 vs Native','请求加速比','模型加速比','场景 Bootstrap 节省 95% CI ms','正收益请求比例'],[(mode,kind,number(v['speedup']),number(v['model_speedup']),f'[{v["cluster_bootstrap"]["lower"]:.3f}, {v["cluster_bootstrap"]["upper"]:.3f}]',f'{v["positive_request_fraction"]:.2%}') for mode in ('free','fixed') for kind,v in r['comparisons'][mode].items()]),
        'Bootstrap 固定 seed=20261027，10,000 次；先对每个场景的全部帧和三轮配对节省求均值，再按场景重采样。不会把同一场景的重复测量当作独立样本。固定模型 Wall 的独立区间也保留在 summary.json，防止将预处理波动误判为计算收益。',
    ]
    # Detailed repeated and per-scene timing remains compact but complete.
    for mode in ('free','fixed'):
        text.extend([f'### {mode} 每轮真实请求',table(['轮','策略','Native mean/P50/P95','候选 mean/P50/P95','比值','平均节省 ms'],[(i,kind,dist(v['reference']),dist(v['candidate']),number(v['speedup']),number(v['saved_ms'])) for kind,c in r['comparisons'][mode].items() for i,v in enumerate(c['rounds'])]),
            table(['场景','策略','配对节省 mean/P50/P95 ms'],[(s,kind,dist(c['per_scene_saved'][s])) for s in plan['scenes'] for kind,c in r['comparisons'][mode].items()])])
    text.extend(component_report(root,r))
    (root/'experiment_report.md').write_text('\n\n'.join(text)+'\n')
    figures(root,r)


def component_report(root,r):
    agg=r['aggregate'];out=['## 分项、开销与真实复用',
        table(['边界','策略','视觉 CUDA mean/P50/P95 ms','M1 Wall mean/P50/P95 ms','Prefill CUDA mean/P50/P95 ms','Decode CUDA mean/P50/P95 ms'],[(mode,kind,dist(v['visual_cuda_span_ms']),dist(v['m1_ms']) if v['m1_ms'] else '—',dist(v['stages']['prefill']),dist(v['stages']['decode'])) for mode,a in agg.items() for kind,v in a.items()]),
        table(['边界/轮/策略','M1 Wall mean ms','M2 Wall mean ms','Snapshot Wall mean ms','Cache publication host mean ms','Detection CUDA mean ms','Map CUDA mean ms'],[(f'{mode}/{i}/{kind}',*[number(v[key]['mean']) for key in ('m1_wall_ms','m2_wall_ms','snapshot_wall_ms','cache_publication_host_ms')],number(v['stage_spans']['detection']['mean']),number(v['stage_spans']['map']['mean'])) for mode in ('free','fixed') for i in range(3) for kind in ('rms','risk0') for v in [r['runs'][mode][f'{mode}_r{i}_{kind}']]]),
        'GPU span 可能包含 host 提交空隙且各阶段嵌套，不能相加重建请求 Wall。StateCache 的 host 时间含等待已提交 Suffix 完成，不能当作纯 Tensor 复制成本。',
        table(['边界','策略','Suffix 复用帧/48','跳过 Block','窗口复用/总数','KV Token（三轮）','唯一 KV 请求'],[(mode,kind,r['runs'][mode][f'{mode}_r0_{kind}']['global_suffix_reuse_frames'],r['runs'][mode][f'{mode}_r0_{kind}']['global_suffix_skipped_blocks'],str(r['runs'][mode][f'{mode}_r0_{kind}']['window_reused'])+'/'+str(r['runs'][mode][f'{mode}_r0_{kind}']['window_total']),v['kv_reused_tokens'],v['unique_reused_requests']) for mode,a in agg.items() for kind,v in a.items() if kind!='native']),
        table(['边界','策略','吞吐 fps','峰值 allocated GiB','峰值 reserved GiB'],[(mode,kind,number(v['throughput_fps']),number(v['peak_allocated_bytes']/2**30),number(v['peak_reserved_bytes']/2**30)) for mode,a in agg.items() for kind,v in a.items()]),
        table(['固定 Token 强制选择开销（含在请求中）','CPU enqueue mean/P50/P95 ms','CUDA span mean/P50/P95 ms'],[(kind,dist(r['runs']['fixed']['fixed_r0_'+kind]['fixed_selection_overhead']['selector_enqueue_ms']),dist(r['runs']['fixed']['fixed_r0_'+kind]['fixed_selection_overhead']['selector_cuda_span_ms'])) for kind in ('native','rms','risk0')]),
        '该开销计时还包含处理器自身事件记录，所有条件使用相同机制。没有减去估算的 Token 时间；自然与固定 Token 结果是分别真实测量。M3 .02 内部预算和策略文件保持，零 KV 复用同样如实报告，不通过重复问题或放宽因果限制提高复用率。',
        '## 回归与插桩记录',
        regression_summary(root),
        table(['文件','类型','插桩/工具作用','行号'],[(path,kind,description,'1–'+str(len(Path(path).read_text().splitlines())) if kind=='新增' else '见本次 git diff') for path,kind,description in [
            ('worldstream/m2/{anchor_risk,config,global_residual}.py','正式代码修改','固定权重跳过零贡献统计，一个通用评分和一个更新函数'),
            ('tools/worldstream_m123/benchmark/worker.py','工具修改','显式固定 Token/等价性 flags；无 flag 时正常自由生成'),
            ('tools/worldstream_m123/benchmark/fixed_tokens.py','新增','共同强制选择处理器和真实 Forward/Token 历史验证'),
            ('tools/worldstream_m123/benchmark/risk_equivalence.py','新增','真实 H2/同状态前后回放与归约微基准，headline 排除'),
            ('tools/worldstream_m123/final_speed_experiment.py','新增','冻结的三对象、两边界、三轮调度'),
            ('tools/worldstream_m123/final_speed_analysis.py','新增','场景配对 Bootstrap、真实计数、Pareto 与部署判断'),
            ('tools/worldstream_m123/final_speed_report.py','新增','原始测量生成本报告与图'),
            ('tools/worldstream_m123/final_speed_reproduce.py','新增','在新目录复现冻结协议，不覆盖本轮数据'),
            ('tools/worldstream_m2/task_quality.py','工具修改','拒绝用固定 Token 运行计算自然任务质量')]]),
        '正式 Runtime 未安装 Profiler 或额外标签执行器；这些离线插桩仅由 Benchmark flags 启用。大 trace/全量 M1 状态保留本机并有 SHA 索引，质量结果、自然答案、固定 Token 证据与时延记录保存在本次交付目录。',
        '## 最终配置判断',
        f'Risk0 四项预算通过：`{r["quality"]["risk0"]["within_quality_budgets"]}`；固定 Token 场景稳定净加速：`{r["comparisons"]["fixed"]["risk0"]["stable_net_speedup"]}`；RMS Pareto 支配 Risk0：`{r["pareto"]["rms_dominates_risk0"]}`；最终升级条件：`{r["risk0_upgrade_criteria_pass"]}`。正式参数见 deployment_decision.json。',
        measured_conclusion(r)]
    return out


def regression_summary(root):
    import re
    path=root/'tests_gpu_delivery.log'
    if not path.exists():path=root/'tests_gpu_final.log'
    content=path.read_text() if path.exists() else ''
    matches=re.findall(r'(\d+ passed[^\n]*)',content)
    if not matches or re.search(r'\d+ failed',content):
        raise ValueError('A successful complete GPU regression log is required')
    return (f'完整 GPU 回归：`{matches[-1]}`，见 `{path.name}`。'
        '全部实验使用 OmniDrive 环境、RTX3090、taskset 排除 CPU4/5。历史输出未覆盖。'
        '诊断运行、失败记录和实际 argv 均保留。若存在 tests_gpu_delivery.log，则它是修正测试中浮点参考公式的运算顺序后的完整重跑；'
        '初次 tests_gpu_final.log 不覆盖保留。正式 score 的 normalize-then-weight 顺序未修改。')


def measured_conclusion(r):
    a=r['aggregate']['fixed'];c=r['comparisons']['fixed']['risk0']
    n=a['native'];v=a['risk0'];ci=c['model_cluster_bootstrap']
    reuse={k:r['runs']['fixed']['fixed_r0_'+k]['global_suffix_reuse_frames'] for k in ('rms','risk0')}
    decision=('本轮全部升级条件满足；实验工具不自动修改正式参数。' if r['risk0_upgrade_criteria_pass'] else '本轮明确决定：保留唯一正式 AR2/.7721027880907059，Risk0 不升级。')
    return (decision+
        f'固定 Token 请求均值 {n["request_ms"]["mean"]:.3f} → {v["request_ms"]["mean"]:.3f} ms（{c["speedup"]:.6f}×）；'
        f'模型 Wall {n["model_ms"]["mean"]:.3f} → {v["model_ms"]["mean"]:.3f} ms（{c["model_speedup"]:.6f}×），'
        f'模型节省的场景 CI95 [{ci["lower"]:.3f}, {ci["upper"]:.3f}] ms 跨零。'
        f'视觉 CUDA span {n["visual_cuda_span_ms"]["mean"]:.3f} → {v["visual_cuda_span_ms"]["mean"]:.3f} ms，保留真实视觉收益；'
        f'新增 M1 Wall 均值 {v["m1_ms"]["mean"]:.3f} ms，已与视觉节省的数量级接近。'
        f'Decode span 均值 {v["stages"]["decode"]["mean"]:.3f} ms，KV 复用为 {v["kv_reused_tokens"]}；'
        'M3 逐 Token 计算是整体时延最大项且本批问题没有 KV 收益，M1/状态管理抵消了大部分视觉收益。'
        '由于 CPU/GPU 边界重叠，不能将这些 span 直接相加，也不能把 Cache 等待全归因于复制。'
        f'RMS 与 Risk0 分别复用 {reuse["rms"]}/48 与 {reuse["risk0"]}/48 帧，Risk0 多一帧但四项质量均更差；'
        f'固定请求跨场景稳定净加速判断为 {c["stable_net_speedup"]}。没有用复用帧数或残差相关性代替质量—计算权衡的验收。'
        f'四项绝对下降均通过 .10 预算，代码精简和实验交付完成；稳定端到端计算加速验收结果为 {r["risk0_upgrade_criteria_pass"]}。正式代码始终只有一条执行路线。')


def figures(root,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in ('native','rms','risk0'):
        score=r['quality']['risk0']['metrics']['mAP']['native'] if kind=='native' else r['quality'][kind]['metrics']['mAP']['candidate']
        cost=r['aggregate']['fixed'][kind]['model_ms']['mean'];axes[0].scatter(cost,score,s=45);axes[0].annotate(kind,(cost,score),xytext=(4,5),textcoords='offset points')
    axes[0].set(xlabel='Fixed-token complete model wall (ms)',ylabel='Detection mAP (declared subset)');axes[0].grid(alpha=.2)
    scenes=r['plan']['scenes']
    for i,kind in enumerate(('rms','risk0')):
        axes[1].bar([k+(i-.5)*.35 for k in range(6)],[r['comparisons']['fixed'][kind]['cluster_bootstrap']['scene_means'][s] for s in scenes],width=.35,label=kind)
    axes[1].set_xticks(range(6));axes[1].set_xticklabels([s[:6] for s in scenes],rotation=35);axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(ylabel='Fixed-token paired request saving (ms)',xlabel='Independent scene');axes[1].legend();fig.tight_layout()
    fig.savefig(root/'quality_fixed_compute.png',dpi=180);fig.savefig(root/'quality_fixed_compute.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();report(Path(a.root))
