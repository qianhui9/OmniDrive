"""Final deployment and measured claims, including the failed frozen candidate."""
import argparse
import hashlib
from pathlib import Path
import re
from tools.worldstream_m2.global_residual_benchmark import read
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.final_speed_report import table,number,dist


def report(root):
    old=read(root/'summary.json');q=read(root/'qualification/summary.json');decision=read(root/'qualification/deployment_decision.json')
    if not q['installed']:raise ValueError('Qualified policy has not been installed')
    canonical=Path('outputs/final_route/m3_reuse_policy.json')
    if canonical.read_bytes()!=(root/'qualification/frozen_m3_policy.json').read_bytes():raise ValueError('Canonical policy differs from verified policy')
    micro=read(root/'prefix_pruning_microbenchmark.json');c=q['comparisons']['reuse_vs_zero'];cn=q['comparisons']['reuse_vs_native'];counts=q['summaries']['free_reuse']['counts']
    compute_proven=(c['request_ms']['stable_net_speedup'] and c['m3_ms']['scene_bootstrap']['lower']>0 and c['model_ms']['scene_bootstrap']['lower']>0)
    test=root/'tests_gpu_qualified.log'
    if not test.exists():test=root/'tests_gpu_delivery.log'
    match=re.findall(r'\d+ passed[^\n]*',test.read_text())
    if not match or re.search(r'\d+ failed',test.read_text()):raise ValueError('Full regression is not passing')
    summary=dict(status='COMPLETED',final_policy_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
        historical_prechange_current_pipeline_reused_tokens=old['historical_current_pipeline']['counts']['skipped_tokens'],
        historical_exact_input_current_code=old['historical_exact_inputs'],original48_first_rejections=old['rejection']['original48']['first_candidate_reasons'],
        frozen_original_multi_counts=old['workloads']['multi']['summaries']['free_reuse']['counts'],
        original_policy_manual_semantic_failure=read(root/'semantic_failure_review.json'),
        final_policy_counts=counts,final_natural_quality=q['m3_quality'],final_system_quality=q['system_quality'],
        final_fixed_comparisons=q['comparisons'],final_e2e_compute_gain_proven=compute_proven,
        single_question_counts=old['workloads']['single']['summaries']['free_reuse']['counts'],
        pruning_microbenchmark=micro['summary'],sampled_kernel_counts=micro['kernel_counts'],
        full_regression=match[-1],single_formal_route=True,no_new_rules_admitted=True)
    write(root/'final_summary.json',summary)
    out=['# M3 WSDR-KV：最终代码、质量与真实计算验收',
        '**完成实现和真实 checkpoint 验证。旧 M3 的 KV 执行能力没有消失，最新单问题零复用主要来自任务分布和已冻结的质量/成本约束。新增语义失败已保留并处理：最终唯一策略禁用 Map Relations Approx，保留其余原规则及合法 Exact 复用。**',
        '## 历史复现和零复用原因',
        f'任务开始 main 在历史24帧×3问题上精确复现5607 Tokens（8次Exact、7次跨帧Bounded Approx）。保守修正任务分类后，使用原始实际Carriers、IDs、RNG、M1 Snapshot、M2 Receipt隔离回放当前M3，Native原始Token全部复现，复用 `{old["historical_exact_inputs"]["reused_tokens"]}` Tokens，`{old["historical_exact_inputs"]["answers_equal"]}/72` 自然回答与相同视觉的Native一致。少一次163 Token近似与将“道路坡度下如何管理车速”从Map收紧为Full Visual有关，不能称作Executor失效。',
        f'原始48问的具体首次拒绝：`{old["rejection"]["original48"]["first_candidate_reasons"]}`。37次Full Visual Exact先被已有critical=1拒绝，5次Object Exact先被质量上界.025974>.02拒绝，另6次冷启动。42次热缓存都存在35 Token的真实精确前缀，但不是已证明可获净收益的前缀；Object Approx普通跨帧最大163 Token，达不到其成本证据至少558 Token的范围。输入长度、风险和缺失校准的详细数值在原始候选日志中。',
        '## 直接修改的正式代码',
        '只改现有TaskConditioner、Scheduler、ReusePlanner和PrefixMatcher：保守修正驾驶动作/通行空间/整体区域问题；共享证据谓词支持精确首次拒绝解释；同帧或近似证据不合格时不再计算无用的Block RMS。Exact/Legal Prefix、Finite、场景/相机/模型/来源/版本、位置和Mask检查保留。PrefillExecutor及PrefixKVCache没有被旧文件覆盖。',
        table(['真实Prefix统计','Wall mean/P50/P95 ms','CUDA span mean/P50/P95 ms'],[(k,dist(v['wall_ms']),dist(v['cuda_ms'])) for k,v in micro['summary'].items()]),
        f'单次Profiler采样：`{micro["kernel_counts"]}`。零需求归约移除34次sqrt、87次sum、17次max；必要的前缀stack保留。微基准前同步、交替测量6组真实相邻输入，不作为完整E2E时延。相同输入/缓存/任务Profile和固定观测调度开销的旧/新Planner决策一致，实际Forward长度与跳过Token数逐项核对。',
        '## 为什么没有把原策略的F1提高当作质量通过',
        '第一批新多问题流曾复用1606 Tokens（2次同帧Exact、3次跨帧Bounded Approx），71/72答案一致，另一答案F1提高且原自动关键词代理未报警。但六相机核查显示：新增“双黄线、禁止超车、每方向单车道”断言缺少图像和标注支持；零复用回答所述斑马线有图像依据。这是独立的新增语义风险，不能被F1上升抵消。',
        '完整失败证据在 semantic_failure_review.json。原自动分数不修改，仅给已有Map Approx规则追加critical拒绝证据，不调风险/长度/成本阈值，不新增规则。该批失败场景已被观察，不再用于新规则拟合或新的独立通过声明。',
        '## 最终收紧策略：另一批新独立场景',
        f'另4场景×6连续帧×3个真实标注问题，选择后保持原标注顺序。自然答案 `{q["m3_quality"]["answers_equal"]}/72` 与相同M1/M2条件下零历史KV对照一致，平均F1下降 `{q["m3_quality"]["mean_f1_loss"]}`，新增答案差异 `{q["m3_quality"]["natural_answer_differences"]}`。仅此小子集支持相对零KV无新增回答退化，不是全验证集保证。',
        table(['工作负载/策略','实际KV分类','输入Prefill Tokens','有效Prefill Tokens','跳过Tokens','下降比例'],[
            ('此前单问题48问',str(summary['single_question_counts']['groups']),summary['single_question_counts']['input_tokens'],summary['single_question_counts']['effective_prefill_tokens'],summary['single_question_counts']['skipped_tokens'],f'{summary["single_question_counts"]["skipped_fraction"]:.3%}'),
            ('最终新多问题72问',str(counts['groups']),counts['input_tokens'],counts['effective_prefill_tokens'],counts['skipped_tokens'],f'{counts["skipped_fraction"]:.3%}')]),
        '两批多问题场景不同，1606与最终3908等计数不能用于宣称参数或算法更优。跨帧Map Approx因新增语义风险不作为最终获准收益；历史功能复现和最终部署结论分别保留。单问题流不能因多问题Exact命中而被宣称也能加速。',
        table(['最终系统指标','Native','M123','绝对下降','相对下降'],[(k,number(v['native']),number(v['candidate']),number(v['loss']),number(v['relative_loss'])) for k,v in q['system_quality']['metrics'].items()]),
        f'四项绝对下降≤.10：`{q["system_quality"]["within_quality_budgets"]}`。内部M3 budget=.02不变。所有任务质量仅使用正常自由生成及真实GT评价；固定Token答案不计质量。',
        '## 固定相同Token的真实计算性能',
        '同一新Manifest、相同六相机输入、相同问题及相同参考Token内容/数量，三个条件各做三轮交替完整请求。零复用只通过Benchmark将同一个Planner决策长度固定为零，仍使用原Executor/Cache；不建立第二Runtime。每次真实HF Forward/Attention/位置/KV更新均执行，实际增量输入历史和Forward次数校验通过。',
        table(['最终策略对照','Prefill CUDA加速比','完整M3 Wall加速比','完整模型加速比','完整请求加速比'],[(label,*[number(v[k]['speedup']) for k in ('prefill_ms','m3_ms','model_ms','request_ms')]) for label,v in (('相同M1/M2零KV',c),('Native完整计算',cn))]),
        table(['最终策略相对零KV','平均节省 ms/帧','场景CI95 ms'],[(k,number(v['saved_ms']['mean']),f'[{v["scene_bootstrap"]["lower"]:.3f}, {v["scene_bootstrap"]["upper"]:.3f}]') for k,v in c.items()]),
        f'由M3/模型计算共同支持的稳定E2E净收益：`{compute_proven}`。请求节省为 `{c["request_ms"]["saved_ms"]["mean"]:.3f}` ms，其中模型之外为 `{c["non_model_ms"]["saved_ms"]["mean"]:.3f}` ms。若M3或模型区间跨零，不能把整段请求改善归因于KV。原冻结策略曾出现请求改善2.04%但约82/91 ms来自模型之外，本轮没有隐藏该结果或据此宣称计算加速。',
        'Bootstrap先在场景内合并三轮重复后按4个场景重采样10,000次，seed=20261027；重复帧不是新增独立场景。完整每轮、P50/P95、显存、吞吐、自然生成计数、来源、拒绝数值和未强制选择logits误差见两个阶段的详细报告/JSON。',
        '## 唯一正式状态与复现',
        f'正式文件 `outputs/final_route/m3_reuse_policy.json` 的SHA为 `{summary["final_policy_sha256"]}`。只上调已失败Map Approx规则的critical字段并附原始证据，旧Full Visual critical=1与Object Exact .025974拒绝记录保留，其他所有规则和成本参数逐项未改。M2仍是AR2/.7721027880907059；ordinary_budget=3、max_reuse_frames=8及M3内部.02不变。仍只有一条M1+M2+M3正式路线，无自动算法切换。',
        f'最终完整回归：`{summary["full_regression"]}`，见 `{test.name}`。全部真实实验使用OmniDrive与RTX3090，taskset排除CPU4/5；没有停止其他用户任务。此前失败/中间日志和历史输出保留。',
        '复现命令见 REPRODUCE.md；冻结候选详细报告为 experiment_report.md，最终策略详细报告为 qualification/experiment_report.md，最终部署判定为 deployment_decision.json。原始大Tensor/trace在本机按SHA索引，不能把诊断额外Forward计入headline时延。']
    (root/'FINAL_REPORT.md').write_text('\n\n'.join(out)+'\n')
    figures(root,c)
    write(root/'final_status.json',dict(status='COMPLETED',final_policy_installed=True,all_task_budgets_pass=q['system_quality']['within_quality_budgets'],m3_natural_answers_equal=q['m3_quality']['answers_equal']==72,
        e2e_compute_gain_proven=compute_proven,full_regression=summary['full_regression'],single_formal_route=True))


def figures(root,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=('prefill_ms','m3_ms','model_ms','request_ms','non_model_ms')
    means=[comparison[k]['scene_bootstrap']['mean_saved_ms'] for k in keys]
    lower=[m-comparison[k]['scene_bootstrap']['lower'] for m,k in zip(means,keys)]
    upper=[comparison[k]['scene_bootstrap']['upper']-m for m,k in zip(means,keys)]
    fig,axis=plt.subplots(figsize=(8,4))
    axis.bar(range(5),means,color=['#3785b5' if m>=0 else '#c06c42' for m in means],alpha=.8)
    axis.errorbar(range(5),means,yerr=[lower,upper],fmt='none',color='black',capsize=4)
    axis.axhline(0,color='black',lw=.7);axis.set_xticks(range(5))
    axis.set_xticklabels(['Prefill','Complete M3','Model','Request','Outside model'])
    axis.set(ylabel='Scene-paired saving vs zero KV (ms), 95% CI',title='Qualified policy: fresh natural multi-question stream')
    axis.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(root/'qualified_fixed_compute.png',dpi=180);fig.savefig(root/'qualified_fixed_compute.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();report(Path(a.root))
