"""Summarize verified frozen-main results without changing any measured metric."""
import argparse
from pathlib import Path

from .benchmark.common import write,sha256
from .frozen_main_benchmark import ROOT,read


def f(value,digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def table(headers,items):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in items)


def synthesize(root):
    completion=read(root/'completion.json');assert completion['status']=='READY_FOR_GIT_REVIEW'
    assert read(root/'reanalysis_validation.json')['status']=='PASS'
    verdict=read(root/'final_research_verdict.json');workloads=read(root/'workload_results.json')
    m1=read(root/'m1_metrics.json');m2=read(root/'m2_metrics.json');m3=read(root/'m3_metrics.json')
    temporal=read(root/'temporal_results.json');tasks=read(root/'task_results.json');conditions=read(root/'condition_results.json')
    matrix=[];bounds={}
    for wid,g in workloads.items():
        q=g['quality_pairs'];n=q['all_requests'];u=q['UNKNOWN'];s=q['scored'];loss=q['loss_A_minus_C']
        bounds[wid]=dict(requests=n,scored=s,UNKNOWN=u,
            loss_A_minus_C_full_request_bounds=[(loss*s-u)/n,(loss*s+u)/n] if loss is not None and n else None,
            all_request_average_quality_budget_certified=False,
            reason='Known-answer average loss is observed; unscored requests remain UNKNOWN, not imputed correct.')
        ci=g['bootstrap']['speedup_CI95']
        matrix.append(dict(workload=g['name'],Native_E2E_ms=g['A']['E2E_ms']['mean'],R0_E2E_ms=g['C']['E2E_ms']['mean'],
            speedup=g['speedup'],latency_reduction=g['latency_reduction'],CI95=ci,
            end_to_end_evidence='BENEFIT_ON_EXPOSED_SUBSET' if ci[0]>1 else 'SLOWER_ON_EXPOSED_SUBSET' if ci[1]<1 else 'NO_RESOLVED_BENEFIT',
            conditional_accuracy_A=q['A_accuracy'],conditional_accuracy_C=q['C_accuracy'],
            scored=s,all_requests=n,new_severe=q['new_severe']))
    aa=m3['arms']['A'];cc=m3['arms']['C']
    language=dict(actual_zero_forward_skips=cc['complete_skipped_requests'],all_C_requests=cc['requests'],
        skip_fraction=cc['complete_skipped_requests']/cc['requests'],
        actual_forwards_A=aa['LLM_forwards'],actual_forwards_C=cc['LLM_forwards'],
        prefill_CUDA_ms_A=aa['Prefill_CUDA_ms']['mean']*aa['requests'],
        prefill_CUDA_ms_C=cc['Prefill_CUDA_ms']['mean']*cc['requests'],
        decode_CUDA_ms_A=aa['Decode_CUDA_ms']['mean']*aa['requests'],
        decode_CUDA_ms_C=cc['Decode_CUDA_ms']['mean']*cc['requests'],
        draft_candidates=cc['draft_candidates'],draft_accepted=cc['draft_accepted'],
        draft_acceptance=cc['speculative_token_acceptance'],actual_reused_KV_tokens=cc['reused_KV_tokens'])
    quality=dict(conditional_known_answer_accuracy_A=aa['LLM_fallback_quality']['scored_answer_accuracy'],
        conditional_known_answer_accuracy_C=(cc['FAST_quality']['scored_answer_accuracy']*cc['FAST_quality']['scored']+
            cc['LLM_fallback_quality']['scored_answer_accuracy']*cc['LLM_fallback_quality']['scored'])/
            (cc['FAST_quality']['scored']+cc['LLM_fallback_quality']['scored']),
        scored_per_arm=aa['LLM_fallback_quality']['scored'],requests_per_arm=aa['requests'],
        unknown_per_arm=aa['LLM_fallback_quality']['UNKNOWN'],full_request_bounds=bounds)
    risk=dict(status='FAILED_NEW_SEVERE_RELATIVE_TO_NATIVE' if verdict['new_severe_relative_to_Native'] else 'NOT_CERTIFIED',
        new_severe_observations=verdict['new_severe_relative_to_Native'],
        new_unique_frame_questions=verdict['new_unique_severe_frame_questions'],
        Native_absolute_severe=verdict['A_absolute_severe'],R0_absolute_severe=verdict['C_absolute_severe'],
        Native_unique_severe=verdict['A_unique_severe_frame_questions'],R0_unique_severe=verdict['C_unique_severe_frame_questions'],
        average_improvement_does_not_offset_new_severe=True,production_allowed=False,independent_TEST=False)
    write(root/'evidence_synthesis.json',dict(status='COMPLETED_WITH_RISK_FAILURE',
        FINAL_MAIN_CONFIGURATION='R0',configuration_sha256=verdict['frozen_config_sha256'],
        completion=completion,workloads=matrix,answer_quality=quality,language=language,risk=risk,
        sources={n:sha256(root/n) for n in ('workload_results.json','m1_metrics.json','m2_metrics.json','m3_metrics.json',
            'final_research_verdict.json','reanalysis_validation.json')},
        main_configuration_decision='Suitable as a fixed research experiment configuration with disclosed limitations; no strict quality, safety or independent generalization pass.'))
    lines=['# OmniDrive 冻结 R0 主实验最终报告','',
        '正式评估已完成。唯一主配置为 **FINAL_MAIN_CONFIGURATION = R0**；M1/M2/M3 算法源码、模型权重和原 Native 逻辑均未修改，原有研究策略与历史输出保持原字节。',
        '',f'范围：18 条历史暴露验证 Log、18 个从首帧到末帧的完整 Scene、723 个不同帧；正式 7,230 次模型调用，另有 8 次试运行。总计 7,238 次真实 RTX3090 模型调用、正式 11,226 个 A/C 问答请求。',
        '',f'冻结配置 SHA256：`{verdict["frozen_config_sha256"]}`。完整配置及全部默认项见 frozen_main_config.json；实际生效参数与逐请求哈希见 main/effective_runtime_configs.jsonl 和原始收据。',
        '', '**结论：高频单问、每帧三问和混合三问有完整 E2E 收益；低频 Every-4 更慢，Every-2 未得到明确收益。存在目标覆盖损失和新增严重错误，不能宣称严格质量或风险认证通过。**',
        '', '## 1. 五种负载相对 Native 的真实差异','']
    lines.append(table(['负载','Native E2E ms','R0 E2E ms','A/C','时延降低','A/C 95% Log CI','新增严重观测'],[
        [g['workload'],f(g['Native_E2E_ms']),f(g['R0_E2E_ms']),f(g['speedup']),f(100*g['latency_reduction'],2)+'%',
            '['+f(g['CI95'][0])+', '+f(g['CI95'][1])+']',g['new_severe']] for g in matrix]))
    lines+=['','主计时包含输入加载、预处理、H2D、问题编码，以及同步完成的整个模型调用与答案导出。无 QA 视觉帧和冷启动均计入。离线摘要、GT 评分、收据写盘和特征误差计算在计时外；问题服务时间不是视觉帧 E2E。每个请求使用相同初始种子和独立 RNG，A/C 图像摘要、Token、相机顺序、时间戳逐项一致。',
        '', '## 2. 哪种问答频率收益最大，状态维护是否抵消收益','',
        'Three-per-frame 收益最大。Every-4 的区间完全低于 1；Every-2 的区间跨过 1。它们说明模块节省无法自动覆盖低频问答时的持续状态维护。',
        '',f'M1 含适配器的平均成本为 {f(m1["cost_ms"]["mean"])} ms/视觉帧；Native 视觉 CUDA 均值 {f(m2["Native_Full_visual_CUDA_ms"]["mean"])} ms，增量视觉为 {f(m2["R0_incremental_visual_CUDA_ms"]["mean"])} ms。各模块均值是实际计时，但嵌套 host/CUDA 时间不能直接相加来解释 E2E。',
        '', '## 3. Cold Start、Transition、Steady State','']
    lines.append(table(['状态','帧观测/臂','Native E2E ms','R0 E2E ms','A/C'],[
        [state,g['A']['frames'],f(g['A']['E2E_ms']['mean']),f(g['C']['E2E_ms']['mean']),f(g['speedup'])] for state,g in temporal.items()]))
    lines+=['','状态由运行前固定的时间间隔、可见 GT 实例集合变化和匹配目标相对位移规则决定；每个 Scene/负载/执行臂均清空状态。帧数包含五种负载的实际重复观测，置信区间按 Log 聚类。',
        '', '## 4. Category、Existence、Position 的质量与时延','']
    lines.append(table(['任务','Native 已评分准确率','R0 已评分准确率','Native 问题服务 ms','R0 问题服务 ms','严重 A/C'],[
        [t,f(tasks[t]['A_quality']['scored_answer_accuracy']),f(tasks[t]['C_quality']['scored_answer_accuracy']),
            f(tasks[t]['service_latency_ms']['A']['mean']),f(tasks[t]['service_latency_ms']['C']['mean']),
            str(tasks[t]['A_quality']['absolute_severe'])+'/'+str(tasks[t]['C_quality']['absolute_severe'])] for t in ('category','existence','position')]))
    lines+=['',f'全部负载每臂 {quality["requests_per_arm"]} 个请求，仅 {quality["scored_per_arm"]} 个可评分，{quality["unknown_per_arm"]} 个 UNKNOWN。已评分事实准确率从 {f(quality["conditional_known_answer_accuracy_A"]*100,2)}% 到 {f(quality["conditional_known_answer_accuracy_C"]*100,2)}%；这不等于完整自由文本准确率或全请求质量认证。未评分请求的最坏/最好质量损失界保存在 evidence_synthesis.json，不把 UNKNOWN 自动计为正确。',
        '', '## 5. 动态、遮挡与交通复杂度','']
    condition_rows=[]
    for dim in ('dynamics','occlusion','density'):
        for label,g in conditions[dim].items():
            condition_rows.append([dim,label,g['A']['frames'],g['A']['logs'],f(g['speedup']),
                f(g['target_coverage']['C']['eligible_target_recall']),
                f(g['target_coverage']['A']['raw_visible_object_recall'])+'/'+f(g['target_coverage']['C']['raw_visible_object_recall'])])
    lines.append(table(['维度','条件','帧/臂','Log','A/C','C Eligible Recall','对象 Recall A/C'],condition_rows))
    lines+=['','这些是分别分层的观测，不是庞大维度交叉。没有标签的昼夜/天气不补造；完整相机可见性、目标变化及环境标签分析见 condition_comparison.md。小组样本不能证明任意环境下鲁棒。',
        '', '## 6. M1 是否提高后续目标覆盖','']
    c=m1['target_coverage']['S4'];d=m1['raw_detection']['S4']
    lines.append(f'在相同的三问请求下，Native 原始目标 proven Recall 为 {f(100*c["A"]["raw_recall"],2)}%，R0 原始检测与状态提交均为 {f(100*c["C"]["raw_recall"],2)}%，最终 M3 Eligible 为 {f(100*c["C"]["eligible_recall"],2)}%。原始可见对象 Recall 从 {f(100*d["A"]["groups"]["visible_GT"]["proven_class_recall"],2)}% 降至 {f(100*d["C"]["groups"]["visible_GT"]["proven_class_recall"],2)}%。')
    lines+=['','因此本轮不能声称世界状态提高了总目标覆盖。A/C 同时改变整条路径，不能因果隔离 M1；Native 不具备 M1 提交与 M3 Eligible 状态，相关单元为 N/A。歧义预测身份保留 UNKNOWN，proven Recall 为保守下界。',
        '', '## 7. M3 实际跳过和实际计算节省','',
        f'共 {language["actual_zero_forward_skips"]}/{language["all_C_requests"]} 个完整请求同时跳过 Prefill 和 Decode（{f(100*language["skip_fraction"],2)}%），每个均有零 Forward 证据，不重复计数。LLM Forward 从 {language["actual_forwards_A"]} 降到 {language["actual_forwards_C"]}。',
        '',f'提议 Token {language["draft_candidates"]}，接受 {language["draft_accepted"]}，接受率 {f(100*language["draft_acceptance"],2)}%。冻结 world_guided 分支的实际持久 Prefill KV 复用 Token 为 {language["actual_reused_KV_tokens"]}；保留代码不等于该分支实际执行了旧前缀规划器。',
        '',f'A/C 的累计 Prefill CUDA 分别为 {f(language["prefill_CUDA_ms_A"]/1000)} / {f(language["prefill_CUDA_ms_C"]/1000)} 秒，Decode 为 {f(language["decode_CUDA_ms_A"]/1000)} / {f(language["decode_CUDA_ms_C"]/1000)} 秒。这些是各臂真实累计时间，不是单模块反事实净收益。',
        '', '## 8. 视觉与地图质量的代价','',
        f'M2 返回特征 Relative L2 均值为 {f(m2["feature_error"]["relative_l2"]["mean"],6)}。精确窗口计算/复用、必须/普通刷新以及逐相机数据均来自执行收据，而非理论 FLOP。',
        '', '官方项目 Lane 评价覆盖这 723 帧：DET_l 与 3D Lane F-Score 分别报告；完整 150-Scene 检测 mAP/NDS、交通元素和拓扑质量仍为 NOT_CERTIFIED。子集结果不能替代全量官方评价。',
        '', '## 9. 严重错误与适用范围','',
        f'Native / R0 的绝对严重错误观测为 {risk["Native_absolute_severe"]}/{risk["R0_absolute_severe"]}；R0 相对 Native 新增 {risk["new_severe_observations"]} 次，涉及 {risk["new_unique_frame_questions"]} 个唯一帧—问题病例。不同负载的重复观测不冒充独立物理病例。平均正确率改善不能抵消这些新增错误。',
        '', '当前配置更适合高频、可由当前目标事实回答的多问负载。低频交互没有得到系统收益；可见目标覆盖下降、方位/最近目标错误、UNKNOWN 标签与历史暴露数据限制均保留。Position 前后、左右、对角、显式类别和可唯一匹配的最近目标错误见 position_error_breakdown.json。仅凭自由文本无法认证 Native 的物理最近目标身份，未观测到此类身份错误不能解释为零风险。',
        '', '## 10. 是否作为论文主实验配置','',
        '**可以作为固定完整 R0 的研究评估配置，性能主张限于本轮有证据的负载；不能作为无质量代价、严格风险通过或独立泛化通过的配置。** 所有 Log 都按历史暴露数据处理，生产默认关闭，不据此重新搜索参数。',
        '', f'完整性与归档重分析均通过；CPU {completion["CPU"]["passed"]} 项通过、{completion["CPU"].get("skipped",0)} 项跳过，真实 RTX3090 CUDA {completion["CUDA"]["passed"]} 项通过。历史 655 个输出文件及所有原始权重保留原字节。六张主表、机器可读 CSV/JSON、PDF/SVG 图和复现命令均在本目录。',
        '', '完整复现步骤见 REPRODUCE.md；本报告可通过 `tools.worldstream_m123.frozen_main_synthesis` 从已验证结果重新生成。']
    (root/'FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('FINAL_EVIDENCE_SYNTHESIS_WRITTEN')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();synthesize(a.root)


if __name__=='__main__':main()
