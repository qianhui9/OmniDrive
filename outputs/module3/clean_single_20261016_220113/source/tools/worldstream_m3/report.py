"""Generate M3 JSON/Markdown/figures only from recorded CUDA experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
from .experiment import write, stats
from worldstream.m3.task_conditioner import task_type


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def generated_count(row):
    return sum(max(0,len(x[0])-1) for x in row['generated_ids'])


def metric_summary(rows):
    output = {}
    modes = list(dict.fromkeys(r['mode'] for r in rows))
    for mode in modes:
        picked = [r for r in rows if r['mode'] == mode]
        result = dict(frames=len(picked), questions=sum(len(r['questions']) for r in picked))
        result['e2e_wall_ms'] = stats([r['wall_ms'] for r in picked])
        for stage in ('prefill','decode','eva','multimodal_preparation','detection','map'):
            result[stage+'_gpu_ms'] = stats([r['components'].get(stage,{}).get('gpu_ms',0.) for r in picked])
            result[stage+'_calls'] = sum(r['components'].get(stage,{}).get('calls',0) for r in picked)
        result['llm_gpu_ms'] = stats([r['components']['prefill']['gpu_ms'] + r['components'].get('decode',{}).get('gpu_ms',0.) for r in picked])
        result['peak_allocated_gib'] = stats([r['peak_allocated']/1024**3 for r in picked])
        result['peak_reserved_gib'] = stats([r['peak_reserved']/1024**3 for r in picked])
        result['generated_tokens'] = sum(generated_count(r) for r in picked)
        result['decode_tokens_per_second'] = 1000*sum(r['components'].get('decode',{}).get('calls',0) for r in picked)/sum(r['components'].get('decode',{}).get('gpu_ms',0) for r in picked)
        result['actual_reused_tokens'] = sum(d.get('reused_tokens',0) for r in picked for d in r['diagnostics'].get('requests',[]))
        result['cache_max_mib'] = max((r['diagnostics'].get('cache_bytes',0)/1024**2 for r in picked),default=0.)
        result['m1_total_ms'] = stats([r['m1'].get('m1_total_including_adapters_ms',0.) for r in picked])
        result['cold_sequence_wall_ms'] = stats([r['wall_ms'] for r in picked if r['cold_sequence']])
        result['later_frame_wall_ms'] = stats([r['wall_ms'] for r in picked if not r['cold_sequence']])
        output[mode] = result
    baseline = output['baseline']
    for mode, result in output.items():
        result['prefill_speedup'] = baseline['prefill_gpu_ms']['p50']/result['prefill_gpu_ms']['p50']
        result['decode_speedup'] = baseline['decode_gpu_ms']['p50']/result['decode_gpu_ms']['p50']
        result['llm_latency_reduction'] = 1-result['llm_gpu_ms']['p50']/baseline['llm_gpu_ms']['p50']
        result['e2e_speedup'] = baseline['e2e_wall_ms']['p50']/result['e2e_wall_ms']['p50']
    return output


def quality_summary(rows):
    baselines = {(r['repeat'],r['index']):r for r in rows if r['mode']=='baseline'}
    results = {}
    for r in rows:
        if r['mode']=='baseline': continue
        b=baselines[(r['repeat'],r['index'])]
        record=results.setdefault(r['mode'],dict(frames=0,frame_text_equal=0,questions=0,token_ids_equal=0,tasks={},failures=[]))
        record['frames']+=1;record['frame_text_equal']+=int(r['answers']==b['answers'])
        for i,q in enumerate(r['questions']):
            equal=r['generated_ids'][i]==b['generated_ids'][i]
            record['questions']+=1;record['token_ids_equal']+=int(equal)
            category=task_type(q)
            task=record['tasks'].setdefault(category,dict(questions=0,token_ids_equal=0))
            task['questions']+=1;task['token_ids_equal']+=int(equal)
            if not equal:
                record['failures'].append(dict(index=r['index'],repeat=r['repeat'],question=q,
                    baseline=b['answers'][i],actual=r['answers'][i],
                    gt=r['ground_truth'].get(q),baseline_token_ids=b['generated_ids'][i],actual_token_ids=r['generated_ids'][i]))
    for record in results.values():
        record['frozen_token_target_pass']=record['questions']==record['token_ids_equal']
    return results


def llm_quality(rows, audit):
    output={}
    baseline={r['token']:r for r in rows if r['mode']=='full'}
    inputs={r['token']:[r['expanded_tokens'][i] for i in r['selected_indices']] for r in audit}
    for r in rows:
        label=f"{r['mode']}_{r['selector']}_{r['ratio']}"
        record=output.setdefault(label,dict(questions=0,token_ids_equal=0,f1=[],numeric_mae=[],input_tokens=0,visual_tokens=0,wall_ms=[]))
        record['wall_ms'].append(r['wall_ms'])
        for i,q in enumerate(r['questions']):
            record['questions']+=1; record['token_ids_equal']+=int(r['token_ids'][i]==baseline[r['token']]['token_ids'][i])
            if r['quality'][i] is not None:
                record['f1'].append(r['quality'][i]['token_f1'])
                if r['quality'][i]['numeric_mae'] is not None: record['numeric_mae'].append(r['quality'][i]['numeric_mae'])
            ds=r['diagnostics'].get('requests',[])
            record['input_tokens']+=ds[i]['input_tokens'] if ds else inputs[r['token']][i]
            record['visual_tokens']+=ds[i]['visual_tokens'] if ds else 513
    for record in output.values():
        record['mean_gt_token_f1']=float(np.mean(record['f1'])) if record['f1'] else None
        record['mean_numeric_mae']=float(np.mean(record['numeric_mae'])) if record['numeric_mae'] else None
        record['carrier_reduction']=1-record['visual_tokens']/(513*record['questions'])
        record['strict_token_retention']=record['token_ids_equal']/record['questions']
        record['wall_ms']=stats(record['wall_ms'])
    return output


def make_figures(dest, rows, audit, latency, quality, llm_rows, llm_summary, prefix_rows, watermark=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    figure_dir=dest/'figures';figure_dir.mkdir(exist_ok=True)
    manifest={}
    def save(fig,name,source):
        if watermark:
            fig.text(.5,1.01,watermark,ha='center',va='bottom',color='#a12525',fontsize=10)
        for extension in ('png','pdf'):fig.savefig(figure_dir/(name+'.'+extension),dpi=300)
        plt.close(fig);manifest[name]=source
    modes=list(latency)
    fig,ax=plt.subplots(figsize=(7,3.2))
    pre=[latency[m]['prefill_gpu_ms']['p50'] for m in modes];dec=[latency[m]['decode_gpu_ms']['p50'] for m in modes]
    ax.bar(modes,pre,label='Prefill');ax.bar(modes,dec,bottom=pre,label='Decode')
    ax.set(ylabel='GPU interval per frame (ms)',title='M3(a) Five-question cost breakdown');ax.legend();save(fig,'m3a_cost_breakdown','frames.jsonl; measured CUDA intervals; stacked component medians need not equal total median')
    fig,ax=plt.subplots(figsize=(5,3.2));same=[v for r in audit for v in r['pairwise_exact_lcp']]
    cross=[r['cross_frame_first_question_lcp'] for r in audit if r['cross_frame_first_question_lcp'] is not None]
    ax.hist([same,cross],bins=24,label=['Same frame','Cross frame'],log=True)
    ax.set(xlabel='Exact expanded prefix tokens',ylabel='Pair count (log)',title='M3(b) Exact prefix distribution');ax.legend();save(fig,'m3b_prefix_distribution','input_audit.json; all original questions, not just timed subset')
    if llm_summary:
        fig,ax=plt.subplots(figsize=(6,3.5))
        for label,r in llm_summary.items():
            if r['mean_gt_token_f1'] is None:continue
            ax.scatter(100*r['carrier_reduction'],r['mean_gt_token_f1'],label=label)
        ax.set(xlabel='Visual carrier reduction (%)',ylabel='Mean GT token F1 (diagnostic)',title='M3(c) Compact input versus QA diagnostic');ax.legend(fontsize=7);save(fig,'m3c_compression_quality','llm_probes.jsonl; unchanged GT; F1 is not safety/semantic accuracy')
    fig,ax=plt.subplots(figsize=(6,3.5))
    by_token={r['token']:r for r in audit}
    for mode in modes:
        picked=[r for r in rows if r['mode']==mode]
        x=[];y=[]
        for r in picked:
            ds=r['diagnostics'].get('requests',[])
            tokens=sum(d['effective_prefill_tokens'] for d in ds) if ds else sum(by_token[r['token']]['expanded_tokens'][i] for i in by_token[r['token']]['selected_indices'])
            x.append(tokens);y.append(r['components']['prefill']['gpu_ms'])
        ax.scatter(x,y,label=mode,s=12,alpha=.6)
    ax.set(xlabel='Effective prefill tokens per frame',ylabel='Prefill GPU interval (ms)',title='M3(d) Measured prefill work');ax.legend(fontsize=8);save(fig,'m3d_prefill_tokens','frames.jsonl + input_audit.json; no fitted/synthetic curve')
    if prefix_rows:
        fig,ax=plt.subplots(figsize=(5,3.5))
        for mode in ('independent','text_prefix','same_frame_visual'):
            picked=[r for r in prefix_rows if r['mode']==mode and not r['warmup']]
            x=[sum(d['reused_tokens'] for d in r['requests'])/sum(d['input_tokens'] for d in r['requests']) for r in picked]
            ax.scatter(x,[r['wall_ms'] for r in picked],label=mode,s=10,alpha=.6)
        ax.set(xlabel='Actual reused KV tokens / input tokens',ylabel='Isolated prefill probe wall (ms)',title='M3(e) Prefix reuse and prefill cost');ax.legend(fontsize=8);save(fig,'m3e_reuse_latency','prefix_ablation.json; one generated token isolates prefill, not full E2E')
    fig,ax=plt.subplots(figsize=(7,3.5))
    for j,key in enumerate(('signature_cpu_ms','selection_cpu_ms','preparation_cpu_ms','match_cpu_ms')):
        values=[float(np.median([sum(d.get(key,0.) for d in r['diagnostics'].get('requests',[])) for r in rows if r['mode']==m])) for m in modes]
        ax.bar(np.arange(len(modes))+(j-1.5)*.18,values,width=.18,label=key.replace('_cpu_ms',''))
    ax.set(xticks=np.arange(len(modes)),xticklabels=modes,ylabel='CPU admission wall per frame (ms)',title='M3(f) Runtime admission overhead (GPU shown in a)');ax.legend(fontsize=8);save(fig,'m3f_runtime_breakdown','frames.jsonl; CPU wall is non-additive with asynchronous GPU intervals')
    fig,ax=plt.subplots(figsize=(7,3.2));bars=ax.bar(modes,[latency[m]['e2e_speedup'] for m in modes])
    for m,b in zip(modes,bars):
        if m in quality and not quality[m]['frozen_token_target_pass']:b.set_hatch('//')
    ax.axhline(1,color='black',lw=.8);ax.set(ylabel='Baseline P50 / method P50',title='M3(g) Full VLM speedup; hatched = quality target failed');save(fig,'m3g_e2e_speedup','frames.jsonl; all visual/state/cache/language included, warmup excluded')
    write(dest/'figures/manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='outputs/module3/run04');p.add_argument('--output',default='outputs/module3')
    p.add_argument('--single-run')
    args=p.parse_args();run=Path(args.run);dest=Path(args.output)
    all_rows=read_lines(run/'frames.jsonl');rows=[r for r in all_rows if not r['warmup']]
    audit=json.loads((run/'input_audit.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    environment=json.loads((run/'environment.json').read_text())
    expected=sum(min(environment['argv']['frames'],len(s['indices'])) for s in protocol['scenes'])*protocol['repeats']*len(protocol['modes'])
    if len(rows)!=expected:raise ValueError(f'Incomplete main run: {len(rows)}/{expected} measured frames')
    latency=metric_summary(rows);quality=quality_summary(rows)
    llm_rows=read_lines(run/'llm_probes.jsonl') if (run/'llm_probes.jsonl').exists() else []
    lquality=llm_quality(llm_rows,audit)
    audit_by_token={r['token']:r for r in audit}
    for mode,summary in latency.items():
        picked=[r for r in rows if r['mode']==mode]
        original_tokens=actual_tokens=effective_tokens=visual_tokens=independent=0
        overhead=[]
        for row in picked:
            audited=audit_by_token[row['token']]
            original_tokens+=sum(audited['expanded_tokens'][i] for i in audited['selected_indices'])
            ds=row['diagnostics'].get('requests',[])
            if ds:
                actual_tokens+=sum(d['input_tokens'] for d in ds)
                effective_tokens+=sum(d['effective_prefill_tokens'] for d in ds)
                visual_tokens+=sum(d['visual_tokens'] for d in ds)
                independent+=sum(not d['reused_tokens'] and not d.get('selection',{}).get('admitted',False) for d in ds)
            else:
                actual_tokens+=sum(audited['expanded_tokens'][i] for i in audited['selected_indices'])
                effective_tokens+=sum(audited['expanded_tokens'][i] for i in audited['selected_indices'])
                visual_tokens+=513*len(row['questions']);independent+=len(row['questions'])
            overhead.append(sum(d.get('match_cpu_ms',0.) for d in ds))
        summary.update(original_input_tokens=original_tokens,actual_input_tokens=actual_tokens,
            effective_prefill_tokens=effective_tokens,exact_prefix_reuse_ratio=summary['actual_reused_tokens']/actual_tokens,
            carrier_reduction_ratio=1-visual_tokens/(513*summary['questions']),
            input_token_reduction_ratio=1-actual_tokens/original_tokens,
            independent_full_prefill_rate=independent/summary['questions'],cache_lookup_cpu_ms=stats(overhead))
        probe=[r for r in llm_rows if (r['mode']=='full' if mode=='baseline' else r['mode']==mode)
               and (r['selector'] in ('task','full'))]
        summary['whole_llm_forward_probe'] = dict(source=str(run/'llm_probes.jsonl'),
            frames=len(probe),prefill_gpu_ms=stats([r['full_forward_gpu_ms']['prefill'] for r in probe]),
            decode_gpu_ms=stats([r['full_forward_gpu_ms']['decode'] for r in probe]),
            complete_llm_wall_ms=stats([r['wall_ms'] for r in probe])) if probe else None
    correctness=json.loads((run/'kv_correctness.json').read_text())
    prefix=json.loads((run/'prefix_ablation.json').read_text())
    if (run/'qa_ground_truth_analysis.json').exists():
        write(dest/'qa_ground_truth_analysis.json',json.loads((run/'qa_ground_truth_analysis.json').read_text()))
    write(dest/'baseline_profile.json',dict(source=str(run),summary=latency['baseline'],
        e2e_gpu_interval_scope=environment.get('timer_scope','Transformer decoder; separate LLM probes include final vocabulary projection'),
        whole_llm_forward_probe=[r for r in llm_rows if r['mode']=='full'],
        rows=[r for r in rows if r['mode']=='baseline']))
    write(dest/'token_statistics.json',dict(source=str(run/'input_audit.json'),rows=audit,
        deployment_questions_per_frame=1,experiment_questions_per_frame=5,
        all_question_counts=stats([r['actual_questions'] for r in audit]),
        raw_text_tokens=stats([n for r in audit for n in r['text_tokens']]),
        expanded_tokens=stats([n for r in audit for n in r['expanded_tokens']]),
        same_frame_lcp=stats([n for r in audit for n in r['pairwise_exact_lcp']]),
        cross_frame_lcp=stats([r['cross_frame_first_question_lcp'] for r in audit if r['cross_frame_first_question_lcp'] is not None]),
        question_occurrences={q:sum(q in r['questions'] for r in audit) for q in sorted({q for r in audit for q in r['questions']})}))
    write(dest/'compact_input_results.json',dict(summary=lquality,raw_source=str(run/'llm_probes.jsonl'),strict_online_enabled=False))
    write(dest/'prefix_match_results.json',dict(audit_source=str(run/'input_audit.json'),ablation=prefix))
    write(dest/'kv_correctness.json',correctness)
    write(dest/'exact_failure_diagnosis.json',json.loads((run/'exact_failure_diagnosis.json').read_text()))
    write(dest/'quality_results.json',dict(paired_e2e=quality,llm=lquality,
        metric_limits='F1/normalized exact and numeric-list error are diagnostics, not relation/safety accuracy; frozen token equality separately enforced',
        extra_source=str(run/'extended_quality.json')))
    write(dest/'latency_results.json',latency)
    write(dest/'memory_profile.json',{m:{k:v for k,v in r.items() if k in ('peak_allocated_gib','peak_reserved_gib','cache_max_mib')} for m,r in latency.items()})
    write(dest/'ablation_results.json',dict(latency=latency,quality=quality,prefix_probe=prefix,
        modes=dict(baseline='Original OmniDrive, fresh outputs',m12='M1 full continuity + existing M2 camera auto',
            compact='M3 task compact, offline approximation',exact='M3 same-frame exact prefix, offline cost override',
            combined='M3 approximate task compact + exact-prefix checking',m123='M1 full continuity + M2 camera auto + M3 same-frame exact')))
    # No policy is auto-approved just by an aggregate speedup. Per-shape independent
    # calibration and held-out quality evidence must be reviewed before admission.
    write(dest/'cost_table.json',dict(approved=[],reason='No automatically published deployment calibration; experimental latency evidence remains separate'))
    write(dest/'quality_certificate.json',dict(approved=[],reason='Token-deletion candidates are not independently quality-certified'))
    make_figures(dest,rows,audit,latency,quality,llm_rows,lquality,prefix)
    report(dest,run,rows,audit,latency,quality,lquality,correctness)
    single_summary=None
    if args.single_run:
        single_summary=single_report(Path(args.single_run),dest)
    write(dest/'delivery_status.json',dict(status='implemented_gpu_protocol_complete',
        final_gpu_experiments_complete=True,source=str(run),single_question_source=args.single_run,
        auto_approved=False,compact_approved=False,
        exact_frozen_token_target_pass=quality['exact']['frozen_token_target_pass']))


def single_report(source,dest):
    all_rows=read_lines(source/'frames.jsonl');rows=[r for r in all_rows if not r['warmup']]
    latency=metric_summary(rows);quality=quality_summary(rows)
    baselines={(r['repeat'],r['index']):r for r in rows if r['mode']=='baseline'}
    perception={}
    for row in rows:
        if row['mode']=='baseline':continue
        baseline=baselines[(row['repeat'],row['index'])]
        r=perception.setdefault(row['mode'],dict(checks=0,exact=0,failures=[]))
        r['checks']+=1
        same=row.get('perception_digest')==baseline.get('perception_digest') and 'perception_digest' in row
        r['exact']+=int(same)
        if not same:r['failures'].append(dict(repeat=row['repeat'],index=row['index']))
    result=dict(source=str(source),generation='Original deployment Sampling 0.1/top_p 0.75, fixed RNG, max_new_tokens 320',
        latency=latency,quality=quality,raw_perception_and_memory=perception,
        interpretation='One question has no same-frame shared-prefix opportunity; no shortened generation or answer-file skip')
    write(dest/'single_question_results.json',result)
    with (dest/'experiment_report.md').open('a') as handle:
        handle.write('\n## 原始单问题部署补充\n\n')
        handle.write(f'实际默认 Sampling、固定 RNG、320 Token 上限，数据源 `{source}`。原始精度 Detection/Map/Temporal Memory 数据保存至 perception/，每个输出与状态均按原 dtype/shape/内容 SHA 比较。\n\n')
        handle.write('| 方法 | E2E P50/P95 ms | 实际复用 KV Token | 输出 Token 一致 | 感知/Memory 内容一致 |\n|---|---:|---:|---:|---:|\n')
        for mode,r in latency.items():
            q=quality.get(mode);m=perception.get(mode)
            qs=f'{q["token_ids_equal"]}/{q["questions"]}' if q else 'baseline'
            ms=f'{m["exact"]}/{m["checks"]}' if m else 'baseline'
            handle.write(f'| {mode} | {r["e2e_wall_ms"]["p50"]:.2f}/{r["e2e_wall_ms"]["p95"]:.2f} | {r["actual_reused_tokens"]} | {qs} | {ms} |\n')
        handle.write('\n单问题结果不能套用五问题的共享收益；实验 exact 单问题仍捕获缓存，其额外成本如实计入，未校准 Auto 则回退原路径。\n')
    return result


def report(dest,run,rows,audit,latency,quality,lquality,correctness):
    baseline=latency['baseline'];exact=latency['exact'];combined=latency['m123']
    counts=[r['actual_questions'] for r in audit]
    tokens=[x for r in audit for x in r['text_tokens']]
    expanded=[x for r in audit for x in r['expanded_tokens']]
    lcp=[x for r in audit for x in r['pairwise_exact_lcp']]
    cross=[r['cross_frame_first_question_lcp'] for r in audit if r['cross_frame_first_question_lcp'] is not None]
    lines=['# M3 — State-Conditioned LLM Execution: 实验交付报告','',
        f'真实 RTX 3090、原始 OmniDrive/LoRA 权重、Transformers 4.31.0；数据源 `{run}`。代码已集成，Compact 默认不获在线批准；Exact 的合法前缀匹配与实际跳算独立报告。','',
        '## 协议与边界','',
        f'六个真实 nuScenes 场景，每场景两个连续帧；每帧从全部原始问题中按问题文本确定五个任务覆盖问题。全部原始问题数 {min(counts)}–{max(counts)}，默认部署配置实际只有一个 planning 问题。一次整序列预热、两次计时，六方法按场景/重放轮换；每方法 {baseline["frames"]} 个计时帧、{baseline["questions"]} 个问题。保留原始 Question/GT；未重新训练、修改权重、改变生成长度上限或复用回答文件。',
        '所有 E2E 方法保留 max_new_tokens=320；为逐 Token 正确性，性能对照使用 Greedy，部署默认 Sampling 不变并另测固定 RNG。CPU 数据加载/scatter 和结果落盘在 E2E 计时外；当前视觉、heads、M1/M2、选择、匹配、KV 拷贝和真实语言生成均在计时内。轻量 CUDA profiling wrapper 对所有方法一致。P95 为有限样本的经验分位，不是服务 SLA。正式 E2E 的 Prefill/Decode GPU interval 覆盖原始 CausalLM.forward（包括词表投影），独立 llm_probes.jsonl 的 full_forward_gpu_ms 也包含最终投影；探索 run03 仅测 decoder 的区间不能与正式结果混加。',
        '冷缓存列表示序列首帧，并非进程加载冷启动；同帧模式每帧第一个问题冷 Prefill，后续问题共享。跨帧探针和单 Token Prefill 探针不作为完整服务结果。', '',
        '## 原始输入与 Carrier 审计','',
        f'每帧 256 Object + 1 Ego + 256 Map = **513** 个视觉 Token，宽度 4096；原始文本 Token 数 {min(tokens)}–{max(tokens)}，实际展开 Prefill {min(expanded)}–{max(expanded)}。图像标记均在原始索引 {sorted({x for r in audit for z in r["image_positions"] for x in z})}；同帧实际 Embedding LCP {min(lcp)}–{max(lcp)}，跨帧首问题 LCP {min(cross)}–{max(cross)}。没有将文本相同或 M1 Anchor 身份当作视觉 Embedding 等价证据。',
        'Carrier 是独立 learned extra queries，先于预测 queries 被分离；proposal Top-K、框和车道输出与 Carrier 没有一对一可证索引。M1 可约束来源和时效，不能可靠地指定删除某个对象 Carrier。详见 carrier_semantics_audit.md。','',
        '## 完整 VLM 性能','',
        '| 方法 | Prefill GPU P50/P95 ms | Decode GPU P50/P95 ms | E2E wall P50/P95 ms | Prefill 加速 | E2E 加速 | Token 一致 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in latency.items():
        fmt=lambda field:f'{r[field]["p50"]:.2f}/{r[field]["p95"]:.2f}'
        q=quality.get(mode);same=f'{q["token_ids_equal"]}/{q["questions"]}' if q else 'baseline'
        lines.append(f'| {mode} | {fmt("prefill_gpu_ms")} | {fmt("decode_gpu_ms")} | {fmt("e2e_wall_ms")} | {r["prefill_speedup"]:.3f}× | {r["e2e_speedup"]:.3f}× | {same} |')
    lines += ['',f'Baseline 的每帧累计 Prefill 为 {baseline["prefill_gpu_ms"]["p50"]:.2f} ms，平均每问题约 {baseline["prefill_gpu_ms"]["p50"]/5:.2f} ms；不能将一个问题的时间当作五问题总成本。Exact 累计复用 {exact["actual_reused_tokens"]} 个实际 KV Token；单独 M3 的 LLM GPU 时延下降 {100*exact["llm_latency_reduction"]:.2f}%。',
        'Decode 加速必须结合输出长度判断。精确路径若 Token IDs 相同，则输出长度相同；近似路径输出变化后的 Decode 降低不能宣称每 Token 更快。latency_results.json 保留输出 Token 总数、真实 Decode 调用数和 tokens/s。Prefill 指实际原始 Transformer interval；匹配、复制和缓存维护的全部净成本仍由 E2E wall 覆盖。','',
        '## 正确性、近似质量与 Auto 决策','',
        f'Exact 配对问题严格一致：{quality["exact"]["token_ids_equal"]}/{quality["exact"]["questions"]}；冻结目标通过：{quality["exact"]["frozen_token_target_pass"]}。Greedy/Sampling 的逐 Token/Logits 探针、变化输入反例和缓存隔离见 kv_correctness.json。相同完整输入的 Profiler Prefill 长度记录为 {correctness["identical_request_profiler"]["forward_prefill_lengths"]}；后续实际 Decode 仍执行。原始 trace 在 {run}/traces/identical_prefix.json。失败请求的逐层 KV 和 Logits 数值差异见 exact_failure_diagnosis.json；有合法依赖不代表不同 FP16 Shape 的输出逐位相同，失败不获 Auto 批准。',
        'Task candidate 仅对 object_relation 问题保留全部 Object/Ego 和一半 Map；地图、全局安全、未知任务保持完整。Top-K（norm，不是框置信度）和随机保序为独立离线对照。删 Token 改变后续位置与注意力，是近似输入；原始顺序不提供输出等价保证。',
        f'Compact 完整 VLM 严格 Token 一致：{quality["compact"]["token_ids_equal"]}/{quality["compact"]["questions"]}；Combined：{quality["combined"]["token_ids_equal"]}/{quality["combined"]["questions"]}。失败逐项保存在 quality_results.json，未放宽冻结目标。',
        '真实 GT 的归一化 exact、Token F1 和对齐数值列表误差保存在 llm_probes.jsonl。F1 不是语义 QA Accuracy，数值数量不匹配不会被隐去为零误差，也不能将这些指标称为驾驶安全认证。扩展任务覆盖以及未覆盖域见 extended_quality.json；覆盖不足的任务不获在线批准。',
        '默认 Auto 只接受匹配设备/后端/模型版本/Shape 的真实成本证据，以及紧凑输入独立 held-out 质量证据。本交付不因总体均值较好就伪造每 Shape 的生产证书；cost_table.json/quality_certificate.json 的 approved 为空。Auto 因未校准直接回退原始生成，离线 exact 的速度不冒充已上线 Auto 收益。','',
        '## 显存与瓶颈','',
        f'Exact 最大持久 Cache {exact["cache_max_mib"]:.2f} MiB；完整请求 peak allocated P50/P95 {exact["peak_allocated_gib"]["p50"]:.3f}/{exact["peak_allocated_gib"]["p95"]:.3f} GiB。预算包含保存的 KV、输入和末 Token logits；每个活跃 Decode 的克隆分支是额外 workspace，在 diagnostics 中分列。',
        f'M1+M2+M3 的 E2E P50 加速为 {combined["e2e_speedup"]:.3f}×；M1 总维护 P50 为 {combined["m1_total_ms"]["p50"]:.2f} ms。M2 真实移动场景是否命中以 frames.jsonl 的实际 diagnostics 为准，不推断发生视觉复用。',
        '主要收益机会是同帧长前缀；跨帧视觉变化仅保留短文本前缀。Decode 仍主导完整时延，KV 克隆/签名/匹配也消耗净收益。紧凑视觉表示缺乏对象级语义映射和训练适应，是近似路径的主要质量限制。','',
        '## 最终问题逐项回答','',
        f'1. 513 个视觉 Token；文本 {min(tokens)}–{max(tokens)}；展开 {min(expanded)}–{max(expanded)}。默认一问题与实验五问题明确区分。',
        f'2. 有，同帧 LCP {min(lcp)}–{max(lcp)}，由实际展开 Embeddings 认证。',
        '3. 有，真实 suffix-only forward 只接收未命中后缀；完全相同请求不再执行 Transformer Prefill，Profiler 和调用长度证据保留。',
        '4. 有，task 候选把适用问题视觉 Carrier 由 513 减至 385；Top-K/随机提供更大比例离线对照。',
        '5. 未训练的删除会改变部分真实答案；失败不获在线批准。具体数值和失败文本见质量 JSON。',
        '6. object_relation 是候选研究域，仍须逐任务质量验证；地图、安全/全局/未知问题保留完整，未证明任何任务可无条件压缩上线。',
        '7. M1 能提供场景/时效约束，尚无 Anchor 到 learned Carrier 的可靠对应，不能据其删除单个 Carrier。',
        f'8. Exact Prefill P50 {baseline["prefill_gpu_ms"]["p50"]:.2f}→{exact["prefill_gpu_ms"]["p50"]:.2f} ms；Decode {baseline["decode_gpu_ms"]["p50"]:.2f}→{exact["decode_gpu_ms"]["p50"]:.2f} ms。',
        f'9. 离线 M3 Exact 多问题 E2E {exact["e2e_speedup"]:.3f}×，质量冻结目标通过={quality["exact"]["frozen_token_target_pass"]}；Auto 未发布校准证书，不混为已上线结果。',
        f'10. 离线 M1/M2/M3 E2E {combined["e2e_speedup"]:.3f}×；仍须结合 P95 和逐问题一致性判断。',
        '11. Decode 占比、单问题没有同帧共享、跨帧只剩短文本前缀、输入压缩质量限制，以及 M1/Cache 成本共同限制净收益；不将输出变短当作 Decode 算法加速。','',
        '## 可复现性与失败记录','',
        f'命令见 commands.sh，生产接口见 docs/worldstream_m3.md；源码快照见 {run}/source/，后加载探针的真实 SHA 另见 llm_probe_source.json。run01 的加载中断、run02 的跨帧策略探索和 run03 的实验包装器引用循环/OOM 均保留且不计入正式表；已修复并新增生命周期回归。原始 M1/M2 交付不被覆盖，本次新产生的 M1 维护日志隔离在 module3 内。','',
        '![M3(a)](figures/m3a_cost_breakdown.png)','![M3(b)](figures/m3b_prefix_distribution.png)',
        '![M3(c)](figures/m3c_compression_quality.png)','![M3(d)](figures/m3d_prefill_tokens.png)',
        '![M3(e)](figures/m3e_reuse_latency.png)','![M3(f)](figures/m3f_runtime_breakdown.png)','![M3(g)](figures/m3g_e2e_speedup.png)']
    (dest/'experiment_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
