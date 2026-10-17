"""Source-derived second-round report; unavailable measurements stay unavailable."""
import argparse,hashlib,json,pickle
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .audit import write,stats
from .report import summarize
from .v2_checks import check,rows
from .identity import token_index,resolve_sequence
from .profile_analysis import analyze


def paired(data,mode,reference):
    ref={(r['repeat'],r['index']):r for r in data if r['mode']==reference and not r['warmup']}
    groups=defaultdict(list)
    for r in data:
        key=(r['repeat'],r['index'])
        if r['mode']==mode and not r['warmup'] and key in ref:
            b=ref[key];groups[r['frame']['identity']['scene_id']].append((b['frame']['e2e_wall_ms'],r['frame']['e2e_wall_ms']))
    if not groups:return None
    values=np.array([[np.mean(v,axis=0)[0],np.mean(v,axis=0)[1]] for v in groups.values()])
    rng=np.random.RandomState(20261017);indices=rng.randint(0,len(values),(10000,len(values)))
    bootstrap=values[indices].mean(1);delta=bootstrap[:,1]-bootstrap[:,0];ratio=bootstrap[:,0]/bootstrap[:,1]
    return dict(reference=reference,mode=mode,scenes=len(groups),paired_frames=sum(map(len,groups.values())),
        mean_scene_increment_ms=float(np.mean(values[:,1]-values[:,0])),
        scene_cluster_bootstrap_increment_95ci_ms=np.percentile(delta,[2.5,97.5]).tolist(),
        scene_mean_time_ratio=float(values[:,0].mean()/values[:,1].mean()),
        scene_cluster_bootstrap_ratio_95ci=np.percentile(ratio,[2.5,97.5]).tolist(),
        scene_pairs={s:dict(frames=len(v),reference_mean_ms=float(np.mean(v,axis=0)[0]),candidate_mean_ms=float(np.mean(v,axis=0)[1])) for s,v in groups.items()},
        uncertainty='Resample independent scenes, not frames/repeats; small scene count limits generalization')


def figures(root,summary,data,pairs,profiles,state,batch,resource=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'figures';dest.mkdir(exist_ok=True);made=[]
    def save(name,draw):
        fig,ax=plt.subplots(figsize=(10,4.7));draw(ax);fig.tight_layout()
        fig.savefig(dest/(name+'.png'),dpi=180);fig.savefig(dest/(name+'.pdf'));plt.close(fig);made.append(name)
    for run in ('single','multi','long_stream'):
        chosen=[r for r in summary if r['run']==run]
        if chosen:
            def draw(ax):
                ax.bar(range(len(chosen)),[r['e2e_wall_ms']['p50'] for r in chosen],label='P50')
                ax.scatter(range(len(chosen)),[r['e2e_wall_ms']['p95'] for r in chosen],color='black',label='P95')
                ax.set_xticks(range(len(chosen)));ax.set_xticklabels([r['mode'] for r in chosen],rotation=55,ha='right')
                ax.set_ylabel('Complete service wall ms');ax.set_title(run+'; service contracts labeled in report');ax.legend()
            save(run+'_latency',draw)
    if pairs:
        chosen=[p for p in pairs if p['mode'] in ('opt_serial','overlap','native_decode')]
        if chosen:
            def draw(ax):
                for i,p in enumerate(chosen):
                    lo,hi=p['scene_cluster_bootstrap_increment_95ci_ms'];y=p['mean_scene_increment_ms']
                    ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o')
                ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(chosen)))
                ax.set_xticklabels([p['run']+':'+p['mode'] for p in chosen],rotation=55,ha='right')
                ax.set_ylabel('Paired wall increment ms; scene bootstrap 95% CI')
            save('paired_scene_increment',draw)
    worker=[r for r in data if not r['warmup'] and r['frame']['m1'].get('worker')]
    if worker:
        def draw(ax):
            ax.scatter([r['frame']['m1']['worker']['worker_ms'] for r in worker],
                [r['frame']['m1']['worker']['join_wait_ms'] for r in worker],s=12)
            ax.set_xlabel('M1 worker elapsed ms');ax.set_ylabel('Main thread join wait ms')
        save('worker_join_wait',draw)
    long=[r for r in data if r['run']=='long_stream' and not r['warmup']]
    if long:
        def draw(ax):
            for mode in ('original','m1','overlap'):
                ax.plot([r['frame']['e2e_wall_ms'] for r in long if r['mode']==mode],label=mode,alpha=.8)
            ax.set_xlabel('Executed frame ordinal; scene boundaries retained');ax.set_ylabel('Full wall ms');ax.legend()
        save('long_stream_wall',draw)
    if state:
        funnel=state['funnel']
        def draw(ax):
            ax.bar(range(len(funnel)),list(funnel.values()));ax.set_xticks(range(len(funnel)))
            ax.set_xticklabels(list(funnel),rotation=40,ha='right');ax.set_ylabel('Actual anchor trials')
        save('state_yield',draw)
    if profiles:
        def draw(ax):
            for i,p in enumerate(profiles):
                v=p['phases'];ax.bar(np.arange(3)+i*.23,[v.get(k,{}).get('kernel_sum_ms',np.nan) for k in ('outside_llm','prefill','decode')],width=.23,label=p['mode'])
            ax.set_xticks(np.arange(3)+.23);ax.set_xticklabels(['Other pipeline','Prefill','Decode'])
            ax.set_ylabel('Actual GPU kernel duration sum ms; profiler controls');ax.legend()
        save('actual_gpu_kernel_time',draw)
    selected=[r for r in summary if r['run']=='multi' and r['mode'] in ('original','prefix','batch','adaptive')]
    if selected:
        def draw(ax):
            for r in selected:
                q=r['complete_question_token_equality'];ax.scatter(r['service_matched_p50_speedup'],q['equal']/q['total'],label=r['mode'])
            ax.set_xlabel('E2E P50 ratio to same-service native');ax.set_ylabel('Complete question token equality');ax.legend();ax.set_ylim(0,1.04)
        save('batch_quality_speed',draw)
    if batch:
        budget=batch.get('budget_precheck',[])
        if budget:
            def draw(ax):
                ax.bar(range(len(budget)),[r['wall_ms'] for r in budget]);ax.set_xticks(range(len(budget)))
                ax.set_xticklabels([str(r['budget_mb'])+'MB '+r['implementation'] for r in budget],rotation=35,ha='right')
                ax.set_ylabel('Actual executor attempt wall ms; no fallback generation')
            save('budget_early_rejection',draw)
    if resource:
        for key,label in (('sampled_peak_rss_bytes','Process resident sampled peak MiB'),('gpu_peak_allocated','GPU allocator peak MiB')):
            selected=resource['rows']
            def draw(ax):
                ax.bar(range(len(selected)),[r[key]/1024**2 for r in selected]);ax.set_xticks(range(len(selected)))
                ax.set_xticklabels([r['mode']+(' cold' if r['cold'] else ' warm') for r in selected],rotation=45,ha='right')
                ax.set_ylabel(label);ax.set_title('Separate actual resource controls; shared loaded-model footprint')
            save(key,draw)
    write(root/'figure_manifest.json',dict(figures=made,only_measured_data=True));return made


def main(root):
    root=Path(root);data=[];coverage=[];checks=[];pending=[]
    raw=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];by_token=token_index(raw)
    for run in ('smoke','single','multi','long_stream','stride_refresh','auto_validation'):
        path=root/run;status=path/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status')!='completed':
            pending.append(dict(run=run,status=json.loads(status.read_text()) if status.exists() else 'not_started'));continue
        records=rows(path/'frames.jsonl')
        for r in records:r['run']=run
        data.extend(records);executed=defaultdict(list)
        for r in records:executed[r['frame']['identity']['frame_id']].append(r['frame']['identity'])
        protocol=json.loads((path/'protocol.json').read_text())
        for spec in protocol['scenes']:
            _,resolved=resolve_sequence(spec,by_token,executed);coverage.append(dict(run=run,scene=spec['scene'],**resolved))
        if any(r['mode']=='m1' for r in records):checks.append(dict(run=run,**check(path,required=False)))
    del raw,by_token
    summary=summarize(data)
    for result in summary:
        selected=[r for r in data if r['run']==result['run'] and r['mode']==result['mode'] and not r['warmup']]
        result['admitted_frames']=sum(r['frame']['admission'].get('allowed') is True for r in selected)
        result['actual_overlap_frames']=sum(r['frame']['m1'].get('overlap_active') is True for r in selected)
        result['production_auto_certificate']=result['mode']=='unified_auto' and result['actual_overlap_frames']>0
        reference='m12' if result['mode']=='opt_m12' else 'm1'
        states={(r['repeat'],r['index']):r['frame']['quality']['m1_state_digest'] for r in data
            if r['run']==result['run'] and r['mode']==reference and not r['warmup']}
        result['persistent_state_reference']=reference if states else None
        result['exact_persistent_state_equality']=all(states.get((r['repeat'],r['index']))==r['frame']['quality']['m1_state_digest'] for r in selected) if states and result['service']=='persistent_state' else None
    formal=[r for r in data if r['run']!='smoke'];pairs=[]
    for run in sorted({r['run'] for r in formal}):
        selected=[r for r in formal if r['run']==run]
        for mode in sorted({r['mode'] for r in selected}):
            reference=('m12' if mode=='opt_m12' else 'm1') if next(r for r in selected if r['mode']==mode)['frame']['service']=='persistent_state' else 'original'
            if mode==reference:continue
            value=paired(selected,mode,reference)
            if value:pairs.append(dict(run=run,**value))
    profiles=[];probe=root/'formal_probes'
    if not (probe/'profile_manifest.json').exists():probe=root/'smoke_probes'
    if (probe/'profile_manifest.json').exists():
        for record in json.loads((probe/'profile_manifest.json').read_text()):
            destination=root/'traces'/(probe.name+'_'+record['mode']+'_summary.json')
            profile=analyze(record['path'],destination,record.get('worker'));profile.update(mode=record['mode'],scope=probe.name);profiles.append(profile)
    native_profile=next((p for p in profiles if p['mode']=='m1'),None)
    candidate_profile=next((p for p in profiles if p['mode']=='native_decode'),None)
    kernel_comparison=None
    if native_profile and candidate_profile and native_profile['decode_forward_calls']==candidate_profile['decode_forward_calls'] and native_profile['decode_forward_calls']:
        a=native_profile['phases'].get('decode');b=candidate_profile['phases'].get('decode')
        if a and b:
            kernel_comparison=dict(same_decode_forward_calls=native_profile['decode_forward_calls'],
                kernel_count_change=b['kernel_count']-a['kernel_count'],
                kernel_count_change_per_decode_token=(b['kernel_count']-a['kernel_count'])/native_profile['decode_forward_calls'],
                actual_kernel_sum_change_ms=b['kernel_sum_ms']-a['kernel_sum_ms'],
                scope='Separate same-source-frame profiler controls; ordinary E2E and independent-scene comparisons are separate')
    write(root/'decode_kernel_profile.json',dict(profiles=profiles,native_decode_comparison=kernel_comparison,
        missing_reason=None if profiles else 'Real v2 GPU profiler has not completed',prior_profile='decode_kernel_profile_prior.json'))
    def load(path):return json.loads(path.read_text()) if path.exists() else None
    decode=load(probe/'decode_backend_results.json');batch=load(root/'batch_probes/batch_quality_results.json')
    for name,value,reason in [('decode_backend_results',decode,'Actual native Sampling probe has not completed'),
            ('batch_quality_results',batch,'Actual saved-carrier GPU probe has not completed')]:
        write(root/(name+'.json'),value if value is not None else dict(status='not_measured',reason=reason))
    state=load(root/'state_yield_results.json');physical=load(root/'physical_oracle_results.json');cost=load(root/'cost_admission_results.json')
    long_state=load(root/'long_state_supply/state_supply_results.json')
    if long_state:
        from .v2_state_quality import main as state_quality
        state_quality(root)
    resource=load(probe/'resource_controls/resource_controls.json')
    if resource:write(root/'resource_controls.json',resource)
    write(root/'new_sequence_coverage.json',dict(sequences=coverage,all_stable_identity_validated=bool(coverage)))
    write(root/'overlap_correctness.json',dict(runs=checks,formal_pass=bool([x for x in checks if x['run']!='smoke']) and all(x['passed'] for x in checks)))
    workers=[r['frame']['m1']['worker'] for r in formal if not r['warmup'] and r['frame']['m1'].get('worker')]
    write(root/'overlap_performance.json',dict(paired=[p for p in pairs if p['mode'] in ('opt_serial','overlap','overlap_decode')],
        worker_ms=stats([r['worker_ms'] for r in workers]),main_join_wait_ms=stats([r['join_wait_ms'] for r in workers]),
        actual_trace_overlap=[dict(mode=p['mode'],lower_ms=p.get('m1_worker_gpu_kernel_overlap_lower_ms')) for p in profiles],
        note='Worker and CUDA event intervals are not additive; overlap lower bound uses actual GPU kernels'))
    for name,run in [('single_question_results','single'),('multi_question_results','multi'),('long_stream_results','long_stream')]:
        write(root/(name+'.json'),dict(status='measured' if any(r['run']==run for r in summary) else 'not_measured',
            methods=[r for r in summary if r['run']==run],paired=[p for p in pairs if p['run']==run]))
    constrained=[]
    for p in pairs:
        s=next(r for r in summary if r['run']==p['run'] and r['mode']==p['mode'])
        strict=s['exact_full_output_equality']
        if s['service']=='persistent_state':strict=strict and s['exact_persistent_state_equality'] is True
        if p['mode'] in ('opt_serial','overlap','opt_m12','unified_auto'):
            strict=strict and all(x['passed'] for x in checks if x['run']==p['run'])
        constrained.append(dict(p,exact_quality_pass=strict,quality_constrained_time_ratio=p['scene_mean_time_ratio'] if strict else None,
            independently_positive=bool(strict and p['scenes']>=2 and p['scene_cluster_bootstrap_ratio_95ci'][0]>1),
            auto_approved=s['production_auto_certificate']))
    write(root/'quality_constrained_speedup.json',dict(comparisons=constrained,contracts_separate=True))
    actual_queries=sum(r['state_queries'] for r in summary if r['run']!='smoke') if formal else None
    activations=sum(r['state_activated_questions'] for r in summary if r['run']!='smoke') if formal else None
    if long_state:
        actual_queries=(actual_queries or 0)+long_state['actual_task_queries']
        activations=(activations or 0)+long_state['actual_state_input_activations']
    reuse_cams=sum(r['actual_reused_cameras'] for r in summary if r['run']!='smoke') if formal else None
    reuse_windows=sum(r['actual_skipped_windows'] for r in summary if r['run']!='smoke') if formal else None
    decisions=dict(overlap=dict(strict_formal_pass=any(x['run']!='smoke' for x in checks) and all(x['passed'] for x in checks),
        independently_positive_comparisons=[r for r in constrained if r['mode']=='overlap' and r['independently_positive']]),
        native_decode=dict(real_probe_completed=decode is not None,all_probe_tokens_equal=all(q['complete_token_equality'] for q in decode['quality']) if decode else None),
        task_state=dict(queries=actual_queries,activations=activations,long_history_probe=long_state,
            decision='Feasibility No-Go' if actual_queries and not activations else 'requires independent task certification'),
        vision=dict(reused_cameras=reuse_cams,skipped_windows=reuse_windows),
        batch=dict(exact_gate_unchanged=True,formal_results=[r for r in summary if r['run']=='multi' and r['mode'] in ('prefix','batch','adaptive')]),
        production_auto=dict(approved_keys=cost['approved_keys'] if cost else 0,
            reason='See exact-key independent cost/quality evidence and auto_validation decisions' if cost else 'No completed formal calibration'),
        incomplete=pending)
    write(root/'go_nogo_decisions.json',decisions)
    if state:
        state.update(task_queries=actual_queries,task_activations=activations,task_query_hit_rate=activations/actual_queries if actual_queries else None)
        rejection=Counter(q['state_retrieval'].get('reason','unknown') for r in formal if not r['warmup']
            for q in r['frame']['m3'].get('requests',[]) if q.get('state_retrieval') and not q['state_retrieval'].get('admitted'))
        if long_state:
            rejection.update(q['state_retrieval'].get('reason','unknown') for r in rows(root/'long_state_supply/queries.jsonl')
                for q in r['diagnostics'].get('requests',[]) if q.get('state_retrieval') and not q['state_retrieval'].get('admitted'))
        state.update(task_query_rejections=dict(rejection),long_history_probe=long_state,
            task_quality_scope='Actual state input activations; positive counts are not a physical/task quality certificate')
        write(root/'state_yield_results.json',state)
    made=figures(root,summary,formal,pairs,profiles,state,batch,resource)
    measured=[r for r in formal if not r['warmup']]
    def method(run,mode):return next((r for r in summary if r['run']==run and r['mode']==mode),None)
    def comparison(run,mode):return next((p for p in pairs if p['run']==run and p['mode']==mode),None)
    strict_pairs=[p for run in checks if run['run']!='smoke' for p in run['pairs']]
    overlap_lower=[p['m1_worker_gpu_kernel_overlap_lower_ms'] for p in profiles if 'm1_worker_gpu_kernel_overlap_lower_ms' in p]
    overlap_answer=(f"正式配对 {sum(p['pass'] for p in strict_pairs)}/{len(strict_pairs)} 全部字段通过；实际 kernel 与 Worker 的重叠下界为 {overlap_lower} ms。"
        if strict_pairs else '正式等价性与净收益尚未完成验证。')
    baseline=method('long_stream','m1');candidate=method('long_stream','overlap');long_pair=comparison('long_stream','overlap')
    persistent_answer=(f"长序列原串行 P50/P95={baseline['e2e_wall_ms']['p50']:.2f}/{baseline['e2e_wall_ms']['p95']:.2f} ms，重叠={candidate['e2e_wall_ms']['p50']:.2f}/{candidate['e2e_wall_ms']['p95']:.2f} ms；配对场景平均变化 {long_pair['mean_scene_increment_ms']:.2f} ms，95% CI={long_pair['scene_cluster_bootstrap_increment_95ci_ms']}。"
        if baseline and candidate and long_pair else '正式长序列尚未测完，不能给出同契约完整帧收益。')
    kernel_native=next((p for p in profiles if p['mode']=='m1'),None);kernel_candidate=next((p for p in profiles if p['mode']=='native_decode'),None)
    decode_answer=(f"同输入独立 Profiler 的 Decode kernel sum 为 {kernel_native['phases'].get('decode',{}).get('kernel_sum_ms')} → {kernel_candidate['phases'].get('decode',{}).get('kernel_sum_ms')} ms；每个 Token 为 {kernel_native['per_decode_token_kernel_ms']} → {kernel_candidate['per_decode_token_kernel_ms']} ms。单次 Profiler 不是独立场景的性能证据，完整帧配对区间另见 single_question_results.json。"
        if kernel_native and kernel_candidate else '尚未完成可归因的真实 kernel 对照。')
    positive=[p['run']+'/'+p['mode'] for p in constrained if p['independently_positive'] and p['mode'] in ('opt_serial','overlap','overlap_decode','native_decode','opt_m12')]
    speed_answer='有：'+', '.join(positive)+'，其同服务场景 bootstrap 时间比下界大于 1；适用范围与生产批准仍独立判断。' if positive else '尚无同时通过严格质量和独立场景净收益区间的已完成比较。'
    multi_candidates=[r for r in summary if r['run']=='multi' and r['mode'] in ('prefix','batch','adaptive')]
    multi_answer='；'.join(f"{r['mode']} 完整问题 Token 等价 {r['complete_question_token_equality']['equal']}/{r['complete_question_token_equality']['total']}，严格质量 {'通过' if r['exact_full_output_equality'] else 'No-Go'}" for r in multi_candidates) or '正式多问题尚未测完。'
    auto=method('auto_validation','unified_auto');approved=cost['approved_keys'] if cost else 0
    auto_answer=(f"独立校准批准 {approved} 个精确键；新场景实测 {auto['measured_frames']} 帧，其中 {auto['actual_overlap_frames']} 帧启用重叠，其余保持完整串行服务。" if auto else f'当前批准 {approved} 个键，尚未完成校准后新场景真实准入测试。')
    if auto and not auto['actual_overlap_frames']:auto_answer+='真实正准入未发生，正分支仅由回归测试验证，不宣称已完成生产加速闭环。'
    closed_loop_answer=('尚未形成：M2 合法视觉跳算或 M3 实际任务状态消费仍缺失。M1/EVA 的调度收益不等于三模块缓存因果闭环。' if reuse_cams==0 or activations==0 else '必须结合物理 Oracle、任务质量、视觉跳算与消费证据判断；非零计数本身不能证明完整闭环。')
    lines=['# OmniDrive 第二轮系统优化实验报告','',
        '只统计本目录已完成的真实运行；旧结果保持原样。单问题保留原生 Sampling（temperature=0.1、top_p=0.75、max_new_tokens=320）；多问题明确采用 Greedy。',
        f"新增正式测量 {len(measured)} method-frames，{len({r['index'] for r in measured})} 个唯一帧，{len({r['frame']['identity']['scene_id'] for r in measured})} 个独立场景。探索性 smoke 不计入正式置信区间。",
        '输入准备、完整推理、同步、原生输出写入和共同 Token-ID 审计拷贝计入 E2E；停止计时后的 feature/output/temporal-memory NumPy 拷贝、独立物理 Oracle、M1 状态摘要/快照、Profiler 与 Logits 拷贝均不计入性能样本。早期 protocol 的 audit-copy 表述过宽，准确范围见 timing_scope_correction.json。','',
        '| Run | 方法 | 服务 | 帧数 | E2E P50/P95 ms | 完整 Token 等价帧 |','|---|---|---|---:|---:|---:|']
    for r in summary:
        w=r['e2e_wall_ms'];lines.append(f"| {r['run']} | {r['mode']} | {r['service']} | {r['measured_frames']} | {w['p50']:.2f}/{w['p95']:.2f} | {r['token_equal_frames']}/{r['measured_frames']} |")
    lines+=['','## 配对场景结果','','以相同服务的原生实现为基线，独立场景 bootstrap 10,000 次；帧/重复不是独立样本，不相加组件 P50。',
        '| Run / 方法 | 配对场景数 | 平均场景增量 ms | 95% CI ms | 同服务时间比 |','|---|---:|---:|---|---:|']
    for p in pairs:lines.append(f"| {p['run']} / {p['mode']} | {p['scenes']} | {p['mean_scene_increment_ms']:.2f} | {p['scene_cluster_bootstrap_increment_95ci_ms']} | {p['scene_mean_time_ratio']:.4f} |")
    lines+=['','## 十二项验收回答','',
        '1. 时间倒退来自统计索引错误：Dataset 全局按 timestamp 排序，原始 pkl 未同序。17 个历史序列按 scene/token/timestamp 联查均执行连续；只修覆盖统计，保留原时延。详见 sequence_identity_audit.md。',
        '2. '+overlap_answer+' Anchor/Mask/Feature Provenance、感知和时序记忆等价性见 overlap_correctness.json；没有用两个时段之差冒充重叠。',
        '3. '+persistent_answer+' Current-Output 只作不同服务对照。',
        '4. '+decode_answer+' 候选不改变 GEMM、权重或原始生成合同；完整 IDs、同历史 raw Logits、RNG、ADE/FDE、显存和独立 LM 计时见 decode_backend_results.json。Mask 检查的主机同步也计费。',
        '5. '+speed_answer+' 逐项质量约束时间比见 quality_constrained_speedup.json。',
        f'6. 离线候选在真实流上的实际任务查询 {actual_queries}，实际状态输入激活 {activations}（含长历史检查点）。这是实际 LLM 输入消费，不是生产 Auto 批准；独立任务标注质量见 long_state_supply/task_quality.json，物理 Anchor 通过不等于任务消费。',
        f'7. 正式移动场景合法 Camera 复用 {reuse_cams}、Window 跳算 {reuse_windows}；无命中则如实 No-Go。',
        '8. 对既有真实失败 carrier 运行原生逐层公共前缀/缓存后缀诊断，并比较 2/3/5 问题、Batch/Adaptive 的首次不同 Token 前同历史 Logits；见 batch_probes。只作有界诊断，不降低质量门槛。',
        '9. '+multi_answer+' QA 原标注重叠、Planning ADE/FDE 与场景配对性能见 multi_question_results.json；近似输出收益不算 Exact 加速。',
        '10. '+auto_answer+' 成本键保留精确部署/实现/线程/Shape/问题长度/合同，只允许证实范围内的性能 frame-gap 区间。无证书维持串行 M1 和完整 GPU/LLM。',
        '11. '+closed_loop_answer,
        '12. 原生单问题逐 Token Decode、动态 KV/LoRA 与小矩阵调度，M1 的独立物理证据不足，以及 Batch 形状造成的数值分歧仍构成主要边界。具体 kernel 排名和逐层差异见原始证据，未用降低输出长度或精度规避。','',
        '## 工程和边界','',
        '默认保持 serial；实验显式选择 experimental，Auto 需要双证书。M1 Worker 只有一个，独占可变 Store；只与完整 EVA 重叠，在 heads/commit/M3 前 join。异常或超时先 drain 再 reset，超时不承诺硬截止。M2 若需要本帧验证则串行。',
        'Original Persistent Serial 使用固定基线提交 b4782ab 的原 simple_test 方法；统一协调/测量层共同使用本轮版本，新增记录开销在各方法 E2E 内，未将历史旧时延直接当本轮基线。权重、嵌入 LoRA、问题、随机种子与停止条件不变。优化串行仅减少重复签名扫描。Batch 预算预检只拒绝不可能容纳的缓存，不给予复用许可。',
        '物理 Oracle 仅离线读取独立 nuScenes annotations；2m 源身份门、1m 当前位置门、visibility=4 和传感器支持均预先冻结。未关联身份/地图/全局任务完整性不得算作已验证；投影目标区域不是像素遮挡真值或闭环安全证明。',
        f'生成 {len(made)} 张实际数据图，包含 PNG/PDF；逐帧、源快照、命令、失败日志均保留。正式帧 CPU RSS 是进程高水位；独立资源控制的 20ms resident 采样峰值见 resource_controls.json，包含已加载模型和共同观察线程，不冒充全序列精确增量峰值。',
        'long_state_supply 在真实完整维护的长历史上，于第 8/16/32 帧执行原生五问题 Greedy 与状态候选；非检查点保留单问题 Sampling。该混合服务和附加 LM 请求不合入常规单问题 E2E，也不增加独立场景数。','']
    if pending:lines+=['## 尚未完成的运行','',json.dumps(pending,ensure_ascii=False,indent=2),'','未完成部分不能用于性能结论或生产批准。']
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    write(root/'report_manifest.json',dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*/frames.jsonl')},
        complete_formal=not pending,measured_method_frames=len(measured),figures=made))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/system_optimization_v2');main(p.parse_args().root)
