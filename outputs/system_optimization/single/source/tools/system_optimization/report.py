"""Report only real measured rows; historical evidence stays explicitly labelled."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import write, stats


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if not row['warmup']:
            groups[(row['run'],row['mode'],row['frame']['service'])].append(row)
    output=[]
    for (run,mode,service),frames in sorted(groups.items()):
        identities=[r['frame']['identity'] for r in frames]
        numerical=[r['frame']['quality']['numerical'] for r in frames]
        exact=all(r['frame']['quality']['exact_token_equality'] and r['frame']['quality']['token_count_matches']
            and all(not d['shape_mismatches'] and all(v['max_abs']==0 for v in d['leaves']) for d in n.values())
            for r,n in zip(frames,numerical))
        components=defaultdict(list)
        for row in frames:
            by_stage=defaultdict(float)
            for stage in row['frame']['stages']:
                if 'cuda_interval_ms' in stage:
                    by_stage[stage['name']]+=stage['cuda_interval_ms']
            for name,value in by_stage.items():components[name].append(value)
        planning=[task['planning'] for r in frames for task in r['tasks'] if task['planning'] is not None]
        qs=sum(r['frame']['question_count'] for r in frames)
        requests=[q for r in frames for q in r['frame']['m3'].get('requests',[])]
        generated=sum(max(0,len(ids[0])-1) for r in frames for ids in r['generated_ids'])
        m1=[r['frame']['m1'].get('m1_total_including_adapters_ms',0.) for r in frames]
        accepted=sum(r['frame']['m1'].get('valid_anchor_count',0) for r in frames)
        decoded=sum(components.get('llm_decode',[]))
        output.append(dict(run=run,mode=mode,service=service,workload=frames[0]['workload'],
            measured_frames=len(frames),unique_frames=len({i['frame_id'] for i in identities}),
            scenes=len({i['scene_id'] for i in identities}),questions=qs,
            e2e_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames]),
            preprocessing_wall_ms=stats([r['frame']['input_preparation_wall_ms'] for r in frames]),
            components_cuda_interval_ms={k:stats(v) for k,v in components.items()},
            exact_full_output_equality=exact,
            token_equal_frames=sum(r['frame']['quality']['exact_token_equality'] for r in frames),
            generated_tokens=generated,effective_decode_tokens_per_second=generated/(decoded/1000) if decoded else None,
            m1_maintenance_ms=stats(m1),verified_anchor_records=accepted,
            useful_state_yield_per_ms=None,  # Task usefulness needs independent query evidence.
            actual_skipped_windows=sum(r['frame']['m2'].get('actual_skipped_windows',0) for r in frames),
            actual_reused_cameras=sum(r['frame']['m2'].get('actual_reused_cameras',0) for r in frames),
            actual_reused_kv_tokens=sum(q.get('reused_tokens',0) for q in requests),
            state_activated_questions=sum(bool(q.get('state_retrieval',{}).get('admitted')) for q in requests),
            state_queries=sum(q.get('state_retrieval') is not None for q in requests),
            fallback_frames=sum(bool(r['frame']['fallback_reasons']) for r in frames),
            fallback_reasons=dict(__import__('collections').Counter(reason for r in frames for reason in r['frame']['fallback_reasons'])),
            memory=dict(cache_peak_bytes=max(r['frame']['memory']['tracked_cache_bytes'] for r in frames),
                gpu_peak_allocated=max(r['frame']['memory'].get('gpu_peak_allocated',0) for r in frames),
                gpu_peak_reserved=max(r['frame']['memory'].get('gpu_peak_reserved',0) for r in frames),
                cpu_rss_highwater_bytes=max(r['frame']['memory'].get('cpu_rss_highwater_bytes',0) for r in frames)),
            cold_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames if r['cold']]),
            warm_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames if not r['cold']]),
            planning=dict(parsed=sum(p.get('status')=='parsed' for p in planning),questions=len(planning),
                ade_m=stats([p['ade_m'] for p in planning if p.get('ade_m') is not None]),
                fde_m=stats([p['fde_m'] for p in planning if p.get('fde_m') is not None]))))
    for row in output:
        base=next((x for x in output if x['run']==row['run'] and x['mode']=='original'),None)
        matched=next((x for x in output if x['run']==row['run'] and x['mode']=='m1' and x['service']==row['service']),None) if row['service']=='persistent_state' else base
        row['p50_ratio_to_original']=base['e2e_wall_ms']['p50']/row['e2e_wall_ms']['p50'] if base else None
        row['service_matched_reference']=matched['mode'] if matched else None
        row['service_matched_p50_speedup']=matched['e2e_wall_ms']['p50']/row['e2e_wall_ms']['p50'] if matched else None
        row['quality_constrained_service_speedup']=row['service_matched_p50_speedup'] if row['exact_full_output_equality'] else None
        row['production_auto_certificate']=False
    return output


def trace_summary(path):
    events=json.loads(path.read_text())['traceEvents']
    def union(items):
        total=0.;end=-float('inf')
        for start,stop in sorted(items):
            if stop>end:total+=stop-max(start,end);end=stop
        return total/1000
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    cuda=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('ph')=='X']
    copies=[e for e in events if e.get('cat')=='gpu_memcpy' and e.get('ph')=='X']
    return dict(source=str(path),evidence='separate_real_full_gpu_trace_not_latency_sample',
        kernel_count=len(kernels),kernel_duration_sum_ms=sum(e['dur'] for e in kernels)/1000,
        kernel_active_union_ms=union([(e['ts'],e['ts']+e['dur']) for e in kernels]),
        gpu_copy_ms={kind:sum(e['dur'] for e in copies if kind.lower() in e['name'].lower())/1000 for kind in ('HtoD','DtoH','DtoD')},
        synchronize_cpu_ms=sum(e['dur'] for e in cuda if 'synchroniz' in e['name'].lower())/1000,
        allocation_api_ms=sum(e['dur'] for e in cuda if 'malloc' in e['name'].lower() or 'free' in e['name'].lower())/1000,
        kernel_names=dict(__import__('collections').Counter(e['name'] for e in kernels).most_common(15)),
        launch_gap_ms=None,note='CUDA intervals include idle gaps; no inference from FLOPs. Gap attribution remains unmeasured.')


def figures(root, summary, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    made={}
    def save(number,title,draw):
        fig,ax=plt.subplots(figsize=(10,4.5));draw(ax)
        ax.set_title(title);fig.tight_layout()
        fig.savefig(root/'figures'/f'{number}.png',dpi=180)
        fig.savefig(root/'figures'/f'{number}.pdf');plt.close(fig)
        made[number]=dict(title=title,status='measured')
    base=[r for r in summary if r['mode']=='original' and r['run'] in ('single','multi')]
    if base:
        def breakdown(ax):
            stages=['eva','detection_head','map_head','llm_prefill','llm_decode']
            for i,row in enumerate(base):
                xs=[row['components_cuda_interval_ms'].get(k,{}).get('p50',0) for k in stages]
                ax.bar(np.arange(len(stages))+i*.35,xs,width=.35,label=row['workload'])
            ax.set_xticks(np.arange(len(stages))+.17);ax.set_xticklabels(stages,rotation=15)
            ax.set_ylabel('CUDA interval P50 ms (not additive)');ax.legend()
        save('S1','Complete VLM measured components',breakdown)
    state=[r for r in rows if not r['warmup'] and r['frame']['m1'].get('m1_total_including_adapters_ms') is not None]
    if state:
        save('S2','Verified physical anchors vs maintenance cost (not task quality)',lambda ax:(
            ax.scatter([r['frame']['m1']['m1_total_including_adapters_ms'] for r in state],
                       [r['frame']['m1'].get('valid_anchor_count',0) for r in state],s=10),
            ax.set_xlabel('M1 wall ms'),ax.set_ylabel('Verified physical anchors')))
    if summary:
        selected=[r for r in summary if r['run'] in ('single','multi','language_candidates')]
        def bars(ax,key,ylabel):
            ax.bar(range(len(selected)),[r[key] for r in selected]);ax.set_xticks(range(len(selected)))
            ax.set_xticklabels([r['run']+':'+r['mode'] for r in selected],rotation=75,fontsize=7);ax.set_ylabel(ylabel)
        save('S3','Actual visual reuse on real moving inputs',lambda ax:bars(ax,'actual_reused_cameras','Reused cameras (zero is measured)'))
        save('S4','Actual M1 semantic-state activation in M3',lambda ax:bars(ax,'state_activated_questions','Activated task queries'))
        candidates=[r for r in summary if r['run']=='language_candidates']
        if candidates:
            save('S5','Offline language quality vs speed (no Auto certificate)',lambda ax:(
                ax.scatter([r['p50_ratio_to_original'] or 1 for r in candidates],
                           [r['token_equal_frames']/r['measured_frames'] for r in candidates]),
                [ax.annotate(r['mode'],(r['p50_ratio_to_original'] or 1,r['token_equal_frames']/r['measured_frames'])) for r in candidates],
                ax.set_xlabel('P50 ratio to native'),ax.set_ylabel('Frames with complete token equality')))
        save('S6','Single/multi service wall measured separately',lambda ax:(
            ax.bar(range(len(base)),[r['e2e_wall_ms']['p50'] for r in base]),
            ax.set_xticks(range(len(base))),ax.set_xticklabels([r['workload'] for r in base]),ax.set_ylabel('Full service P50 ms')))
        cache=[r for r in summary if r['run'].startswith('budget_')]
        if cache:
            save('S7','Cache memory vs real service performance',lambda ax:(
                ax.scatter([r['memory']['cache_peak_bytes']/1024**2 for r in cache],
                    [r['e2e_wall_ms']['p50'] for r in cache]),ax.set_xlabel('Tracked cache peak MiB'),ax.set_ylabel('Service P50 ms')))
        long=[r for r in rows if r['run']=='long_stream' and not r['warmup']]
        if long:
            def draw_long(ax):
                for mode in sorted({r['mode'] for r in long}):
                    values=[r for r in long if r['mode']==mode]
                    ax.plot([r['frame']['e2e_wall_ms'] for r in values],label=mode)
                ax.set_xlabel('Unique frame order (scene boundary retained)');ax.set_ylabel('E2E ms');ax.legend(fontsize=7)
            save('S8','Long-stream complete VLM latency',draw_long)
        ablation=[r for r in summary if r['run']=='single']
        if ablation:
            save('S9','Cross-module ablation; service semantics are labelled',lambda ax:(
                ax.bar(range(len(ablation)),[r['e2e_wall_ms']['p50'] for r in ablation]),
                ax.set_xticks(range(len(ablation))),ax.set_xticklabels([r['mode']+'\n'+r['service'] for r in ablation],rotation=60,fontsize=7),ax.set_ylabel('E2E P50 ms')))
        admitted=[r for r in summary if r['quality_constrained_service_speedup'] is not None and r['run'] in ('single','multi')]
        if admitted:
            save('S10','Observed output-equal service ratios (not deployment approval)',lambda ax:(
                ax.bar(range(len(admitted)),[r['quality_constrained_service_speedup'] for r in admitted]),
                ax.set_xticks(range(len(admitted))),ax.set_xticklabels([r['run']+':'+r['mode'] for r in admitted],rotation=75,fontsize=7),ax.axhline(1,color='black',linewidth=.7)))
    for i in range(1,11):made.setdefault('S'+str(i),dict(status='unmeasured_no_fabricated_figure'))
    write(root/'figures/manifest.json',made)


def main(root):
    root=Path(root);rows=[];runs={};sources={}
    for path in sorted(root.glob('*/frames.jsonl')):
        run=path.parent.name
        # Smoke runs check the harness and are never merged with formal tables.
        if run.startswith('smoke'):continue
        status=json.loads((path.parent/'status.json').read_text())
        runs[run]=status
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            row=json.loads(line);row['run']=run;rows.append(row)
    summary=summarize(rows)
    write(root/'scheduler_results.json',dict(results=summary,run_status=runs,source_sha256=sources))
    for name,keep in [('single_question_results',lambda r:r['run']=='single'),
            ('multi_question_results',lambda r:r['run'] in ('multi','language_candidates')),
            ('long_stream_results',lambda r:r['run']=='long_stream'),
            ('ablation_results',lambda r:r['run'] in ('single','multi')),
            ('memory_results',lambda r:True)]:
        write(root/(name+'.json'),dict(evidence='new_real_weight_gpu',results=[r for r in summary if keep(r)]))
    write(root/'exact_correctness.json',dict(results=[{k:r[k] for k in ('run','mode','measured_frames','token_equal_frames','exact_full_output_equality')} for r in summary],
        caveat='No reuse when skipped computation is zero; full fallback equality is not reuse certification'))
    write(root/'task_quality.json',dict(results=[dict(run=r['run'],mode=r['mode'],planning=r['planning']) for r in summary],
        qa_rows=[dict(run=r['run'],mode=r['mode'],index=r['index'],tasks=r['tasks']) for r in rows if not r['warmup']],
        physical_false_acceptance='not newly measured; historical heldout evidence in state_consumption.json and workload_audit.md',
        critical_change_recall='not newly measured',official_detection_map_metrics=None,closed_loop_safety=None))
    with (root/'cross_module_trace.jsonl').open('w') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    profiles=[trace_summary(p) for p in root.glob('*/pipeline_trace.json')]
    latency=json.loads((root/'latency_breakdown.json').read_text())
    latency['new_gpu_profiles']=profiles;latency['new_frame_components']=summary
    write(root/'latency_breakdown.json',latency)
    historical=json.loads((root/'state_consumption.json').read_text())
    historical['new_runs']=[dict(run=r['run'],mode=r['mode'],verified_anchor_records=r['verified_anchor_records'],
        task_queries=r['state_queries'],activated_questions=r['state_activated_questions']) for r in summary]
    write(root/'state_consumption.json',historical)
    decisions=dict(A='Go: prioritize measured Decode and remove rejection-only copy cost',
        B='No-Go: insufficient task-consumable verified history; physical validity never proves feature equality',
        C='No-Go for real-scene reuse when actual skipped operations remain zero',
        D='No production Prefix/Batch certificate; native Full fallback retained',
        E='Lifecycle and service coordination implemented; see CPU/GPU regression and native comparison',
        F='Judge measured full-service net cost per service; no deployment approval from aggregate speedup')
    write(root/'go_nogo_decisions.json',decisions)
    write(root/'quality_admission.json',dict(approved=[],reasons=decisions,
        rule='No untested shapes or global average speedups can grant approval; observations are not certificates'))
    figures(root,summary,rows)
    report=['# OmniDrive 跨模块系统优化实验报告','',
        '审计与新增实验分开归档。原始模型、LoRA、Transformers 和质量门限保持不变。'
        '本报告不把完整回退的正确性称为状态复用成功。','',
        '## 新增真实 GPU 实验','',
        '| Run | 方法 | 服务 | 唯一帧/计时帧 | E2E P50/P95 ms | 完整输出等价 | 同服务参考 | 同服务 P50 比值 |',
        '|---|---|---|---:|---:|---|---|---:|']
    for r in summary:
        speed=r['service_matched_p50_speedup']
        report.append(f"| {r['run']} | {r['mode']} | {r['service']} | {r['unique_frames']}/{r['measured_frames']} | "
            f"{r['e2e_wall_ms']['p50']:.2f}/{r['e2e_wall_ms']['p95']:.2f} | {r['exact_full_output_equality']} | "
            f"{r['service_matched_reference']} | {speed:.3f} |" if speed is not None else
            f"| {r['run']} | {r['mode']} | {r['service']} | {r['unique_frames']}/{r['measured_frames']} | "
            f"{r['e2e_wall_ms']['p50']:.2f}/{r['e2e_wall_ms']['p95']:.2f} | {r['exact_full_output_equality']} | 未测同服务基线 | — |")
    report+=['','单问题使用原始 Sampling，多问题使用受控 Greedy；两者不合并。'
        '所有问题均来自原始数据管线并在实验前固定，320 Token 上限与正常 EOS 保留。'
        '完整服务 wall 包含本次输入读取/预处理、collate/scatter、同步、缓存和实际推理；'
        'GT 与输出差异计算在计时外。原生输出解析和所有方法同样的 Token 记录在计时内。'
        '组件 CUDA interval 包含 launch gap，不能当作 kernel time；嵌套阶段不相加，组件 P50 不相加。','',
        '## 十二项结论','',
        '1. 主要瓶颈是原生 LLM Decode；阶段 A 的历史逐帧依据见 workload_audit.md，本轮 GPU trace 的独立 kernel 统计见 latency_breakdown.json。',
        '2. 当前可确认的工程改进是拒绝路径的采样快照。任何 Prefix/Batch 离线速度必须同时看本表输出等价，不能自动归因于世界状态。',
        '3. 有效物理 Anchor、通过验证的观测区域、视觉计算状态、任务语义状态分别记录。历史延长序列任务消费为零，来源帧元数据补齐不创造新的证据。',
        '4. M1 成本只有实际跳算/任务收益才可能支付；前两层 Window 的历史理想节省约 43.6ms，尚未包含认证/缓存，不能支付既有完整 M1 维护。',
        '5. M2 是否发生真实跳算以 actual_skipped_windows/actual_reused_cameras 为准；为零时不声称视觉复用加速。',
        '6. M3 状态消费以 state_activated_questions 为准；零激活或全回退维持 No-Go。',
        '7. Shared Prefix/Batch 的 FP16 shape 敏感性尚无完整原生输出等价修复。Adaptive 分组是独立离线候选，Sampling 在重排前回退。',
        '8. 单问题没有同帧 Prefix 共享收益；完整服务和通用 inference-mode 控制实验分开报告，后者不属于世界状态收益。',
        '9. 多问题质量约束下的观察比值只在完整输出等价时给出；小样本通过不自动发布生产证书。',
        '10. Persistent-State 与 Current-Output 使用不同同服务参考。跳过 M1 只减少服务责任，不能当作同服务算法加速。',
        '11. 保留合法来源、早期拒绝和跨模块失效假设；不支持以物理关联代替 Feature Equality、以默认完整回退证明状态消费的假设。',
        '12. 自回归 Decode 的逐 Token 依赖、全局视觉注意力、未建立的对象到 Carrier 映射、原生 FP16 shape 敏感性，仍是当前推理修改无法保证解决的结构性限制。','',
        '## 完整性、限制和复现','',
        f'本轮执行状态：`{json.dumps(runs,ensure_ascii=False)}`。',
        '正式结果只读取独立运行目录 frames.jsonl；smoke 和失败记录不混入。源代码快照、原始 Token IDs、原生参考张量、逐任务 GT 和完整 trace 均保留。',
        '物理 FAR/关键变化召回的新独立 Oracle、官方检测/地图指标和闭环安全未测，不能由 Token 等价推断。重复计时不增加独立帧或独立场景数量。',
        '详见 commands.sh、tools/system_optimization/run.sh 与 docs/worldstream_system_optimization.md。',
        '插桩清单：worldstream/execution.py（嵌套时间与来源）；worldstream/runtime.py（统一入口与请求级包装）；'
        'petr3d.py（可选统一入口）；tools/system_optimization/experiment.py（真实模型、数据、输出与 Profiler）。'
        '不配置 stream_config 时保持原入口。']
    (root/'experiment_report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/system_optimization')
    main(parser.parse_args().root)
