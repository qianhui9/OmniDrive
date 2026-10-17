"""Derive refinement delivery exclusively from complete recorded GPU protocols."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from .experiment import write,stats
from .llm_probe import quality as qa_diagnostics
from worldstream.m3.task_conditioner import task_type
from .offline_reference import resolve_reference

LABELS=dict(baseline='Baseline',legacy_exact='Old Exact',refined_exact='Refined Exact',compact='Compact',
    state_only='State Only',state_full='State + Full',batched='Batched',refined_combined='Refined Batch',m123='M1+M2+Batch',m12='M1+M2',
    common_prefix_seq='Common Prefix Seq',precision_deterministic='Deterministic',precision_no_fp16_reduction='No FP16 Reduction',precision_fp32_lora='FP32 LoRA')


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_recordings(row):
    """Early main-run instrumentation logged native fallback returns twice.
    The final block is the complete, ordered runtime return. Verify duplicates
    against precisely the native-fallback request indices, never drop questions.
    Raw frames.jsonl stays unchanged; actual model calls were never duplicated.
    """
    count=len(row['questions']);recorded=row['generated_ids'];excess=len(recorded)-count
    if not excess:return 0
    assert 0<excess<=count
    ordered=recorded[-count:];requests=row['diagnostics'].get('requests',[])
    fallback=[ordered[i] for i,d in enumerate(requests) if d.get('mode')=='full'] if requests else ordered
    assert recorded[:excess]==fallback
    assert row['components']['prefill']['calls']==count
    row['generated_ids']=ordered
    row['recording_normalization']=dict(raw_count=len(recorded),question_count=count,
        reason='Duplicate native fallback recording; one real prefill per question; ordered runtime return retained')
    return 1


def audit(run):
    protocol=read(run/'protocol.json');completion=read(run/'completion.json');rows=lines(run/'frames.jsonl')
    expected={(r,s['scene'],i,m) for r in range(protocol['warmup']+protocol['repeats'])
        for s in protocol['scenes'] for i in s['indices'][:2] for m in protocol['modes']}
    actual=[(r['repeat'],r['scene'],r['index'],r['mode']) for r in rows]
    assert completion['complete'] and completion['weights_versions_unchanged']
    assert len(rows)==completion['expected_rows']==len(expected) and len(set(actual))==len(actual) and set(actual)==expected
    normalized=0
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('ckpts/pretrain_qformer/',use_fast=False,local_files_only=True)
    tokenizer.pad_token=tokenizer.unk_token
    for row in rows:
        normalized+=normalize_recordings(row)
        assert len(row['questions'])==len(row['generated_ids'])==len(row['answers'])==protocol['questions']
        assert all(1<len(x[0])<=321 for x in row['generated_ids'])
        si=next(i for i,s in enumerate(protocol['scenes']) if s['scene']==row['scene'])
        fi=protocol['scenes'][si]['indices'].index(row['index'])
        path=run/f'answers/r{row["repeat"]}_{si}_{row["mode"]}_{fi}'/row['token']
        assert read(path)==row['answers']
        assert all(tokenizer.batch_decode(ids,skip_special_tokens=True)==answer['A']
            for ids,answer in zip(row['generated_ids'],row['answers']))
    source={p:digest(p)==sha for p,sha in read(run/'source_manifest.json').items()
        if p.startswith('worldstream/m3/') or p.endswith('detectors/petr3d.py')}
    assert source and all(source.values()),'Inference source changed after the recorded GPU run'
    lengths={}
    for path in (run/'carriers').glob('*.pt'):
        snap=torch.load(path,map_location='cpu');visual=snap['carriers'].shape[1]
        lengths[path.stem]=[int(snap['ids'][i].numel()+int((snap['ids'][i]==-200).sum())*(visual-1)) for i in snap['selected_indices']]
    for row in rows:
        row['_original_input_lengths']=lengths[row['token']]
    return rows,dict(source=str(run),rows=len(rows),measured=sum(not r['warmup'] for r in rows),
        expected_unique_cells_verified=True,raw_answers_and_question_counts_verified=True,
        normalized_duplicate_fallback_records=normalized,tokenizer_decode_matches_raw_answers=True,
        unchanged_parameter_versions=True,source_sha256_verified=source,frames_sha256=digest(run/'frames.jsonl'))


def nvml(run):
    path=run.parent/'logs'/f'{run.name}_gpu.csv';records=[]
    for line in path.read_text().splitlines():
        parts=line.split(',')
        try:records.append((datetime.strptime(parts[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp(),float(parts[1]),float(parts[2]),float(parts[3])))
        except (ValueError,IndexError):continue
    return records


def summarize(rows,gpu_records):
    baseline={(r['repeat'],r['token']):r for r in rows if r['mode']=='baseline'};summary={};failures=[]
    for mode in dict.fromkeys(r['mode'] for r in rows):
        selected=[r for r in rows if r['mode']==mode];equal=questions=0;tasks=defaultdict(lambda:dict(questions=0,equal=0,f1=[],numeric_errors=[],numeric_count_mismatch=0,annotated=0))
        for r in selected:
            b=baseline[(r['repeat'],r['token'])]
            for i,q in enumerate(r['questions']):
                same=r['generated_ids'][i]==b['generated_ids'][i];equal+=same;questions+=1
                task=tasks[task_type(q)];task['questions']+=1;task['equal']+=same
                reference,reference_source=resolve_reference(r['token'],q,r['ground_truth'])
                if reference is not None:
                    text=' '.join(str(x) for x in r['answers'][i]['A']);v=qa_diagnostics(text,reference)
                    task['annotated']+=1;task['f1'].append(v['token_f1']);task['numeric_count_mismatch']+=not v['numeric_count_match']
                    if v['numeric_mae'] is not None:task['numeric_errors'].append(v['numeric_mae'])
                if not same:failures.append(dict(mode=mode,scene=r['scene'],token=r['token'],repeat=r['repeat'],split=r['split'],
                    task=task_type(q),question=q,baseline=b['answers'][i],actual=r['answers'][i],original_gt=reference,
                    ground_truth_source=reference_source,
                    reference_ids=b['generated_ids'][i],actual_ids=r['generated_ids'][i]))
        for task in tasks.values():
            task['mean_gt_token_f1']=float(np.mean(task['f1'])) if task['f1'] else None
            task['aligned_numeric_list_mae']=float(np.mean(task['numeric_errors'])) if task['numeric_errors'] else None
        metric=dict(frames=len(selected),questions=questions,token_equal=equal,frozen_token_target_pass=equal==questions,tasks=dict(tasks))
        metric['wall_ms']=stats([r['wall_ms'] for r in selected])
        for stage in ('prefill','decode','eva','detection','map','multimodal_preparation'):
            metric[stage+'_gpu_ms']=stats([r['components'].get(stage,{}).get('gpu_ms',0) for r in selected])
        metric['llm_gpu_ms']=stats([sum(r['components'].get(s,{}).get('gpu_ms',0) for s in ('prefill','decode')) for r in selected])
        metric['generated_tokens']=sum(len(x[0])-1 for r in selected for x in r['generated_ids'])
        metric['useful_decode_tokens']=sum(max(0,len(x[0])-2) for r in selected for x in r['generated_ids'])
        gpu=sum(r['components'].get('decode',{}).get('gpu_ms',0) for r in selected)
        metric['useful_decode_tokens_per_second']=1000*metric['useful_decode_tokens']/gpu if gpu else None
        metric['peak_allocated_gib']=stats([r['peak_allocated']/2**30 for r in selected])
        metric['cache_peak_mib']=max(r['diagnostics'].get('cache_bytes',0) for r in selected)/2**20
        metric['m1_maintenance_ms']=stats([r['m1'].get('m1_total_including_adapters_ms',0) for r in selected])
        metric['m2_actual_reused_cameras']=sum(r['m2'].get('actual_reused_cameras',0) for r in selected)
        metric['cold_sequence_wall_ms']=stats([r['wall_ms'] for r in selected if r['cold_sequence']])
        metric['later_frame_wall_ms']=stats([r['wall_ms'] for r in selected if not r['cold_sequence']])
        physical=[];useful=[];reused=[];padding=[];util=[];input_tokens=[];state_active=0
        for r in selected:
            diag=r['diagnostics'];requests=diag.get('requests',[]);batch=diag.get('batch',{});seed=diag.get('shared_seed') or {}
            n=sum(d.get('input_tokens',0) for d in requests)
            if not requests:n=sum(r['_original_input_lengths'])
            input_tokens.append(n)
            saved=batch.get('actual_reused_tokens',seed.get('actual_saved_prefix_tokens',sum(d.get('reused_tokens',0) for d in requests)))
            reused.append(saved)
            useful.append(batch.get('effective_prefill_tokens',(sum(d.get('effective_prefill_tokens',0) for d in requests)
                if requests else n)+seed.get('prefix_tokens',0)))
            physical.append(batch.get('physical_prefill_tokens',useful[-1]))
            padding.append(dict(suffix=batch.get('suffix_padding_tokens',0),decode=batch.get('decode_padding_tokens',0)))
            state_active+=sum(d.get('state_retrieval',{}).get('admitted',False) for d in requests)
            clock=r['components'].get('service_clock')
            if clock:util.extend(v[1] for v in gpu_records if clock['start_epoch_s']<=v[0]<=clock['end_epoch_s'])
        metric.update(actual_reused_prefix_tokens=sum(reused),effective_prefill_tokens=sum(useful),physical_prefill_tokens=sum(physical),
            input_tokens=sum(input_tokens),original_input_tokens=sum(sum(r['_original_input_lengths']) for r in selected),
            suffix_padding_tokens=sum(x['suffix'] for x in padding),decode_padding_tokens=sum(x['decode'] for x in padding),
            state_context_activated_questions=state_active,nvml_utilization_percent=stats(util),
            nvml_scope='1 Hz samples inside each synchronized service interval, finite-window descriptive utilization')
        summary[mode]=metric
    for mode,m in summary.items():
        m['prefill_speedup']=summary['baseline']['prefill_gpu_ms']['p50']/m['prefill_gpu_ms']['p50']
        m['e2e_speedup']=summary['baseline']['wall_ms']['p50']/m['wall_ms']['p50']
        active=mode not in ('state_only','state_full') or m['state_context_activated_questions']>0
        m['quality_constrained_offline_e2e_speedup']=m['e2e_speedup'] if m['frozen_token_target_pass'] and active else None
    return summary,failures


def paired_scenes(rows):
    baseline={(r['repeat'],r['token']):r for r in rows if r['mode']=='baseline'};result={}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        if mode=='baseline':continue
        by_scene=defaultdict(list)
        for r in rows:
            if r['mode']==mode:by_scene[r['scene']].append((baseline[(r['repeat'],r['token'])]['wall_ms'],r['wall_ms']))
        pairs=np.array([[np.mean([x[0] for x in v]),np.mean([x[1] for x in v])] for v in by_scene.values()])
        rng=np.random.RandomState(73);draw=rng.randint(len(pairs),size=(4096,len(pairs)));sample=pairs[draw].mean(1)
        ratio=1-sample[:,1]/sample[:,0]
        result[mode]=dict(scene_count=len(pairs),mean_savings_ms=float((pairs[:,0]-pairs[:,1]).mean()),
            descriptive_scene_cluster_reduction_ci95=np.percentile(ratio,[2.5,97.5]).tolist(),
            scope='Scene is the sampling unit; two replays are not independent seeds. Not a population guarantee.')
    return result


def numerical_summary(root,common):
    rows=lines(root/'sequences.jsonl');out={}
    if common:rows+=lines(common/'sequences.jsonl')
    for policy in dict.fromkeys(r['policy'] for r in rows):
        picked=[r for r in rows if r['policy']==policy]
        comparisons=[q for r in picked for q in r['comparisons']]
        out[policy]=dict(frames=len(picked),questions=len(comparisons),token_equal=sum(q['equal'] for q in comparisons),
            wall_ms=stats([r['wall_ms'] for r in picked]),
            prefill_gpu_ms=stats([r['components']['prefill']['gpu_ms'] for r in picked]),
            decode_gpu_ms=stats([r['components']['decode']['gpu_ms'] for r in picked]),
            peak_allocated_gib=stats([r['peak_allocated']/2**30 for r in picked]),
            limitation='One real-carrier replay per observed frame; separate from continuous VLM E2E. Original numerical_02 unfiltered decode max includes different histories after divergence.')
    return out


def trace_summary(path):
    events=read(path)['traceEvents'];names=Counter(e.get('name') for e in events)
    shapes=[e.get('args',{}).get('Input Dims') for e in events if e.get('name')=='aten::bmm']
    shapes=[s for s in shapes if s and len(s[0])==3]
    return dict(path=str(path),sha256=digest(path),cuda_kernel_events=sum(e.get('cat')=='kernel' for e in events),
        regions={k:v for k,v in names.items() if k and k.startswith('m3_')},
        bmm_query_shapes=sorted({tuple(s[0]) for s in shapes}))


def make_figures(root,summary,numeric,overhead,single,numerical_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    directory=root/'figures';directory.mkdir(exist_ok=True);manifest={}
    def save(fig,name,evidence):
        for ext in ('png','pdf'):fig.savefig(directory/(name+'.'+ext),dpi=220)
        plt.close(fig);manifest[name]=evidence
    modes=list(summary);labels=[LABELS.get(m,m) for m in modes];x=np.arange(len(modes))
    fig,ax=plt.subplots(figsize=(10,3.8));bars=ax.bar(x,[summary[m]['e2e_speedup'] for m in modes])
    for bar,m in zip(bars,modes):
        if not summary[m]['frozen_token_target_pass']:bar.set_hatch('//')
    ax.axhline(1,color='black',lw=1);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=35,ha='right')
    ax.set(ylabel='Baseline P50 / method P50',title='Continuous VLM speedup; hatching = frozen output equality failed')
    save(fig,'e2e_speedup','Complete measured frames; failed quality remains marked, not approved')
    fig,ax=plt.subplots(figsize=(10,3.8));pre=[summary[m]['prefill_gpu_ms']['p50'] for m in modes];dec=[summary[m]['decode_gpu_ms']['p50'] for m in modes]
    ax.bar(x,pre,label='Prefill');ax.bar(x,dec,bottom=pre,label='Decode');ax.legend();ax.set_xticks(x);ax.set_xticklabels(labels,rotation=35,ha='right');ax.set(ylabel='Cumulative GPU interval (ms)',title='Five original questions, max_new_tokens=320')
    save(fig,'language_breakdown','Measured full causal-LM intervals including vocabulary projection; component medians nonadditive')
    fig,ax=plt.subplots(figsize=(7,3.5))
    for policy in ('native','deterministic','no_fp16_reduction','fp32_lora'):
        values=read(numerical_root/(policy+'_operators.json'))['prefix_kv']
        ax.plot([v['layer'] for v in values],[v['key_max_abs'] for v in values],label=policy)
    ax.set(xlabel='Decoder layer',ylabel='Prefix K MaxAbs',title='Same prefix dependencies, different floating point results');ax.legend(fontsize=7)
    save(fig,'prefix_numerics','Actual full-prefix K differences across all prefix positions, fixed real failed sample')
    fig,axes=plt.subplots(1,3,figsize=(9,3.2))
    for ax,(a,b,title) in zip(axes,[('clone','borrow','Prefix branch'),('legacy_preparation','preparation','Input preparation'),('legacy_match','match','Prefix matching')]):
        ax.bar(['Before','After'],[overhead[a]['wall_ms']['p50'],overhead[b]['wall_ms']['p50']]);ax.set(title=title,ylabel='Controlled wall P50 ms')
    save(fig,'runtime_overhead','10 measured repetitions after 2 warmups; real resident carriers and actual legacy KV')
    fig,ax=plt.subplots(figsize=(10,3.6));ax.bar(x,[summary[m]['peak_allocated_gib']['p50'] for m in modes]);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=35,ha='right');ax.set(ylabel='Peak allocated GiB',title='Persistent prefix + live request/workspace included')
    save(fig,'memory','torch.cuda.max_memory_allocated per measured service')
    smodes=list(single);fig,ax=plt.subplots(figsize=(8,3.5));ax.bar([LABELS.get(m,m) for m in smodes],[single[m]['wall_ms']['p50'] for m in smodes]);ax.tick_params(axis='x',rotation=25);ax.set(ylabel='VLM wall P50 ms',title='Native one-question sampling workload; no shared-prefix benefit assumed')
    save(fig,'single_question','Separate one-question real workload, original sampling temperature/top-p/320-token limit')
    write(directory/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--single-run',required=True)
    p.add_argument('--numerical',required=True);p.add_argument('--common-prefix',required=True);p.add_argument('--numeric-e2e',required=True)
    p.add_argument('--state-sweep');p.add_argument('--test-log',required=True);p.add_argument('--output',default='outputs/module3_refinement');args=p.parse_args()
    root=Path(args.output);run=Path(args.run);single_run=Path(args.single_run);numeric_run=Path(args.numeric_e2e)
    rows,main_audit=audit(run);single_rows,single_audit=audit(single_run);numeric_rows,numeric_audit=audit(numeric_run)
    measured=[r for r in rows if not r['warmup']];single_measured=[r for r in single_rows if not r['warmup']]
    records=nvml(run);single_records=nvml(single_run)
    summary,failures=summarize(measured,records);single,single_failures=summarize(single_measured,single_records)
    nsummary,nfail=summarize([r for r in numeric_rows if not r['warmup']],nvml(numeric_run))
    split={}
    for name in ('calibration','heldout'):
        selected=[r for r in measured if r['split']==name];split[name]=summarize(selected,records)[0]
    scenes=read(run/'protocol.json')['scenes'];cal={s['scene'] for s in scenes if s['split']=='calibration'};held={s['scene'] for s in scenes if s['split']=='heldout'}
    assert len(cal)==6 and len(held)==2 and not cal&held
    # Fill native inputs from the real original input-ID/carrier snapshots.
    for group in [summary,*split.values(),single,nsummary]:
        for m,v in group.items():
            v['input_token_scope']='Native raw/expanded inputs retained per frame in carriers/*.pt; optimized counts are diagnostic actual forwards'
    numeric=numerical_summary(Path(args.numerical),Path(args.common_prefix))
    controlled=run/'controlled_probes';overhead=read(controlled/'overhead.json')
    assert overhead['cache_immutable_after_borrow']
    state_rows=[r for r in measured if r['mode'] in ('state_only','state_full')]
    reasons=Counter(d['state_retrieval']['reason'] for r in state_rows for d in r['diagnostics'].get('requests',[]) if 'state_retrieval' in d)
    active=sum(v['state_context_activated_questions'] for m,v in summary.items() if m in ('state_only','state_full'))
    state=dict(activated_questions=active,verified_history_records=sum(len((r['state_snapshot'] or {}).get('records',[])) for r in state_rows),
        reasons=dict(reasons),decision='Feasibility No-Go: no usable verified historical task input' if not active else 'Offline approximation only; no independent task certificate',
        provenance_rule='M1 precheck snapshot before current perception commit; stale/same-frame new predictions rejected',
        source_frame_ids='Unknown: current M1 anchors retain timestamps/cameras but not source frame IDs',
        source=str(run/'frames.jsonl'))
    sweep=Path(args.state_sweep) if args.state_sweep else root/'state_sweep_01'
    if (sweep/'completion.json').exists():
        extended=read(sweep/'completion.json');assert extended['complete'] and extended['weights_versions_unchanged']
        coverage=lines(sweep/'coverage.jsonl');assert len(coverage)==extended['frames']
        state['extended_history']=dict(summary=extended,source=str(sweep),coverage_sha256=digest(sweep/'coverage.jsonl'),
            verification_rejections=dict(sum((Counter(r['verification_rejection_reasons']) for r in coverage),Counter())),
            note='Perception/M1 feasibility only; no synthetic QA and no VLM latency claim')
        if (sweep/'conditional_qa.jsonl').exists():
            conditional=lines(sweep/'conditional_qa.jsonl')
            state['extended_history']['controlled_qa']={m:dict(questions=sum(len(r['questions']) for r in conditional if r['mode']==m),
                token_equal=sum(sum(r['token_equal']) for r in conditional if r['mode']==m)) for m in ('full','compact','state_only','state_full')}
    tests=Path(args.test_log).read_text();passed=int(re.findall(r'(\d+) passed',tests)[-1]);skipped=sum(map(int,re.findall(r'(\d+) skipped',tests)))
    assert not re.search(r'\d+ (?:failed|errors?)',tests) and skipped==0
    traces=dict(batch=trace_summary(controlled/'batched_decode_trace.json'),borrow=trace_summary(controlled/'runtime_borrow_trace.json'))
    assert traces['batch']['regions'].get('m3_batch_shared_prefix')==1 and traces['batch']['regions'].get('m3_batch_suffix_prefill')==1
    admission={}
    for mode,metric in summary.items():
        if mode=='baseline':continue
        admission[mode]=dict(calibration_token_target=split['calibration'][mode]['frozen_token_target_pass'],
            heldout_token_target=split['heldout'][mode]['frozen_token_target_pass'],
            measured_positive_p50_net_benefit=metric['e2e_speedup']>1,
            state_context_activated=metric['state_context_activated_questions'],
            auto_approved=False,
            reason='No deployment certificate covering exact workload Shape, generation contract and independent quality/net-cost evidence; aggregate speedup is not approval')
    audit_result=dict(main=main_audit,single=single_audit,numeric_e2e=numeric_audit,
        test_log=str(args.test_log),tests_passed=passed,tests_skipped=skipped,traces=traces,
        calibration_scenes=sorted(cal),heldout_scenes=sorted(held),scene_sets_disjoint=True,
        numerical_source=str(args.numerical),common_prefix_source=str(args.common_prefix),
        report_script_sha256=digest(__file__),raw_sampling_proof=read(controlled/'sampling_proof.json')['token_equal'])
    audit_result['offline_analysis_sources']={p:digest(p) for p in ('tools/worldstream_m3/qa_analysis.py','tools/worldstream_m3/offline_reference.py')}
    write(root/'delivery_audit.json',audit_result);write(root/'performance_results.json',dict(all=summary,split=split,numeric_e2e=nsummary,paired_scene_statistics=paired_scenes(measured)))
    write(root/'quality_results.json',dict(main_failures=failures,single_failures=single_failures,numeric_e2e_failures=nfail,
        paired_original_gt_source=str(run/'qa_ground_truth_analysis.json'),limits='Token F1 is not semantic accuracy or safety certification'))
    write(root/'single_question_results.json',single);write(root/'numerical_summary.json',numeric);write(root/'runtime_overhead.json',overhead)
    write(root/'state_feasibility.json',state);write(root/'admission_review.json',admission)
    write(root/'cost_table.json',dict(approved=[],reason='No automatically published per-workload production calibration',evidence=str(root/'performance_results.json')))
    write(root/'quality_certificate.json',dict(approved=[],reason='Frozen failures and/or missing independent task/generation coverage; no threshold relaxation',evidence=str(root/'quality_results.json')))
    make_figures(root,summary,numeric,overhead,single,Path(args.numerical))
    build_report(root,args,summary,split,single,nsummary,numeric,state,overhead,audit_result,admission)
    write(root/'delivery_status.json',dict(status='implementation_and_gpu_protocol_complete',final_gpu_experiments_complete=True,
        online_optimization_approved=False,persistent_state_quality_certified=False,tests_passed=passed,tests_skipped=skipped,
        source=str(run),single_source=str(single_run),numeric_e2e_source=str(numeric_run),report='refinement_report.md',
        decision='Retain native Auto fallback; review candidate-specific measured quality and speed separately'))


def build_report(root,args,summary,split,single,nsummary,numeric,state,overhead,audit_result,admission):
    text=['# M3 Refinement：数值路径、持久状态与语言执行改进','',
        '**代码改进、真实 GPU 实验及质量审查已完成。当前不发布生产优化批准，默认 Auto 保持原生完整生成。逐策略的实测通过情况见下表；工程交付完成不代表持久状态质量目标或全部优化目标已经达成。**','',
        f'原始 OmniDrive/LoRA 权重、RTX 3090、OmniDrive 环境；主实验 `{args.run}`，单问题 `{args.single_run}`。旧 `outputs/module3/` 保留不覆盖。GPU 回归 {audit_result["tests_passed"]} 项通过、跳过 {audit_result["tests_skipped"]}。','',
        '## 协议与真实执行边界','',
        '保留六个原场景、每场景两个连续帧和五个原始问题，新增两个互斥场景作为 held-out。按原问题文本选择任务覆盖，不读 GT 做选择；一轮预热、两轮计时，各方法按场景/重放轮换。额外数值 E2E 只针对原失败场景单独运行，不加入主实验性能统计。默认单问题另测原始 Sampling。全部完整生成保持 max_new_tokens=320，EOS 正常结束，不减少问题数或截短答案。',
        'CUDA Prefill/Decode interval 覆盖整个 CausalLM.forward（含词表投影）；完整 wall 包含当前视觉、Detection/Map、真实 M1/M2 服务、M3 签名/匹配/输入/缓存与语言生成。数据加载/scatter、采样监控及回答记录的额外 CPU 拷贝/落盘在计时外；原生输出解析仍计入服务。Profiler 控制实验不作为常规速度表。独立 GPU 利用率以 1 Hz 采样，与实际服务时间戳对齐；短窗口利用率不作为瞬时 SM 占用率。','',
        f'主实验有 {audit_result["main"]["normalized_duplicate_fallback_records"]} 条记录在 native fallback 和 runtime 返回处重复记录 IDs；实际 Prefill 调用数仍与问题数一致。分析仅保留最后一组完整有序 runtime 返回，并逐项验证额外记录正好对应 native fallback、所有 Token IDs 解码与原始回答一致。raw frames.jsonl 未改写，GPU 生成未重复，也未丢弃任何问题。后续单问题与数值 E2E 使用已修正记录器。','',
        '## 1. Exact Prefix 数值定位','',
        '旧实验两条失败是同一真实样本在两次计时重放中重复失败，不是两种独立案例。本轮保留该样本完整生成，并覆盖六场景十二个实际帧的不同问题长度和自然生成长度。区分输入前缀完全相同、因果依赖合法、浮点结果相同和最终 argmax 相同四个命题。',
        '首个可观测分歧位于第 0 层 Q/K/V 基础 Linear：公共 Embedding 完全相同，采样行的 input/RMSNorm 相同，但 577 行与 570 行 FP16 投影不同；Q/K MaxAbs=0.00390625。该位置 LoRA A/B 的采样输出仍相同，因此 LoRA 不是本案例的首发源。差异经 RoPE、attention score/softmax/output、MLP 和 residual 传播。真实全前缀逐层 K/V、32 层采样算子、最终 norm/LM Head，以及原生与后缀路径的中间张量均保存。',
        '控制 GEMM sweep 固定相同输入行与原始权重，只改变矩阵行数，直接复现基础投影差异；FP32 误差更小但转回 FP16 后仍可跨舍入边界，不能据此声称恢复原始输出。当前实际后端是 Transformers 4.31 eager LlamaAttention，不运行 FlashAttention；没有升级后端或混入未经验证的 FlashAttention 性能。Profiler 记录算子 Shape 与 CUDA Kernel，但不把观察到的 F.linear Shape 敏感性扩大为已隔离某个特定 cuBLAS 内部算法的证明。','',
        '| 真实 Carrier 完整序列候选 | Token 一致 | Prefill GPU P50/P95 ms | 完整 LLM wall P50/P95 ms |','|---|---:|---:|---:|']
    for name,r in numeric.items():text.append(f'| {name} | {r["token_equal"]}/{r["questions"]} | {r["prefill_gpu_ms"]["p50"]:.2f}/{r["prefill_gpu_ms"]["p95"]:.2f} | {r["wall_ms"]["p50"]:.2f}/{r["wall_ms"]["p95"]:.2f} |')
    text+=['','独立公共前缀顺序计算是新增候选，保留其失败记录，不因批量探针某个样本通过就称为全序列修复。原始 numerical_02 的 decode 最大差包含分歧后的不同历史，不把该值当成同输入误差；控制实验 same_history_logits.json 只比较首次分歧之前及当步。原失败动作首次分歧为第 28 个生成 Token。没有引入低 logit margin 的“保证一致”回退规则，也未改变冻结目标。','',
        '### 数值候选的真实 VLM E2E（失败场景独立协议）','',
        '| 方法 | Prefill GPU P50 ms | VLM wall P50/P95 ms | Token 一致 |','|---|---:|---:|---:|']
    for mode,r in nsummary.items():text.append(f'| {mode} | {r["prefill_gpu_ms"]["p50"]:.2f} | {r["wall_ms"]["p50"]:.2f}/{r["wall_ms"]["p95"]:.2f} | {r["token_equal"]}/{r["questions"]} |')
    text+=['','精度设置只在语言 runtime 调用内生效，不改变视觉 Backbone/Head 的计算策略。局部 FP32 仅计算 LoRA GEMM，不编辑、合并或整体转成 FP32 模型。每次退出恢复 forward 和全局精度/确定性设置。','',
        '## 2. Task-Verified World-State Retrieval','',
        '新增持久语义视图、轻量任务查询与可控文本插入。视图在真实 M1 precheck 后、当前感知 commit 前冻结；要求匹配 scene/frame/time/epoch/execution signature，源观测必须早于当前帧且在有效时限内。只接受 M1 当前验证的历史 anchor，保留类别、相对坐标、已有可靠速度、几何、源/验证时间、Camera 与确实有效的 Feature Version。原 M1 不保存每 anchor 的源 Frame ID，因此保持 Unknown；不伪造灯色、动作意图、拓扑或 Anchor-to-Carrier 映射。',
        '按问题的类、方向、运动依赖检索；地图查询仅用地图 anchor，运动依赖需要可靠速度。部分历史不用于证明全场数量/缺失，全局安全/未知问题保持完整视觉。C=历史状态文本替代 Carrier；D=状态文本+Full Carrier，均是位置/格式变化的近似输入。文本超预算、无验证证据、过时或任务覆盖不足均直接走原生完整生成。B 为原有 task deletion，A 为完整原输入，C/D 不偷偷混入 Prefix 收益。',
        f'本轮状态视图累计包含 {state["verified_history_records"]} 条验证历史记录，真正进入 LLM 的状态问题数 {state["activated_questions"]}。结论：**{state["decision"]}**。不能将 C/D 全回退后的输出一致称为世界状态输入质量通过。来源与回退原因见 state_feasibility.json。','',
        '## 3. Shared-Prefix Batched Decode 与 KV Runtime','',
        '新增批执行器独立计算实际共同前缀，按真实后缀长度左侧 padding，维护每行 positions/mask，并由原始 HF greedy loop 管理 EOS 和长度上限，恢复原问题顺序并剔除 EOS 后为同批其他请求产生的 pad。Sampling、定制停止/处理器和不支持的输出契约回退逐问题原生 generate，避免宣称改变 Batch RNG 调度仍逐位相同。持久和活跃工作区分别预算，空间不足回退。',
        '只读分支仅对已核实的 4.31 原生 LlamaAttention out-of-place torch.cat 开放，其他后端保留 clone。它省掉显式分支 clone；attention 的 cat 仍产生独立动态 KV，批量 KV 也会物化各行前缀，未实现 paged KV 或物理零拷贝。控制实验逐张量验证持久 KV 未修改，真实峰值包含所有动态开销。','',
        '| 运行时组件，真实控制实验 | 原 wall P50 ms | 新 wall P50 ms |','|---|---:|---:|']
    for a,b in [('clone','borrow'),('legacy_preparation','preparation'),('legacy_match','match')]:text.append(f'| {a} → {b} | {overhead[a]["wall_ms"]["p50"]:.4f} | {overhead[b]["wall_ms"]["p50"]:.4f} |')
    text+=[f'','单分支避免的显式前缀拷贝为 '+f'{overhead["clone_bytes_avoided_per_branch"]/2**20:.2f} MiB。签名检查仍覆盖实际模型/LoRA/forward 与数值设置，未为节约几毫秒删除有效性检查。CUDA event 跨越 CPU 签名检查时包含排队空隙，因此开销表使用 wall，不将签名 interval 称为 GPU 算术。','',
        '## 4. 五问题完整服务结果','',
        '| 方法 | Prefill GPU P50/P95 ms | Decode GPU P50/P95 ms | VLM wall P50/P95 ms | E2E 加速 | Token 一致 |','|---|---:|---:|---:|---:|---:|']
    for mode,r in summary.items():
        fmt=lambda k:f'{r[k]["p50"]:.2f}/{r[k]["p95"]:.2f}'
        text.append(f'| {mode} | {fmt("prefill_gpu_ms")} | {fmt("decode_gpu_ms")} | {fmt("wall_ms")} | {r["e2e_speedup"]:.3f}× | {r["token_equal"]}/{r["questions"]} |')
    text+=['','Refined Exact 保留原始算术与第一问题来源，加入只读 KV、向量化 provenance/LCP 和单问题快速回退；失败的独立前缀数值候选单独归档。Refined Batch 将这些运行时改进与批生成结合；M1+M2+Batch 保留完整 M1 服务及既有 M2 camera auto。记录实际 Camera 命中，不把没有发生的视觉复用计为收益。','',
        '| 方法 | Decode 有效 tokens/s | 输出 Token 总数 | 后缀 pad / Decode pad | 实际节省前缀 Token | peak allocated P50/P95 GiB | 计时窗口 NVML 利用率 P50 |','|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in summary.items():
        util=r['nvml_utilization_percent'];mean='缺失' if util is None else f'P50 {util["p50"]:.1f}%'
        text.append(f'| {mode} | {r["useful_decode_tokens_per_second"]:.2f} | {r["generated_tokens"]} | {r["suffix_padding_tokens"]}/{r["decode_padding_tokens"]} | {r["actual_reused_prefix_tokens"]} | {r["peak_allocated_gib"]["p50"]:.3f}/{r["peak_allocated_gib"]["p95"]:.3f} | {mean} |')
    text+=['','有效 Decode 吞吐按所有问题的真实非 padding 后续 Token 数除以 Decode GPU 时间；批量 forward 调用次数不能当作 Token 数。原始输出长度和 pad 开销保留，未将回答变短当作每 Token 算法加速。NVML 为离散窗口统计，不称为逐 kernel 占用率。冷缓存表示序列第一帧，不是进程冷启动；cold/later 分位在 performance_results.json，未把相邻第二帧短文本 LCP 当成共享视觉。','',
        '## 5. 独立测试质量、GT 与单问题','',
        '| 方法 | 六场景 calibration Token 一致 | 两场景 held-out Token 一致 |','|---|---:|---:|']
    for mode in summary:
        a=split['calibration'][mode];b=split['heldout'][mode]
        text.append(f'| {mode} | {a["token_equal"]}/{a["questions"]} | {b["token_equal"]}/{b["questions"]} |')
    text+=['','逐任务 Token 一致、原始 GT Token F1、归一化 exact、可对齐数值列表误差和数量不匹配全部保留。F1 仅是文本重叠诊断，不等于语义准确率；安全动作缺少文本 GT 的项目保持 Unknown，没有当前 GT 参与在线查询或选择。',
        '离线补齐了原生 description 模板对应的 desc/val 描述注释，并记录源文件 SHA256；这是读取原有注释，不修改 GT 或在线选择。keywords/val 不存在，未把 desc/action 的长理由文本冒充 short-action 的原始目标。扩展注释覆盖后的 F1 不与上一轮不同覆盖范围的绝对 F1 混比；本轮方法始终按同问题、同注释配对。原始 GT 与有效 mask 按 sample token 评价六步 Planning：','',
        '| 方法 | 轨迹解析数 | ADE m | FDE m |','|---|---:|---:|---:|']
    qa=read(Path(args.run)/'qa_ground_truth_analysis.json')
    for r in qa['summary']:
        if not r['warmup'] and r['planning_samples']:
            ade='缺失' if r['planning_ade_m'] is None else f'{r["planning_ade_m"]:.6f}'
            fde='缺失' if r['planning_fde_m'] is None else f'{r["planning_fde_m"]:.6f}'
            text.append(f'| {r["mode"]} | {r["planning_parsed"]}/{r["planning_samples"]} | {ade} | {fde} |')
    text+=['','这些轨迹误差不替代碰撞率或闭环安全认证。错误原文、GT 和 Token IDs 在 quality_results.json；规划 GT 和 task 明细在各 run 的 qa_ground_truth_analysis.json。','',
        '| 默认单问题 Sampling | VLM wall P50/P95 ms | 实际共享 Token | Token 一致 | 感知/Memory 摘要匹配 |','|---|---:|---:|---:|---:|']
    single_rows=[r for r in lines(Path(args.single_run)/'frames.jsonl') if not r['warmup']]
    sb={(r['repeat'],r['token']):r for r in single_rows if r['mode']=='baseline'}
    for mode,r in single.items():
        values=[v for v in single_rows if v['mode']==mode];matches=sum(v['perception_digest']==sb[(v['repeat'],v['token'])]['perception_digest'] for v in values)
        text.append(f'| {mode} | {r["wall_ms"]["p50"]:.2f}/{r["wall_ms"]["p95"]:.2f} | {r["actual_reused_prefix_tokens"]} | {r["token_equal"]}/{r["questions"]} | {matches}/{len(values)} |')
    text+=['','单问题没有同帧共享机会，Refined Exact/Batch 显式提前回退，不做 M3 签名/展开/缓存捕获。微小时延波动不能归因于不存在的 KV 复用；M1/M2 维护成本保留在联合运行中。采样 batch 默认原生顺序回退，固定 RNG 对照见 sampling_proof.json。','',
        '## 6. 双重准入与最终十项结论','',
        '| 策略 | calibration 逐 Token 目标 | held-out 逐 Token 目标 | 实测 P50 净收益 | Auto 批准 |','|---|---:|---:|---:|---:|']
    for mode,r in admission.items():text.append(f'| {mode} | {r["calibration_token_target"]} | {r["heldout_token_target"]} | {r["measured_positive_p50_net_benefit"]} | False |')
    text+=['','改进后的 common-prefix Auto 入口要求独立且互斥场景的质量证据，并同时匹配 model version、execution key、完整问题长度列表、prefix length 和 generation key；成本必须包含冷捕获/维护且缓存满足预算。不能把旧 first-question cache 成本行用于批准独立 prefix seeding，也不能用 greedy 行批准不同 Sampling 配置。实际 approved 数组仍为空，未因某张表全局平均较好制造证书。批量与状态输入仍是显式离线实验。','',
        '1. 原失败的首发数值路径定位到基础 Q/K/V GEMM 的 Shape 敏感性，证据包括相同输入、逐层 hook、直接投影 sweep 和真实完整生成。两次旧失败为一个唯一请求的重放。',
        '2. 确定性与局部精度候选及独立 prefix seeding 的全序列结果如实列出；没有建立普适误差界或宣称所有 Exact 生成已修复。',
        f'3. M1 接口实际支持带来源的历史类别/几何/已有可靠速度查询；本协议实际激活 {state["activated_questions"]} 个状态问题，不伪装成已经获得可用覆盖。',
        '4. 未取得可上线的独立紧凑状态质量证据。C/D 无可用状态时完整回退，不能用回退答案一致性批准新输入格式。',
        '5. 批量 Decode 的速度、pad、吞吐、峰值与逐 Token 质量分别见完整服务表；单个控制样本通过不能推广到全部场景。',
        '6. 显式 KV clone 和输入 provenance 同步开销降低已由真实控制实验量化；匹配/签名仍保留完整依赖检查。',
        '7. 单问题走原生快速回退，未把五问题收益套用为单问题加速。',
        '8. 五问题实测收益必须同时读取质量列；quality_constrained_offline_e2e_speedup 对失败或未真正激活的新输入为 null。',
        '9. 联合运行包括实际 M1/M2 维护，M2 camera 实际命中数单列；M3 不要求视觉 cache 命中。',
        '10. 当前不批准生产优化；native Full 回退持续生效。精确输入依赖、完整输出保留、GT 质量和道路安全分别报告，没有放宽冻结标准。','',
        '## 复现与归档','',
        '命令见 commands.sh，GPU 全流程见 tools/worldstream_m3/run_refinement.sh。各运行保存 source/source_manifest、原始 Question/GT/IDs、回答文件、计时与峰值、1 Hz GPU 记录、状态来源和真实 Profiler。numerical_01 的追踪形状错误及两个控制阶段 pilot 记录均保留，不混入正式性能。推理参数版本未修改，当前源码与正式 GPU 快照逐文件一致。','',
        '![E2E](figures/e2e_speedup.png)','![Language](figures/language_breakdown.png)',
        '![Numerics](figures/prefix_numerics.png)','![Runtime](figures/runtime_overhead.png)',
        '![Memory](figures/memory.png)','![Single](figures/single_question.png)','']
    if 'extended_history' in state:
        sweep=state['extended_history'];r=sweep['summary']
        extra=['### 更长连续历史的可行性核查','',
            f'为避免只看两帧冷历史就作全局结论，另对 {r["scenes"]} 个实际场景各执行 {r["frames_per_scene"]} 个连续帧，共 {r["frames"]} 帧。该阶段仅执行原生感知与真实 M1，显式关闭语言服务；LM 参数对象保留并恢复，参数版本不变，没有伪造回答或把该时延称为 VLM 加速。',
            f'累计当前验证的历史记录 {r["verified_history_records"]} 条，满足任务条件且可装入上下文预算的查询 {r["task_queries_with_valid_context"]} 个。结论：**{r["decision"]}**。原始逐帧验证拒绝原因、Query 和冻结上下文见 `{sweep["source"]}`。','']
        if 'controlled_qa' in sweep:
            extra+=['可用历史帧上的 A/B/C/D 问答是独立真实 Carrier 重放，保持原问题和 320 上限，不混入主连续性能表。','',
                '| 补充控制策略 | Token 一致 |','|---|---:|']
            for mode,q in sweep['controlled_qa'].items():extra.append(f'| {mode} | {q["token_equal"]}/{q["questions"]} |')
            extra+=['','这仍不是生产任务质量证书；原 GT 仅在生成后评价。']
        index=text.index('## 3. Shared-Prefix Batched Decode 与 KV Runtime')
        text[index:index]=extra+['']
    (root/'refinement_report.md').write_text('\n'.join(text))


if __name__=='__main__':main()
