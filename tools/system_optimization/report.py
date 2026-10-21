"""Report only real measured rows; historical evidence stays explicitly labelled."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .audit import write, stats


def measured_stages(frame):
    """Recover legacy suffix labels from explicit executed KV-branch events.

    Older timers used `past_key_values != None` as decode. These experiments
    clamp shared prefixes below every prompt length, so the first actual model
    call after a branch is an embedding suffix prefill. Never rewrite raw rows;
    new timers record the true input shape and classify directly.
    """
    pending=False
    shared=bool(frame['m3'].get('shared_seed') or frame['m3'].get('batch'))
    for stage in frame['stages']:
        label=stage['name']
        if label=='m3_kv_branch':pending=True
        if label in ('llm_prefill','llm_decode'):
            if label=='llm_decode' and pending and shared and 'input_shape' not in stage:
                label='llm_prefill'
            pending=False
        yield label,stage


def prefill_token_accounting(frame):
    """Count executed forward rows, including independent seed and padding.

    Producer `reused_tokens` has different first-question charging conventions;
    it is not itself the number of GPU prefill tokens avoided.
    """
    prefill=[s for label,s in measured_stages(frame) if label=='llm_prefill']
    actual=sum(s['forward_tokens'] for s in prefill) if prefill and all('forward_tokens' in s for s in prefill) else None
    diag=frame['m3'];requests=diag.get('requests',[]);batch=diag.get('batch',{})
    if actual is None and 'physical_prefill_tokens' in batch:actual=batch['physical_prefill_tokens']
    if actual is None and batch.get('groups') and all('physical_prefill_tokens' in g['diagnostics'] for g in batch['groups']):
        actual=sum(g['diagnostics']['physical_prefill_tokens'] for g in batch['groups'])
    if actual is None and requests and all('forward_prefill_lengths' in q for q in requests):
        actual=sum(sum(q['forward_prefill_lengths']) for q in requests)+(diag.get('shared_seed') or {}).get('prefix_tokens',0)
    original=sum(q['input_tokens'] for q in requests) if requests and all('input_tokens' in q for q in requests) else None
    approximate=any(q.get('selection',{}).get('approximate') for q in requests)
    return dict(physical_forward_tokens=actual,
        duplicate_prefill_tokens_avoided=original-actual if original is not None and actual is not None and not approximate else None)


def summarize(rows):
    groups=defaultdict(list)
    references={(r['run'],r['repeat'],r['index']):r['generated_ids'] for r in rows if r['mode']=='original'}
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
        stage_wall=defaultdict(list);stage_cpu=defaultdict(list);stage_exclusive=defaultdict(list)
        for row in frames:
            by_stage=defaultdict(float)
            by_wall=defaultdict(float);by_cpu=defaultdict(float);by_exclusive=defaultdict(float)
            for label,stage in measured_stages(row['frame']):
                if 'cuda_interval_ms' in stage:
                    by_stage[label]+=stage['cuda_interval_ms']
                for field,bucket in (('wall_ms',by_wall),('cpu_process_ms',by_cpu),('exclusive_wall_ms',by_exclusive)):
                    if field in stage:bucket[label]+=stage[field]
            for name,value in by_stage.items():components[name].append(value)
            for source,target in ((by_wall,stage_wall),(by_cpu,stage_cpu),(by_exclusive,stage_exclusive)):
                for name,value in source.items():target[name].append(value)
        planning=[task['planning'] for r in frames for task in r['tasks'] if task['planning'] is not None]
        qa=[task['text_overlap'] for r in frames for task in r['tasks'] if task['text_overlap'] is not None]
        qs=sum(r['frame']['question_count'] for r in frames)
        def question_equality(selected):
            equal=total=0
            for r in selected:
                reference=references.get((r['run'],r['repeat'],r['index']))
                if reference is None:return None
                total+=r['frame']['question_count']
                equal+=sum(a==b for a,b in zip(r['generated_ids'],reference))
            return dict(equal=equal,total=total)
        requests=[q for r in frames for q in r['frame']['m3'].get('requests',[])]
        state_queries=[q['state_retrieval'] for q in requests if q.get('state_retrieval') is not None]
        activated=sum(bool(q.get('admitted')) for q in state_queries)
        consumed={(r['frame']['identity']['frame_id'],state['anchor_id']) for r in frames
            for request in r['frame']['m3'].get('requests',[]) if (request.get('state_retrieval') or {}).get('admitted')
            for state in request['state_retrieval'].get('records',[])}
        def batch_metric(row,name):
            batch=row['frame']['m3'].get('batch',{})
            if name in batch:return batch[name]
            return sum(g['diagnostics'].get(name,0) for g in batch.get('groups',[]))
        generated=sum(max(0,len(ids[0])-1) for r in frames for ids in r['generated_ids'])
        useful_decode=sum(max(0,len(ids[0])-2) for r in frames for ids in r['generated_ids'])
        m1=[r['frame']['m1'].get('m1_total_including_adapters_ms',0.) for r in frames]
        accepted=sum(r['frame']['m1'].get('valid_anchor_count',0) for r in frames)
        token_counts=[prefill_token_accounting(r['frame']) for r in frames]
        decoded=sum(components.get('llm_decode',[]))
        def fallback_reasons(row):
            frame=row['frame'];reasons=list(frame['fallback_reasons'])
            batch_fallback=frame['m3'].get('batch_fallback')
            if batch_fallback is not None:
                reasons.append('batch_full_fallback:'+batch_fallback.get('reason','unknown'))
            for group in frame['m3'].get('batch',{}).get('groups',[]):
                diag=group['diagnostics']
                if diag.get('admitted') is False:reasons.append('batch_group_full_fallback:'+diag.get('reason','unknown'))
            for request in frame['m3'].get('requests',[]):
                query=request.get('state_retrieval')
                if query is not None and not query.get('admitted'):
                    reasons.append('state_query_full_fallback:'+query.get('reason','unknown'))
            return reasons
        output.append(dict(run=run,mode=mode,service=service,workload=frames[0]['workload'],
            measured_frames=len(frames),unique_frames=len({i['frame_id'] for i in identities}),
            scenes=len({i['scene_id'] for i in identities}),questions=qs,
            e2e_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames]),
            preprocessing_wall_ms=stats([r['frame']['input_preparation_wall_ms'] for r in frames]),
            components_cuda_interval_ms={k:stats(v) for k,v in components.items()},
            components_wall_ms={k:stats(v) for k,v in stage_wall.items()},
            components_cpu_process_ms={k:stats(v) for k,v in stage_cpu.items()},
            components_exclusive_wall_ms={k:stats(v) for k,v in stage_exclusive.items()},
            timing_classification='Actual input-shape labels when present; legacy cached suffixes identified by explicit preceding KV branch; raw rows unchanged',
            exact_full_output_equality=exact,
            token_equal_frames=sum(r['frame']['quality']['exact_token_equality'] for r in frames),
            complete_question_token_equality=question_equality(frames),
            token_equality_by_split={s:question_equality([r for r in frames if r['split']==s]) for s in ('calibration','heldout')},
            generated_tokens=generated,useful_decode_tokens=useful_decode,
            effective_decode_tokens_per_second=useful_decode/(decoded/1000) if decoded else None,
            m1_maintenance_ms=stats(m1),verified_anchor_records=accepted,
            useful_state_yield_per_ms=0. if state_queries and not activated and sum(m1)>0 else None,
            actual_skipped_windows=sum(r['frame']['m2'].get('actual_skipped_windows',0) for r in frames),
            actual_reused_cameras=sum(r['frame']['m2'].get('actual_reused_cameras',0) for r in frames),
            actual_reused_kv_tokens=sum(q.get('reused_tokens',0) for q in requests),
            suffix_padding_tokens=sum(batch_metric(r,'suffix_padding_tokens') for r in frames),
            decode_padding_tokens=sum(batch_metric(r,'decode_padding_tokens') for r in frames),
            physical_prefill_forward_tokens=sum(c['physical_forward_tokens'] for c in token_counts) if all(c['physical_forward_tokens'] is not None for c in token_counts) else None,
            duplicate_prefill_tokens_avoided=sum(c['duplicate_prefill_tokens_avoided'] for c in token_counts) if all(c['duplicate_prefill_tokens_avoided'] is not None for c in token_counts) else None,
            state_activated_questions=activated,state_queries=len(state_queries),
            task_query_hit_rate=activated/len(state_queries) if state_queries else None,
            task_consumed_unique_frame_anchor_pairs=len(consumed),
            fallback_frames=sum(bool(fallback_reasons(r)) for r in frames),
            fallback_rate=sum(bool(fallback_reasons(r)) for r in frames)/len(frames),
            fallback_reasons=dict(__import__('collections').Counter(reason for r in frames for reason in fallback_reasons(r))),
            memory=dict(cache_peak_bytes=max(r['frame']['memory']['tracked_cache_bytes'] for r in frames),
                cache_peak_scope='Observed before budget enforcement; includes payload explicitly invalidated at completion. Not a post-eviction residency guarantee.',
                m1_payload_accounted=all('m1_payload_bytes' in r['frame']['memory'] for r in frames),
                gpu_peak_allocated=max(r['frame']['memory'].get('gpu_peak_allocated',0) for r in frames),
                gpu_peak_reserved=max(r['frame']['memory'].get('gpu_peak_reserved',0) for r in frames),
                cpu_rss_highwater_bytes=max(r['frame']['memory'].get('cpu_rss_highwater_bytes',0) for r in frames)),
            cold_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames if r['cold']]),
            warm_wall_ms=stats([r['frame']['e2e_wall_ms'] for r in frames if not r['cold']]),
            qa_ground_truth=dict(available_questions=len(qa),total_questions=qs,
                exact_matches=sum(bool(q['exact']) for q in qa),
                mean_lexical_token_f1=float(np.mean([q['token_f1'] for q in qa])) if qa else None,
                numeric_mae=stats([q['numeric_mae'] for q in qa if q['numeric_mae'] is not None]),
                note='Original annotation text overlap, not an LLM judge or driving safety metric'),
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
    full_input=any(e.get('name')=='stream/input_preparation' for e in events)
    return dict(source=str(path),evidence='separate_real_full_gpu_trace_not_latency_sample',
        kernel_count=len(kernels),kernel_duration_sum_ms=sum(e['dur'] for e in kernels)/1000,
        kernel_active_union_ms=union([(e['ts'],e['ts']+e['dur']) for e in kernels]),
        scope='input_preparation_scatter_and_inference' if full_input else 'inference_only_preparation_outside_trace',
        gpu_copy_ms={kind:sum(e['dur'] for e in copies if kind.lower() in e['name'].lower())/1000 for kind in ('HtoD','DtoH','DtoD')},
        synchronize_cpu_ms=sum(e['dur'] for e in cuda if 'synchroniz' in e['name'].lower())/1000,
        allocation_api_ms=sum(e['dur'] for e in cuda if 'malloc' in e['name'].lower() or 'free' in e['name'].lower())/1000,
        kernel_names=dict(__import__('collections').Counter(e['name'] for e in kernels).most_common(15)),
        launch_gap_ms=None,note='CUDA intervals include idle gaps; no inference from FLOPs. Gap attribution remains unmeasured.')


def figures(root, summary, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (root/'figures').mkdir(parents=True,exist_ok=True)
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
                xs=[row['preprocessing_wall_ms']['p50']]+[row['components_cuda_interval_ms'].get(k,{}).get('p50',0) for k in stages]+[row['e2e_wall_ms']['p50']]
                ax.bar(np.arange(len(stages)+2)+i*.35,xs,width=.35,label=row['workload'])
            ax.set_xticks(np.arange(len(stages)+2)+.17);ax.set_xticklabels(['input prep\n(wall)']+stages+['E2E\n(wall)'],rotation=15)
            ax.set_ylabel('P50 ms; components = CUDA intervals (not additive)');ax.legend()
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
        joint=next((r for r in summary if r['run']=='multi' and r['mode']=='m123'),None)
        queried=next((r for r in summary if r['run']=='language_candidates' and r['mode']=='state_only'),None)
        if joint and queried:
            def consumption(ax):
                values=[joint['verified_anchor_records'],joint['actual_reused_cameras'],queried['state_activated_questions']]
                ax.bar(range(3),values);ax.set_xticks(range(3))
                ax.set_xticklabels(['M1 verified physical\nanchor records','M2 actual reused\ncameras',f"M3 state-input activations\n({queried['state_queries']} actual queries)"])
                for i,value in enumerate(values):ax.annotate(str(value),(i,value),xytext=(0,4),textcoords='offset points',ha='center')
                ax.set_ylabel('Actual counts; units differ, no implied conversion')
            save('S4','M1 to M2 to M3: actual state consumption',consumption)
        candidates=[r for r in summary if r['run']=='language_candidates']
        if candidates:
            def language_quality(ax):
                for i,r in enumerate(candidates):
                    label=r['mode']
                    if label.startswith('state_'):label+=f" ({r['state_activated_questions']} state activations)"
                    ax.scatter(r['p50_ratio_to_original'] or 1,r['token_equal_frames']/r['measured_frames'],
                        color=f'C{i}',label=label,marker='o' if r['exact_full_output_equality'] else 'x',s=45)
                ax.set_xlabel('P50 ratio to native');ax.set_ylabel('Frames with complete token equality')
                ax.set_ylim(0,1.08);ax.legend(loc='upper right',fontsize=8)
            save('S5','Offline language quality vs speed (no Auto certificate)',language_quality)
        save('S6','Single/multi service wall measured separately',lambda ax:(
            ax.bar(range(len(base)),[r['e2e_wall_ms']['p50'] for r in base]),
            ax.set_xticks(range(len(base))),ax.set_xticklabels([r['workload'] for r in base]),ax.set_ylabel('Full service P50 ms')))
        budget_names=['budget_0','budget_128','language_candidates']
        sample_sets=[{r['index'] for r in rows if r['run']==name and r['mode']=='original' and not r['warmup']} for name in budget_names]
        common=set.intersection(*sample_sets) if all(sample_sets) else set()
        if common:
            prompts=[json.loads((root/name/'questions.json').read_text()) for name in budget_names]
            common={i for i in common if all(p[str(i)]==prompts[0][str(i)] for p in prompts)}
        if common:
            references={(r['run'],r['repeat'],r['index']):r['frame']['e2e_wall_ms'] for r in rows if r['mode']=='original'}
            points=[r for r in rows if r['run'] in budget_names and r['mode']=='batch' and not r['warmup'] and r['repeat']==1 and r['index'] in common]
            def memory_serving(ax):
                for name,budget,color in zip(budget_names,(0,128,512),('C0','C1','C2')):
                    for equal in (True,False):
                        selected=[r for r in points if r['run']==name and r['frame']['quality']['exact_token_equality']==equal]
                        if not selected:continue
                        ax.scatter([r['frame']['memory']['tracked_cache_bytes']/1024**2 for r in selected],
                            [references[(r['run'],r['repeat'],r['index'])]/r['frame']['e2e_wall_ms'] for r in selected],
                            color=color,marker='o' if equal else 'x',label=f'M3 budget {budget} MiB: tokens '+('equal' if equal else 'differ'))
                ax.axhline(1,color='black',linewidth=.7);ax.set_xlabel('Actual retained cache MiB')
                ax.set_ylabel('Paired E2E ratio to native, same frame');ax.legend(fontsize=7)
            save('S7','Offline Batch: M3 cache budgets on matched real frames',memory_serving)
            made['S7'].update(matched_frame_indices=sorted(common),repeat=1,scope='same frames/questions, no independent extra samples or production approval')
        long=[r for r in rows if r['run'].startswith('long_stream') and not r['warmup']]
        if long:
            def draw_long(ax):
                for mode in sorted({r['mode'] for r in long}):
                    values=[r for r in long if r['mode']==mode]
                    ax.plot([r['frame']['e2e_wall_ms'] for r in values],label=mode)
                state=[r for r in long if r['mode']=='m123']
                if state:
                    other=ax.twinx();other.plot([r['frame']['m1'].get('valid_anchor_count',0) for r in state],
                        color='black',linestyle=':',alpha=.6,label='M1 rule-verified physical anchors')
                    other.set_ylabel('M1 verified anchor count (not task certification)')
                    boundaries=[i for i in range(1,len(state)) if state[i]['frame']['identity']['scene_id']!=state[i-1]['frame']['identity']['scene_id']]
                    for boundary in boundaries:ax.axvline(boundary,color='gray',linestyle='--',linewidth=.7)
                ax.set_xlabel('Unique frame order; dashed = scene switch');ax.set_ylabel('E2E ms');ax.legend(fontsize=7)
            save('S8','Long-stream latency and physical state validity',draw_long)
        ablation=[r for r in summary if r['run']=='single']
        if ablation:
            save('S9','Cross-module ablation; service semantics are labelled',lambda ax:(
                ax.bar(range(len(ablation)),[r['e2e_wall_ms']['p50'] for r in ablation]),
                ax.set_xticks(range(len(ablation))),ax.set_xticklabels([r['mode']+'\n'+r['service'] for r in ablation],rotation=60,fontsize=7),ax.set_ylabel('E2E P50 ms')))
        admitted=[r for r in summary if r['quality_constrained_service_speedup'] is not None and r['run'] in ('single','multi')]
        if admitted:
            save('S10','Observed output-equal service ratios (not deployment approval)',lambda ax:(
                ax.bar(range(len(admitted)),[r['quality_constrained_service_speedup'] for r in admitted]),
                ax.set_xticks(range(len(admitted))),ax.set_xticklabels([r['run']+':'+r['mode'] for r in admitted],rotation=75,fontsize=7),
                ax.set_ylabel('P50 ratio to service-matched reference'),ax.axhline(1,color='black',linewidth=.7)))
    for i in range(1,11):made.setdefault('S'+str(i),dict(status='unmeasured_no_fabricated_figure'))
    write(root/'figures/manifest.json',made)
