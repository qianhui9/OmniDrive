"""Report only real measured rows; historical evidence stays explicitly labelled."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
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


def main(root):
    root=Path(root);rows=[];runs={};sources={}
    for path in sorted(root.glob('*/frames.jsonl')):
        run=path.parent.name
        # Smoke runs check the harness and are never merged with formal tables.
        if run.startswith('smoke'):continue
        status=json.loads((path.parent/'status.json').read_text())
        runs[run]=status
        if status.get('status')!='completed':continue
        if run=='lifecycle_validation':continue  # Separate controlled correctness validation.
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            row=json.loads(line);row['run']=run;rows.append(row)
    summary=summarize(rows)
    # Offline description of actual selected sequences, never online admission
    # or quality calibration. No instance IDs exist in this info file, so class
    # count changes are not labelled as individual enter/leave ground truth.
    annotation_path=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    coverage=[]
    if annotation_path.exists():
        import pickle
        with annotation_path.open('rb') as handle:infos=pickle.load(handle)['infos']
        for run in runs:
            if runs[run].get('status')!='completed' or run=='lifecycle_validation':continue
            protocol_path=root/run/'protocol.json'
            if not protocol_path.exists():continue
            protocol=json.loads(protocol_path.read_text())
            for scene in protocol['scenes']:
                selected=[infos[i] for i in scene['indices']]
                ts=[i['timestamp']/1e6 for i in selected]
                positions=np.asarray([i['ego2global_translation'] for i in selected])
                class_counts=[dict(__import__('collections').Counter(map(str,i['gt_names']))) for i in selected]
                coverage.append(dict(run=run,scene=scene['scene'],split=scene['split'],
                    scene_name=selected[0].get('scene_name'),description=selected[0].get('description'),
                    unique_frames=len(selected),duration_s=ts[-1]-ts[0],
                    measured_frame_gaps_s=np.diff(ts).tolist(),ego_sampled_path_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
                    classes_observed=sorted(set(k for counts in class_counts for k in counts)),
                    class_counts=class_counts,independent_instance_enter_leave_recall=None,
                    scope='Offline annotation coverage only, not a safety score or an online state input'))
        del infos
    write(root/'sequence_coverage.json',dict(source=str(annotation_path),sequences=coverage))
    write(root/'scheduler_results.json',dict(results=summary,run_status=runs,source_sha256=sources))
    for name,keep in [('single_question_results',lambda r:r['run']=='single'),
            ('multi_question_results',lambda r:r['run'] in ('multi','language_candidates')),
            ('long_stream_results',lambda r:r['run'].startswith('long_stream')),
            ('ablation_results',lambda r:r['run'] in ('single','multi')),
            ('memory_results',lambda r:True)]:
        write(root/(name+'.json'),dict(evidence='new_real_weight_gpu',results=[r for r in summary if keep(r)]))
    write(root/'exact_correctness.json',dict(results=[{k:r[k] for k in ('run','mode','measured_frames','token_equal_frames','exact_full_output_equality')} for r in summary],
        controlled_lifecycle=[dict(source=str(p),**json.loads(p.read_text())) for p in root.glob('*/lifecycle_probe/results.json')],
        caveat='No reuse when skipped computation is zero; full fallback equality is not reuse certification'))
    write(root/'task_quality.json',dict(results=[dict(run=r['run'],mode=r['mode'],planning=r['planning'],qa_ground_truth=r['qa_ground_truth']) for r in summary],
        qa_rows=[dict(run=r['run'],mode=r['mode'],index=r['index'],tasks=r['tasks']) for r in rows if not r['warmup']],
        physical_false_acceptance='not newly measured; historical heldout evidence in state_consumption.json and workload_audit.md',
        critical_change_recall='not newly measured',official_detection_map_metrics=None,closed_loop_safety=None))
    with (root/'cross_module_trace.jsonl').open('w') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    profiles=[trace_summary(p) for p in root.glob('*/pipeline_trace.json')
        if not p.parent.name.startswith('smoke') and runs.get(p.parent.name,{}).get('status')=='completed']
    latency=json.loads((root/'latency_breakdown.json').read_text())
    latency['new_gpu_profiles']=profiles;latency['new_frame_components']=summary
    bounds={}
    for workload in ('single','multi'):
        native=[r for r in rows if r['run']==workload and r['mode']=='original' and not r['warmup']]
        if not native:continue
        repeated=[];decode=[];walls=[]
        for r in native:
            stages=r['frame']['stages']
            prefill=[s['cuda_interval_ms'] for s in stages if s['name']=='llm_prefill' and 'cuda_interval_ms' in s]
            if len(prefill)!=r['frame']['question_count']:continue
            walls.append(r['frame']['e2e_wall_ms'])
            repeated.append(sum(prefill[1:]))
            decode.append(sum(s.get('cuda_interval_ms',0.) for s in stages if s['name']=='llm_decode'))
        if walls:
            bounds[workload]=dict(source='new original per-forward CUDA intervals and measured complete frame wall',
                eliminate_later_question_prefill=dict(saving_ms=stats(repeated),
                    residual_e2e_ms=stats([w-s for w,s in zip(walls,repeated)]),
                    aggregate_time_ratio=sum(walls)/sum(w-s for w,s in zip(walls,repeated)),
                    assumption='Optimistic zero-cost later-question prefill, including unique suffixes; first question retained. Single-question saving exactly zero.'),
                eliminate_decode=dict(saving_ms=stats(decode),
                    residual_e2e_ms=stats([w-s for w,s in zip(walls,decode)]),
                    aggregate_time_ratio=sum(walls)/sum(w-s for w,s in zip(walls,decode)),
                    assumption='Ideal complete removal of measured decoder-forward intervals, not a deployable autoregressive method'))
    latency['new_idealized_bounds']=bounds
    write(root/'latency_breakdown.json',latency)
    historical=json.loads((root/'state_consumption.json').read_text())
    historical['new_runs']=[dict(run=r['run'],mode=r['mode'],verified_anchor_records=r['verified_anchor_records'],
        task_queries=r['state_queries'],activated_questions=r['state_activated_questions']) for r in summary]
    audits=[]
    for path in root.glob('*/m1_state.jsonl'):
        if (path.parent.name.startswith('smoke') or path.parent.name=='lifecycle_validation'
                or runs.get(path.parent.name,{}).get('status')!='completed'):continue
        unique={}
        for line in path.open():
            entry=json.loads(line);p=entry['precheck']
            unique.setdefault((p['scene_id'],p['frame_id']),p)
        reasons=__import__('collections').Counter();checked=accepted=0;areas=[]
        for p in unique.values():
            anchors=p.get('diagnostics',{}).get('anchors',{})
            checked+=len(anchors);accepted+=len(p['valid_anchor_ids'])
            for check in anchors.values():reasons.update(check.get('reasons',[]))
            for region in p['invalidation_regions'].values():
                if 'verified_stable_area_ratio' in region:areas.append(region['verified_stable_area_ratio'])
        audits.append(dict(run=path.parent.name,source=str(path),unique_frames=len(unique),
            history_anchor_checks=checked,accepted_physical_anchors=accepted,
            verified_physical_anchor_coverage=accepted/checked if checked else None,
            mean_verified_observation_area_ratio=float(np.mean(areas)) if areas else None,
            rejection_reasons=dict(reasons),reason_counts_are_multilabel=True,
            task_consumable_states='see actual M3 query/activation rows; not implied by physical coverage',
            physical_false_acceptance_rate=None,valid_state_recovery=None,critical_change_recall=None,
            denominator='First occurrence of each real frame within this run; repeated timings and modes deduplicated'))
    historical['new_m1_rejection_audit']=audits
    historical['rejection_interpretation']=dict(
        mandatory_until_independent_evidence=['motion_unknown','no_reliable_view','observation_mismatch','projected_occlusion','source/version mismatch'],
        potentially_improvable_with_real_historical_evidence=['no_historical_observation','temporal_support_insufficient','temporal_source_missing'],
        no_validation_threshold_disabled=True)
    write(root/'state_consumption.json',historical)
    reuse=json.loads((root/'reuse_coverage.json').read_text())
    reuse['new_real_moving_runs']=[{k:r[k] for k in (
        'run','mode','measured_frames','unique_frames','actual_skipped_windows',
        'actual_reused_cameras','actual_reused_kv_tokens')} for r in summary]
    reuse['controlled_repeats_excluded']=True
    write(root/'reuse_coverage.json',reuse)
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
        '**结论：尚未建立可生产批准的质量约束端到端净加速。** '
        '真实移动输入的视觉跳算和任务世界状态消费以实测计数为准；'
        '离线 Prefix/Batch 未满足完整原生输出等价，Auto 保持 Full 回退。'
        '生命周期和来源正确性通过不代表已获得状态化计算收益。','',
        f"本轮正式计时共 {sum(r['measured_frames'] for r in summary)} 次方法帧执行，"
        f"覆盖 {len({r['frame']['identity']['frame_id'] for r in rows})} 个唯一真实帧、"
        f"{len({r['frame']['identity']['scene_id'] for r in rows})} 个场景。"
        '唯一数量跨方法、预热及重复去重；故障控制和独立 Profiler 不计入。','',
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
    # Concrete answers accompany the general interpretation above, never infer
    # quality or speed when a workload has not completed.
    report+=['','## 本轮关键实测值','']
    for workload in ('single','multi'):
        selected=[r for r in summary if r['run']==workload]
        if not selected:
            report.append(f'- {workload}：尚无完整运行结果。');continue
        for r in selected:
            if r['mode'] in ('original','m1','m123','unified_persistent','unified_output'):
                report.append(f"- {workload}/{r['mode']}：M1 P50={r['m1_maintenance_ms']['p50']:.2f} ms；"
                    f"实际 Camera 复用={r['actual_reused_cameras']}，Window 跳算={r['actual_skipped_windows']}；"
                    f"问题完整 Token 等价={r['complete_question_token_equality']}。")
            if r['mode']=='original':
                report.append(f"- {workload} 原生 Planning：{r['planning']['parsed']}/{r['planning']['questions']} 可解析；"
                    f"ADE P50={r['planning']['ade_m']['p50']} m，FDE P50={r['planning']['fde_m']['p50']} m；"
                    f"原始 QA 文本标注可用 {r['qa_ground_truth']['available_questions']}/{r['questions']}，"
                    f"平均词级 F1={r['qa_ground_truth']['mean_lexical_token_f1']}。未标注问题不填补答案。")
    for r in summary:
        if r['run']=='language_candidates':
            report.append(f"- 离线 {r['mode']}：完整 Token 等价={r['complete_question_token_equality']}；"
                f"状态查询激活={r['state_activated_questions']}/{r['state_queries']}；"
                f"源计算 KV 复用={r['actual_reused_kv_tokens']} tokens；仍无生产证书。")
    long=[r for r in summary if r['run'].startswith('long_stream')]
    if long:
        report+=['','## 长序列与生命周期','',
            '以下为新场景的连续真实帧；跨方法重复不增加独立样本数。场景运动、类别和实际帧间隔见 sequence_coverage.json。']
        for r in long:
            report.append(f"- {r['mode']}：{r['scenes']} 场景、{r['unique_frames']} 唯一帧；"
                f"完整输出等价={r['exact_full_output_equality']}；实际相机复用={r['actual_reused_cameras']}；"
                f"保留缓存峰值={r['memory']['cache_peak_bytes']/1024**2:.2f} MiB，"
                f"GPU allocated 峰值={r['memory']['gpu_peak_allocated']/1024**3:.2f} GiB。")
        for path in root.glob('*/lifecycle_probe/results.json'):
            control=json.loads(path.read_text())
            report.append(f"- 单独受控复用/故障恢复：实际复用 {control['controlled_reuse_source']['actual_reused_cameras']} 个相机；"
                f"清空后的恢复与干净原生完整结果相同={control['recovered_matches_fresh_native']}。"
                f"详见 {path.relative_to(root)}；受控重复不计入上表速度或复用覆盖率。")
            if control.get('real_scene_switch') is not None:
                report.append(f"- 同一 Runtime 的真实场景切换：触发 {control['real_scene_switch']['lifecycle_resets']}；"
                    f"完整输出/特征/时序 memory 与干净原生结果相同={control['real_scene_switch']['matches_fresh_native']}。")
    decode=root/'single/decode_probe/results.json'
    if decode.exists():
        result=json.loads(decode.read_text())
        report.append('- 原生单问题 no_grad / inference_mode 专项：`'+json.dumps(result['summary'],ensure_ascii=False)+'`。'
            '单个真实输入的通用执行上下文控制，不是世界状态收益或独立后端准入。')
    report+=['','冷缓存表示场景第一帧；不表示进程冷启动。模型加载、每序列的配置构造和 GC 不在逐帧服务计时内。'
        '早期 single 运行的细分缓存计数只覆盖 M1 捕获张量，未含全部 CPU 历史数组；其 CPU RSS/GPU peak 是实际测量。'
        '后续运行 m1_payload_accounted=True 时包含历史图像、几何、掩码和证据。各运行源快照明确保留该诊断差异。']
    report+=['','## 缓存预算、刷新与复现边界','',
        'budget_0 / budget_128 设置每个计算模块的缓存上限为 0 / 128 MiB，统一 Runtime 的总保留预算为该值的两倍；'
        '默认对应 512 / 1024 MiB。S7 仅比较相同帧、原始问题和计时轮次的离线 Batch，标注完整 Token 是否相同。'
        '预算为零时持久历史会显式失效，不能把其耗时降低解释为同等历史维护服务的加速。',
        'stride_refresh 同时设置 frame_stride=2、full_refresh_interval=2，与默认间隔为联合敏感性对照；'
        '这不能独立估计两项设置各自的因果贡献。真实帧间隔保存在 sequence_coverage.json。',
        '源码改动清单和补丁见 instrumentation_changelog.md、stage_BC.patch、stage_DE.patch；'
        '逐轮源快照与最终交付差异见 delivery_source_audit.json。']
    (root/'experiment_report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/system_optimization')
    main(parser.parse_args().root)
