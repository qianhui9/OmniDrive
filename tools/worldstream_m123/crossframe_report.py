"""Strict paired audit and reports for the one-route cross-frame experiment."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pstats
import pickle
import re
import numpy as np

from tools.worldstream_m2.global_residual_benchmark import read, rows
from tools.worldstream_m2.global_residual_analysis import compare_quality
from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality, quality_upper, fit_policy, cross_frame_visual
from tools.worldstream_m123.final_speed_analysis import cluster_bootstrap

METRICS=('prefill_cuda_ms','decode_cuda_ms','m3_wall_ms','model_ms','request_ms',
    'preprocessing_ms','transfer_and_collate_ms','output_transfer_ms','non_model_ms')
KINDS=('native','zero','reuse')


def value(frame,key):
    if key=='non_model_ms':return frame['request_ms']-frame['model_ms']
    if key in ('prefill_cuda_ms','decode_cuda_ms'):
        return frame['stages'].get('prefill' if key=='prefill_cuda_ms' else 'decode',{}).get('gpu_span_ms',0.)
    if key=='m3_wall_ms':
        return frame['stages']['m3_auto_total' if frame.get('m3') else 'native_generation']['host_wall_ms']
    return frame[key]


def reuse(questions):
    counts=Counter();cross=[];lengths=[];ages=[];depths=[]
    for q in questions:
        d=q.get('reuse_diagnostics') or {};length=q['reused_prefix_length'];counts['input_tokens']+=q['input_tokens']
        counts['effective_prefill_tokens']+=q['input_tokens']-length
        if d:
            if sum(d['forward_prefill_lengths'])!=d['effective_prefill_tokens'] or d['effective_prefill_tokens']!=q['input_tokens']-length:
                raise ValueError('Actual prefill forward lengths do not reconcile')
            if d['reuse_length']!=length:raise ValueError('Planner/executor reuse mismatch')
        if not length:continue
        lengths.append(length);ages.append(d['cache_age_frames']);depths.append(d['approx_reuse_depth'])
        counts['actually_reused_kv_tokens']+=length;counts['cache_hits']+=1
        same=d['cache_source_frame']==q['frame_id']
        category=('same_frame_exact' if same else 'cross_frame_exact') if d['reuse_type']=='exact' else ('same_frame_inherited_approx' if same else 'cross_frame_bounded_approx')
        counts[category+'_tokens']+=length
        if not same:
            spans=d.get('visual_group_spans')
            if not spans:raise ValueError('Cross-frame visual provenance missing')
            visual=sum(max(0,min(length,stop)-start) for _,start,stop in spans)
            counts['cross_frame_visual_tokens']+=visual;counts['cross_frame_requests']+=1
            cross.append(dict(scene_id=q['scene_id'],target_frame=q['frame_id'],source_frame=d['cache_source_frame'],
                computed_source=d['computed_source'],prefix_lineage=d['prefix_lineage'],
                question=q['question'],task_profile=d['task_profile'],prefix_length=length,visual_tokens=visual,
                cache_age=d['cache_age_frames'],approx_depth=d['approx_reuse_depth'],root_frame_index=d['root_frame_index'],
                quality_risk=d['estimated_quality_risk'],forward_lengths=d['forward_prefill_lengths'],input_tokens=q['input_tokens'],
                m1_state_valid=d.get('m1_state_valid'),m2_refresh_ratio=d.get('m2_refresh_ratio')))
    for key in ('actually_reused_kv_tokens','cache_hits','same_frame_exact_tokens','cross_frame_exact_tokens',
            'cross_frame_bounded_approx_tokens','cross_frame_visual_tokens','cross_frame_requests'):
        counts.setdefault(key,0)
    return dict(counts,requests=len(questions),cache_hit_ratio=counts['cache_hits']/len(questions),
        skipped_fraction=counts['actually_reused_kv_tokens']/counts['input_tokens'],
        cross_frame_reused_tokens=counts['cross_frame_exact_tokens']+counts['cross_frame_bounded_approx_tokens'],
        cross_frame_visual_fraction=counts['cross_frame_visual_tokens']/counts['input_tokens'],
        reused_length=distribution(lengths),all_request_reuse_length=distribution(q['reused_prefix_length'] for q in questions),
        cache_age=distribution(ages),approx_depth=distribution(depths),cross_frame_evidence=cross,
        rejection_reasons=dict(Counter((q.get('reuse_diagnostics') or {}).get('approximate_statistics_rejection')
            or (q.get('reuse_diagnostics') or {}).get('decision_reason','native') for q in questions if not q['reused_prefix_length'])))


def paired(frames,workload,mode,reference,plan):
    result={}
    for key in METRICS:
        samples=[];rr=[];cc=[];rounds=[]
        for repeat in range(3):
            a=frames[f'{workload}_{mode}_r{repeat}_{reference}'];b=frames[f'{workload}_{mode}_r{repeat}_reuse']
            aa=[value(f,key) for f in a];bb=[value(f,key) for f in b]
            rr+=aa;cc+=bb;rounds.append(float(np.mean(aa)-np.mean(bb)))
            samples.extend(dict(scene_id=f['scene_id'],saved_ms=x-y) for f,x,y in zip(a,aa,bb))
        ci=cluster_bootstrap(samples,plan['performance'])
        result[key]=dict(reference_mean=float(np.mean(rr)),candidate_mean=float(np.mean(cc)),
            saved_ms=float(np.mean(rr)-np.mean(cc)),speedup=float(np.mean(rr)/np.mean(cc)) if np.mean(cc)>0 else None,
            round_saved_ms=rounds,scene_bootstrap=ci,positive_ci=ci['lower']>0)
    return result


TRAFFIC={
    'double_yellow':r'\bdouble yellow\b', 'no_passing':r'\b(no[- ]passing|no[- ]overtaking|prohibit\w* (passing|overtaking))\b',
    'one_way':r'\bone[- ]way\b','two_way':r'\btwo[- ]way\b',
    'lane_count':r'\b(one|two|three|four|\d+) lanes?\b'}


def traffic_claims(answer):
    return {key:sorted(set(m[0] for m in re.finditer(pattern,answer.lower()))) for key,pattern in TRAFFIC.items() if re.search(pattern,answer.lower())}


def quality_pairs(root,workload,questions):
    output=[]
    for repeat in range(3):
        native=questions[f'{workload}_free_r{repeat}_native'];zero=questions[f'{workload}_free_r{repeat}_zero'];candidate=questions[f'{workload}_free_r{repeat}_reuse']
        for n,z,c in zip(native,zero,candidate):
            identity=lambda x:(x['frame_id'],x['question'],x['reference'])
            if identity(n)!=identity(z) or identity(z)!=identity(c):raise ValueError('Unpaired natural quality')
            for reference,label in ((n,'native'),(z,'zero')):
                p=paired_quality(c['answer'],reference['answer'],c['reference'])
                before,after=traffic_claims(reference['answer']),traffic_claims(c['answer'])
                new={k:v for k,v in after.items() if v!=before.get(k)}
                output.append(dict(scene_id=c['scene_id'],frame_id=c['frame_id'],repeat=repeat,question=c['question'],reference=c['reference'],
                    baseline=label,baseline_answer=reference['answer'],answer=c['answer'],actual_reuse=c['reused_prefix_length'],
                    new_traffic_claims_for_review=new,**p))
    write(root/(workload+'_quality_pairs.json'),output)
    result={}
    for baseline in ('native','zero'):
        pp=[p for p in output if p['baseline']==baseline]
        changes=[p for p in pp if not p['answer_equal']]
        unique={(p['frame_id'],p['question']) for p in pp}
        kinds=('numeric_exact','spatial_label_match','motion_label_match','safety_label_match')
        result[baseline]=dict(request_measurements=len(pp),unique_requests=len(unique),mean_f1_loss=float(np.mean([p['f1_loss'] for p in pp])),
            # Identical repeats do not become independent quality samples.
            empirical_request_bound=quality_upper([dict(quality=p) for p in pp if p['repeat']==0]),
            scene_f1_loss_ci=cluster_bootstrap([dict(scene_id=p['scene_id'],saved_ms=p['f1_loss']) for p in pp if p['repeat']==0],read(root/'experiment_plan.json')['performance']),
            changed_answers=len(changes),critical_proxy_regressions=sum(p['critical_regressions'] for p in pp),
            new_traffic_claims_needing_review=[p for p in pp if p['repeat']==0 and p['new_traffic_claims_for_review']],
            baseline_proxy_errors={k:sum(p['baseline_task_metrics'].get(k) is False for p in pp if p['repeat']==0) for k in kinds},
            candidate_proxy_errors={k:sum(p['task_metrics'].get(k) is False for p in pp if p['repeat']==0) for k in kinds},
            changed_repeat0_answers=[p for p in changes if p['repeat']==0],
            limitation='Numeric/spatial/motion/safety parsers and traffic-assertion flags are proxies. Human scene review is required to certify changed semantic claims.')
    return result


def calibration(root):
    source=root/'calibration128';records=rows(source/'paired_records.jsonl');original=read(source/'m3_reuse_policy.json')
    fitted,curve=fit_policy(records,original['binding'],read(root/'development_sequence.json'))
    write(root/'calibration_refit.json',dict(policy=fitted,curve=curve,source_sha256=sha256(source/'paired_records.jsonl'),
        known_critical_evidence=['historical_semantic_failure.json','calibration_semantic_review.json'],
        deployment_allowed=False,
        scope='Same frozen development observations; cross-frame evidence fix, no new threshold search or deployment writes. Historical semantic failures remain binding regardless of a raw fitted lexical bound.'))
    stats=[]
    for profile,threshold in sorted({(r['profile'],r['threshold']) for r in records}):
        rr=[r for r in records if r['profile']==profile and r['threshold']==threshold]
        cross=[r for r in rr if cross_frame_visual(r)]
        text=[r for r in rr if r['diagnostics']['cache_source_frame'] and r['diagnostics']['cache_source_frame']!=r['frame_id'] and r['diagnostics']['reused_tokens']==35]
        stats.append(dict(profile=profile,threshold=threshold,requests=len(rr),
            cross_frame_visual_requests=len(cross),cross_frame_visual_reused_tokens=sum(r['diagnostics']['reused_tokens']-35 for r in cross),
            calibration_cross_frame_visual_requests=sum(r['split']=='calibration' for r in cross),
            validation_cross_frame_visual_requests=sum(r['split']=='validation' for r in cross),
            cross_visual_quality_loss_upper=quality_upper(cross),critical_proxy_regressions=sum(r['quality']['critical_regressions'] for r in cross),
            text35_saved_ms=distribution(r['baseline_wall_ms']-r['candidate_wall_ms'] for r in text),
            cross_visual_saved_ms=distribution(r['baseline_wall_ms']-r['candidate_wall_ms'] for r in cross)))
    write(root/'calibration_summary.json',stats)
    return stats


def block_frontiers(root):
    """Actual carrier risk frontiers; no model execution/speed/quality claim."""
    import torch
    from tools.worldstream_m123.latency_controls import load_capture
    from worldstream.m3.reuse_planner import WEIGHTS
    from worldstream.m3.task_conditioner import task_profile
    capture=root/'development_diagnostic/m3_inputs';policy=read(root/'frozen_m3_policy.json')
    frames={};output=[]
    for r in read(root/'prefix_structure.json'):
        source,target=r['source'][1],r['frame_id']
        for token in (source,target):
            if token not in frames:frames[token]=load_capture(capture/(token+'.pt'))
        a,b=frames[source],frames[target]
        current=b['carriers'][0].float();previous=a['carriers'][0].float()
        if current.shape!=previous.shape or a['layout']!=b['layout']:raise ValueError('Carrier layout mismatch')
        markers=(b['ids'][0].reshape(-1)==-200).nonzero().reshape(-1)
        if markers.numel()!=1:raise ValueError('A unique native image expansion is required')
        start=int(markers[0]);profile=task_profile(r['question'])[1];signals=r['world_signals']
        factor=1.+.25*signals['m2_refresh_ratio']+.05*min(4,signals['m2_cache_age_frames'])
        if not signals['m2_context_valid']:factor+=.25
        thresholds=[policy['profiles'].get(profile,{}).get('approx',{}).get('risk_limit',0.)]
        results={}
        for size in (16,32,64):
            blocks=[];offset=0;risk=0.
            for (group,count),weight in zip(b['layout'].groups,WEIGHTS[profile]):
                for lo in range(offset,offset+count,size):
                    hi=min(lo+size,offset+count)
                    old,new=previous[lo:hi],current[lo:hi]
                    delta=(new-old).square().sum(-1)
                    scale=torch.maximum(old.square().sum(-1),new.square().sum(-1)).clamp_min(1e-12)
                    rms=float((delta.sum()/scale.sum()).sqrt());maximum=float((delta/scale).sqrt().max())
                    risk=max(risk,weight*factor*(rms+.25*maximum))
                    blocks.append(dict(group=group,start=start+lo,end=start+hi,rms=rms,max_token=maximum,risk=risk))
                offset+=count
            results[str(size)]=dict(blocks=blocks,
                risk_frontier={str(threshold):max([r['exact_prefix']]+[v['end'] for v in blocks
                    if v['end']<=min(start+128,r['structural_prefix']) and v['risk']<=threshold]) for threshold in thresholds})
        output.append(dict(frame_id=target,source_frame=source,question=r['question'],profile=profile,exact_prefix=r['exact_prefix'],sizes=results))
    write(root/'block_size_frontiers.json',dict(records=output,
        scope='CPU analysis of actual captured native LLM visual carriers, proven Object/Ego/Map expansion; same measured M1/M2 factor. Only raw incremental carrier-risk frontiers, not policy eligibility, semantic approval, skipped forwards or measured speedup. Exact LCP is never rounded by block size.'))


def resource_index(root):
    """Hash used external resources without copying private data into outputs."""
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    annotation=Path(cfg.data.test.ann_file)
    tokens=set(read(root/'development_sequence.json')['tokens']+read(root/'single_sequence.json')['tokens'])
    infos=pickle.load(annotation.open('rb'))['infos'];frames=[]
    paths={annotation,Path('ckpts/iter_10548.pth')}
    for key in ('lane_anno_file',):
        if cfg.data.test.get(key):paths.add(Path(cfg.data.test[key]))
    for path in ('data/nuscenes/data_dict_sample.pkl','data/nuscenes/data_dict_subset_B_val.pkl'):
        if Path(path).exists():paths.add(Path(path))
    paths.update(p for p in Path('ckpts/pretrain_qformer').iterdir() if p.is_file())
    paths.update(p for p in Path('data/nuscenes/v1.0-trainval').glob('*.json') if p.is_file())
    for info in infos:
        if info['token'] not in tokens:continue
        cameras={c:str(Path(v['data_path'])) for c,v in info['cams'].items()}
        paths.update(Path(v) for v in cameras.values())
        for kind in ('conv','vqa'):
            p=Path('data/nuscenes')/kind/'val'/(info['token']+'.json')
            if p.exists():paths.add(p)
        frames.append(dict(frame_id=info['token'],scene_id=info['scene_token'],timestamp=info['timestamp'],cameras=cameras))
    files={}
    for i,path in enumerate(sorted(paths)):
        files[str(path)]=dict(resolved=str(path.resolve()),bytes=path.stat().st_size,sha256=sha256(path))
        if i%100==0:print('RESOURCE SHA',i+1,'/',len(paths),flush=True)
    if {f['frame_id'] for f in frames}!=tokens:raise ValueError('External resource frame coverage missing')
    write(root/'external_resources_sha256.json',dict(files=files,frames=frames,
        scope='All 60 used development/acceptance frames and their six original camera files and labelled question sources; configured annotation/checkpoint/tokenizer/preload files and SDK tables. Private resource contents remain in their existing server paths. This is not a copy or hash of every unused nuScenes sample.'))


def analyze(root):
    plan=read(root/'experiment_plan.json')
    for name,digest in plan['manifest_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen manifest changed')
    if sha256('outputs/final_route/m3_reuse_policy.json')!=sha256(root/'frozen_m3_policy.json'):
        raise ValueError('Formal deployment policy changed during acceptance')
    frames={};questions={};summaries={};csv_rows=[];frame_rows=[];audits={};checkpoint_audits={}
    from tools.worldstream_m2.production_acceptance import validate_rows
    for workload in ('single','multi'):
        sequence=read(root/(workload+'_sequence.json'))
        native_inputs=None;zero_outputs=None
        for mode in ('free','fixed'):
            for repeat in range(3):
                for kind in KINDS:
                    name=f'{workload}_{mode}_r{repeat}_{kind}';run=root/name
                    checkpoint_check=read(run/'checkpoint_validation.json')
                    if read(run/'status.json').get('integrity')!='PASS' or checkpoint_check['status']!='PASS':
                        raise ValueError('Model/worker integrity failure: '+name)
                    checkpoint_audits[name]={k:v for k,v in checkpoint_check.items() if k!='metadata'}
                    for path,digest in read(run/'m1_m2_m3_source_hashes.json').items():
                        if sha256(path)!=digest:raise ValueError('Measured M1/M2/M3 source differs from delivery: '+path)
                    ff=rows(run/'per_frame_metrics.jsonl');qq=rows(run/'per_question_metrics.jsonl')
                    if [f['frame_id'] for f in ff]!=sequence['tokens']:raise ValueError('Input frame order changed')
                    expected=[(t,i,q) for t in sequence['tokens'] for i,q in enumerate(sequence['questions'][t])]
                    if [(q['frame_id'],q['question_index'],q['question']) for q in qq]!=expected:raise ValueError('Natural question order changed')
                    if native_inputs is None:native_inputs=[f['input_tensor_digest'] for f in ff]
                    if [f['input_tensor_digest'] for f in ff]!=native_inputs:raise ValueError('Actual six-camera inputs changed')
                    if kind=='zero' and any(q['reused_prefix_length'] for q in qq):raise ValueError('Zero control reused historical KV')
                    if kind!='native':
                        audit=validate_rows(ff,require_reuse=False)
                        if audit['status']!='PASS':raise ValueError('M1/M2 execution regression')
                        audits[name]=audit
                        outputs=[f['output_digest'] for f in ff]
                        if zero_outputs is None:zero_outputs=outputs
                        if outputs!=zero_outputs:raise ValueError('M3 changed M1/M2 perception outputs')
                    if mode=='fixed':
                        reference=read(root/(workload+'_fixed_reference.json'))
                        for f in ff:
                            refs=reference['frames'][f['frame_id']];steps=sum(len(q['tokens']) for q in refs)
                            frame_questions=[q for q in qq if q['frame_id']==f['frame_id']]
                            prefill_calls=sum(q['input_tokens']>q['reused_prefix_length'] for q in frame_questions)
                            if (f['fixed_tokens']['generated_steps']!=steps or f['fixed_tokens']['actual_decode_forward_calls']!=steps-len(refs)
                                    or f['stages'].get('decode',{}).get('calls',0)!=steps-len(refs)
                                    or f['stages'].get('prefill',{}).get('calls',0)!=prefill_calls
                                    or f['fixed_tokens']['actual_total_llm_forward_calls']!=steps-len(refs)+prefill_calls):
                                raise ValueError('Fixed-token decode forward evidence mismatch')
                        for q in qq:
                            ref=reference['frames'][q['frame_id']][q['question_index']]
                            if q['token_ids'][0][1:]!=ref['tokens'] or q['quality'] is not None:raise ValueError('Fixed-token sequence or quality isolation failed')
                    frames[name]=ff;questions[name]=qq
                    summaries[name]=dict(reuse=reuse(qq),**{k:distribution(value(f,k) for f in ff) for k in METRICS},
                        throughput_questions_s=len(qq)*1000/sum(f['request_ms'] for f in ff),throughput_frames_s=len(ff)*1000/sum(f['request_ms'] for f in ff),
                        amortized_request_ms_per_question=sum(f['request_ms'] for f in ff)/len(qq),
                        peak_gpu_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),peak_gpu_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff),
                        peak_m3_cache_bytes=max((q.get('reuse_diagnostics') or {}).get('cache_bytes',0) for q in qq),
                        peak_m3_branch_workspace_bytes=max((q.get('reuse_diagnostics') or {}).get('branch_workspace_bytes',0) for q in qq))
                    summaries[name]['peak_m2_cache_bytes']=max((f.get('m2') or {}).get('cache_bytes',0) for f in ff)
                    summaries[name]['peak_combined_cache_bytes']=max((f.get('m2') or {}).get('cache_bytes',0)+(f.get('m3') or {}).get('cache_bytes',0) for f in ff)
                    summaries[name]['stage_spans']={label:{boundary:distribution(f['stages'][label][boundary] for f in ff if label in f['stages'])
                        for boundary in ('host_wall_ms','gpu_span_ms')} for label in ('m1_precheck','m1_commit','m2_total','eva_native','detection','map','prefill','decode','m3_auto_total','native_generation')}
                    summaries[name]['m3_host_stages']={key:distribution((q.get('reuse_diagnostics') or {})[key] for q in qq if key in (q.get('reuse_diagnostics') or {}))
                        for key in ('layout_preparation_ms','planning_ms','cache_branch_cpu_enqueue_ms','cache_update_cpu_enqueue_ms','prefill_including_cache_cpu_enqueue_ms')}
                    for q in qq:
                        d=q.get('reuse_diagnostics') or {}
                        csv_rows.append(dict(run=name,mode=mode,kind=kind,repeat=repeat,scene_id=q['scene_id'],frame_id=q['frame_id'],question_index=q['question_index'],
                            question=q['question'],task_profile=d.get('task_profile'),input_tokens=q['input_tokens'],effective_prefill_tokens=q['input_tokens']-q['reused_prefix_length'],
                            actually_reused_tokens=q['reused_prefix_length'],planned_reuse_tokens=d.get('reuse_length',0),reuse_type=d.get('reuse_type','none'),
                            source_frame=d.get('cache_source_frame'),root_frame_index=d.get('root_frame_index'),cache_age=d.get('cache_age_frames'),approx_depth=d.get('approx_reuse_depth'),
                            risk=d.get('estimated_quality_risk'),decision=d.get('decision_reason'),statistics_rejection=d.get('approximate_statistics_rejection'),
                            forward_prefill_lengths=json.dumps(d.get('forward_prefill_lengths')),generated_tokens=q['generated_tokens'],
                            planning_ms=d.get('planning_ms'),layout_preparation_ms=d.get('layout_preparation_ms'),
                            cache_branch_ms=d.get('cache_branch_gpu_span_ms'),cache_update_enqueue_ms=d.get('cache_update_cpu_enqueue_ms')))
                    for f in ff:frame_rows.append(dict(run=name,scene_id=f['scene_id'],frame_id=f['frame_id'],**{k:value(f,k) for k in METRICS},
                        stage_spans=f['stages'],peak_allocated=f['peak_gpu_allocated_bytes'],peak_reserved=f['peak_gpu_reserved_bytes'],
                        input_tensor_digest=f['input_tensor_digest'],perception_output_digest=f['output_digest'],fixed_token_evidence=f.get('fixed_tokens')))
    write(root/'execution_audits.json',audits)
    write(root/'checkpoint_integrity.json',checkpoint_audits)
    test_result=read(root/'regression_test_result.json')
    test_log=(root/test_result['log']).read_text()
    if test_result['status']!='PASS':raise ValueError('Regression tests did not pass')
    if ' passed' not in test_log or ' failed' in test_log:raise ValueError('Regression tests did not pass')
    with (root/'per_request_summary.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    for name,data in [('per_request_summary.jsonl',csv_rows),('per_frame_summary.jsonl',frame_rows)]:
        (root/name).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in data))
    comparisons={w:{mode:{ref:paired(frames,w,mode,ref,plan) for ref in ('native','zero')} for mode in ('free','fixed')} for w in ('single','multi')}
    quality={w:dict(system=compare_quality(root,f'{w}_free_r0_native',f'{w}_free_r0_reuse'),
        paired=quality_pairs(root,w,questions)) for w in ('single','multi')}
    cal=calibration(root)
    block_frontiers(root)
    resource_index(root)
    stats=pstats.Stats(str(root/'full_request_cpu.prof')).stats
    pipeline=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
        for k,v in stats.items() if ('datasets/pipelines' in k[0] or any(s in k[2] for s in ('imread','imdecode','resize','normalize','collate','scatter')))]
    write(root/'full_request_cpu_summary.json',dict(functions=sorted(pipeline,key=lambda x:-x['self_ms']),
        scope='Separate cProfile diagnostic, six real development requests plus warmup and process/model setup. Function self/cumulative times can nest; never add them to infer request latency. Excluded from acceptance timing.'))
    traces=rows(root/'development_diagnostic/decision_traces.jsonl')
    rejected=Counter();by_profile=defaultdict(Counter)
    for trace in traces:
        reasons={v['first_rejection'] for v in trace.get('candidate_evaluations',[]) if v['first_rejection']}
        if not trace.get('cache'):reasons.add('cold_cache')
        for reason in reasons:
            rejected[reason]+=1;by_profile[trace['details']['task_profile']][reason]+=1
    write(root/'diagnostic_rejections.json',dict(requests=len(traces),requests_with_each_reason=dict(rejected),
        by_profile={k:dict(v) for k,v in by_profile.items()},
        scope='Original formal-policy developer diagnostic, actual candidate evaluations; each reason counted at most once per request, not once per block. Timed acceptance decisions remain separately reported.'))
    cross=any(summaries[f'{w}_free_r0_reuse']['reuse']['cross_frame_visual_tokens']>0 for w in ('single','multi'))
    safe=all(q['system']['within_quality_budgets'] and not q['paired']['zero']['critical_proxy_regressions']
        and not q['paired']['native']['critical_proxy_regressions']
        and not q['paired']['native']['new_traffic_claims_needing_review']
        and not q['paired']['zero']['new_traffic_claims_needing_review'] and q['paired']['zero']['empirical_request_bound']<=.02
        and q['paired']['zero']['scene_f1_loss_ci']['upper']<=.02 for q in quality.values())
    m3=all(comparisons[w]['fixed']['zero'][k]['positive_ci'] for w in ('single','multi') for k in ('prefill_cuda_ms','m3_wall_ms'))
    model_gain=all(comparisons[w]['fixed']['zero']['model_ms']['positive_ci'] for w in ('single','multi'))
    e2e=all(comparisons[w]['fixed'][ref]['request_ms']['positive_ci'] and comparisons[w]['fixed'][ref]['request_ms']['speedup']>1
        for w in ('single','multi') for ref in ('native','zero'))
    decision=dict(action='RETAIN_EXISTING_CONSERVATIVE_POLICY',formal_policy_changed=False,
        formal_policy_sha256=sha256(root/'frozen_m3_policy.json'),new_candidate_deployed=False,
        code_correctness=read(root/'planner_equivalence.json')['all_passed'],
        regression_log=test_result['log'],regression_log_sha256=sha256(root/test_result['log']),
        system_absolute_quality_budgets_pass=all(q['system']['within_quality_budgets'] for q in quality.values()),
        m3_quality_proxy_pass=all(not q['paired']['zero']['critical_proxy_regressions']
            and not q['paired']['zero']['new_traffic_claims_needing_review']
            and q['paired']['zero']['empirical_request_bound']<=.02
            and q['paired']['zero']['scene_f1_loss_ci']['upper']<=.02 for q in quality.values()),
        native_critical_proxy_findings_repeat0={w:sum(p['critical_regressions'] for p in q['paired']['native']['changed_repeat0_answers']) for w,q in quality.items()},
        independent_cross_frame_visual_reuse=cross,quality_proxy_gate=safe,prefill_and_m3_latency_ci_positive=m3,
        model_wall_ci_positive=model_gain,
        cross_frame_acceptance_pass=cross and safe and m3,complete_service_acceptance_pass=cross and safe and m3 and model_gain and e2e,
        e2e_positive_paired_ci=e2e,world_referenced_cross_frame_acceleration_claim_supported=cross and safe and m3 and model_gain and e2e,
        limitations=['Six independent scenes; repeats do not increase independent sample size.',
            'Disjointness covers M3/language development and service runs. Prior full-split perception evaluations are disclosed in historical_data_audit.json.',
            'Approximate trial remains development-only. Existing Map/Full-Visual semantic failures are preserved.',
            'Proxy scores cannot certify unreviewed semantic changes.'],
        calibration_decision='No quality/latency qualified new cross-frame rule selected; no risk threshold or internal budget increase.')
    write(root/'deployment_decision.json',decision)
    result=dict(plan=plan,summaries=summaries,comparisons=comparisons,quality=quality,calibration=cal,
        deployment=decision,planner_microbenchmark=read(root/'planner_microbenchmark.json'),
        development_fixed_control=read(root/'development_fixed_control_summary.json'))
    write(root/'summary.json',result)
    reports(root,result)
    write(root/'status.json',dict(status='COMPLETED',code_and_experiments_complete=True,
        measured_service_runs=36,independent_acceptance_scenes=6,
        cross_frame_acceptance_pass=decision['cross_frame_acceptance_pass'],
        complete_service_acceptance_pass=decision['complete_service_acceptance_pass'],formal_policy_changed=False))
    return result


def reports(root,r):
    d=r['deployment'];perf=['# Performance evidence\n',
        'Six independent scenes × six continuous frames; three alternating rounds for each mode and method. Positive savings mean the optimized route is faster. CPU wall and overlapping CUDA spans are separate measurements. No overlapping stage sums are used as attribution.\n',
        'Request/Model/M3 totals refer to one six-camera frame service call: one natural question in single, three ordered natural questions in multi. Prefill/decode forward spans sum those questions within the frame. Frame-request/s and question/s are separately reported; per-question amortized request time is labelled as amortized.\n',
        '| Workload | Boundary | Reference | Prefill saved ms | M3 saved ms | Model saved ms | Request saved ms [scene 95% CI] | E2E speedup |',
        '|---|---|---|---:|---:|---:|---|---:|']
    for workload,modes in r['comparisons'].items():
        for mode,refs in modes.items():
            for ref,m in refs.items():
                q=m['request_ms'];ci=q['scene_bootstrap']
                perf.append(f"| {workload} | {mode} | {ref} | {m['prefill_cuda_ms']['saved_ms']:.3f} | {m['m3_wall_ms']['saved_ms']:.3f} | {m['model_ms']['saved_ms']:.3f} | {q['saved_ms']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] | {q['speedup']:.4f} |")
    perf += ['\nAll raw per-scene savings, repeats, P50/P95 latencies, decode spans, peak allocated/reserved memory, cache bytes and throughput are in `summary.json`. Timed workers contain no capture, DecisionTrace or sampled cProfile/torch.profiler probes. Existing CUDA timing observers remain active equally in paired methods. Fixed-token selection overhead remains included.\n',
        'Prefill/decode CUDA metrics sum actual model-forward event spans. Full HF decode-loop spans (including sampling/host gaps) remain in raw M3 per-request diagnostics; complete M3 wall includes the whole loop. These boundaries are not added together.\n',
        'CUDA peak allocated/reserved memory is reset after warmup and measured over the frame stream. Owned persistent M2/M3 cache bytes and branch-workspace bytes are reported separately from the allocator peak.\n',
        'The earlier ~54.5 ms non-model difference is assessed directly by paired request-minus-model measurements, plus preprocessing, collate/transfer and output transfer boundaries. It is not assumed to be M3 work.\n',
        '| Workload | Fixed reuse vs zero non-model saved ms [95% CI] | Preprocess saved ms | Transfer/collate saved ms |',
        '|---|---|---:|---:|']
    for w in ('single','multi'):
        m=r['comparisons'][w]['fixed']['zero'];n=m['non_model_ms'];ci=n['scene_bootstrap']
        perf.append(f"| {w} | {n['saved_ms']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] | {m['preprocessing_ms']['saved_ms']:.3f} | {m['transfer_and_collate_ms']['saved_ms']:.3f} |")
    perf.append('\nThe multi-question non-model penalty is reproduced at about 49.5 ms, almost entirely in preprocessing. The single-question control instead saves about 61.7 ms outside the model despite zero KV reuse. This locates the discrepancy outside prefill and does not identify a causal M3 preprocessing instruction. The host/measurement mechanism remains unresolved; neither a positive request-only CI nor these opposite workload signs establishes KV computation acceleration.\n')
    bottlenecks=[]
    for w in ('single','multi'):
        runs=[r['summaries'][f'{w}_fixed_r{i}_reuse'] for i in range(3)]
        average=lambda key:float(np.mean([x[key]['mean'] for x in runs]))
        stage_means={label:float(np.mean([x['stage_spans'][label]['host_wall_ms']['mean'] for x in runs])) for label in ('m2_total','detection','map','m3_auto_total')}
        largest=max(stage_means,key=stage_means.get)
        text=f"{w}: largest measured model section is {largest}, host wall {stage_means[largest]:.3f} ms; M3 decode CUDA span {average('decode_cuda_ms'):.3f} ms; preprocessing wall {average('preprocessing_ms'):.3f} ms. Host sections and nested/overlapping CUDA spans are not added."
        bottlenecks.append(text);perf.append('\n'+text+'\n')
    perf.append('\nImage I/O/resize/normalization and CPU input preparation are separately profiled in `full_request_cpu_summary.json`; the profiler run is excluded from all headline measurements.\n')
    micro=r['planner_microbenchmark']['summary']
    perf.append(f"\nIsolated real-input planner wall: before {micro['before']['wall_ms']['mean']:.4f} ms; after {micro['after']['wall_ms']['mean']:.4f} ms. This diagnostic result is not an E2E claim. See per-profile records in `planner_microbenchmark.json`.\n")
    perf.append('\nDevelopment-only equal-token Zero/trial GPU control (per-question Runtime calls on two calibration scenes, three alternating rounds, unchanged real reference steps and T−1 actual decode forwards):\n')
    for length,metrics in r['development_fixed_control']['by_length'].items():
        perf.append(f"- Prefix {length}: actual prefill CUDA savings {metrics['prefill_cuda_ms_saved']['mean']:.3f} ms; complete M3 savings {metrics['m3_wall_ms_saved']['mean']:.3f} ms. Natural quality rejection remains binding. This is not independent acceptance or full-service speedup.\n")
    perf.append('\nCost scope: original natural calibration compares the native LM head on identical M1/M2 carriers with the complete M3 runtime. It is not the incremental Zero-KV comparison. Legacy rows contain planning-only cost recovery and can conservatively deduct preparation/signature again online; they are preserved and explicitly labelled. Updated calibration records the actual preparation/signature-plus-planning overhead so the fitter and Scheduler deduct it once. The separate same-runtime fixed-token Zero/trial controls provide the direct incremental measurement. No fitted policy is deployed from these diagnostic points.\n')
    (root/'performance_report.md').write_text('\n'.join(perf))
    quality=['# Quality evidence\n','Official unchanged evaluators on the declared 36-frame subset; these are not full-validation scores. Absolute score loss must be ≤ 0.10. Internal M3 quality budget remains 0.02. Negative loss means an improvement.\n',
        '| Workload | mAP loss | NDS loss | Lane DET_l loss | VLM Token F1 loss | Four budgets |',
        '|---|---:|---:|---:|---:|---|']
    for w,q in r['quality'].items():
        quality.append('| '+w+' | '+' | '.join(f"{q['system']['metrics'][k]['loss']:.6f}" for k in ('mAP','NDS','DET_l','Token_F1'))+f" | {q['system']['within_quality_budgets']} |")
    for w,q in r['quality'].items():
        quality.append(f'\n## {w}\nFull official per-scene metrics, populations and proxy error summaries are retained in `summary.json` and the official evaluator directories.\n')
        z=q['paired']['zero'];quality.append(f"M3-only mean Token F1 loss vs zero historical KV: {z['mean_f1_loss']:.6f}; changed answers {z['changed_answers']}; critical proxy regressions {z['critical_proxy_regressions']}. Native and candidate proxy errors are separately retained in `{w}_quality_pairs.json` and `summary.json`. Traffic assertions requiring review: {len(z['new_traffic_claims_needing_review'])}.\n")
        n=q['paired']['native'];unique_proxy=sum(p['critical_regressions'] for p in n['changed_repeat0_answers'])
        quality.append(f"Native→optimized spatial/numeric/motion proxy findings in repeat 0: {unique_proxy}; repeats do not increase independent evidence. Native baseline proxy errors: {n['baseline_proxy_errors']}; optimized proxy errors: {n['candidate_proxy_errors']}. These are parser findings, not independent human confirmation of new hard semantic failures. The overall semantic proxy gate remains conservative.\n")
    quality.append('Historical double-yellow/no-passing/lane-count hallucination remains a rejection. Its valid M1 state and 94.4% M2 refresh did not establish semantic safety. The development-only 128-visual-token trial does not remove that failure. Higher lexical F1 does not override it. Changed-answer evidence and traffic-assertion flags remain available for direct scene review; parser success alone is not semantic certification.\n')
    quality.append('New development failure: at `f38ef51177c84c2a9fa32584b79e601e`, the 163-token trial changed waiting pedestrians to people currently crossing, contradicting the unchanged annotation that no pedestrians are crossing. Token F1 improved by 0.0680 and automatic critical count was zero. This is preserved as a separate motion-semantic rejection in `calibration_semantic_review.json`, with all six image paths/SHA and state signals. It does not become a qualified Object Approx rule.\n')
    quality.append('Both the historical Map failure and the new crossing failure executed all 22 global M2 suffix blocks, skipped zero suffix blocks, and published a fresh global anchor (age after update 0). World-change scores were 0.7083 and 1.0, respectively. Freshness alone cannot reauthorize approximation; these sparse observations do not establish a reliable safe/unsafe threshold. Full state/refresh evidence and source SHA are in `semantic_state_attribution.json`.\n')
    (root/'quality_report.md').write_text('\n'.join(quality))
    report=['# M3 cross-frame / E2E experiment\n',f"Base commit: `{r['plan']['initial_commit']}`. Checkpoint: `ckpts/iter_10548.pth`; SHA and environment in worker invocation/configuration/checkpoint records and `environment.json`.\n",
        '## Acceptance\n```json\n'+json.dumps(d,indent=2)+'\n```\n',
        '## Direct answers\n',f"1. Independent natural cross-frame visual KV reuse: **{d['independent_cross_frame_visual_reuse']}**. Development trial reuse is reported separately and cannot satisfy acceptance.\n"]
    for w in ('single','multi'):
        u=r['summaries'][f'{w}_free_r0_reuse']['reuse']
        report.append(f"2. {w}: cross-frame KV {u['cross_frame_reused_tokens']} / {u['input_tokens']} input tokens ({100*u['cross_frame_reused_tokens']/u['input_tokens']:.3f}%); cross-frame visual tokens {u['cross_frame_visual_tokens']}, cross-frame requests {u['cross_frame_requests']}. All reuse {u['actually_reused_kv_tokens']} tokens ({100*u['skipped_fraction']:.3f}%); same-frame exact {u['same_frame_exact_tokens']}. See source/target/forward evidence in `summary.json`.\n")
    trials=[x for x in r['calibration'] if x['threshold']>0]
    report.append(f"Development-only natural trial: {sum(x['cross_frame_visual_requests'] for x in trials)} cross-frame visual-prefix requests, {sum(x['cross_frame_visual_reused_tokens'] for x in trials)} actually reused visual tokens. These observed development scenes do not count as independent acceptance. Natural motion-semantic failure and lack of a qualified cross-frame validation/cost working point prevent admission.\n")
    report.append('The 163-token trial prefix contains 35 text tokens and the first 128 Object latent-query tokens. It does not reuse Map KV merely because the question is labelled map_relations. Later Map KV cannot be stitched around a changed earlier Object prefix.\n')
    report += ['3. Actual Object/Ego/Map strict equality, normalized changes, M1/M2 state, cache age and 16/32/64 block endpoints are in `prefix_structure.json`. Native Object→Ego→Map order is retained. A changed Object prefix cannot be bypassed to reuse Map KV.\n',
        '4–5. Prefill, full M3, model and full request paired savings and confidence intervals are in `performance_report.md`. Module savings are not counted as complete service savings.\n',
        '6. Four official-subset score losses and native/new proxy/traffic errors are in `quality_report.md`; lexical improvements do not erase semantic failures.\n',
        '7. Non-model overhead is measured at explicit request/model boundaries. Remaining bottlenecks and individual stage spans are in `per_frame_summary.jsonl`; CPU profiler hotspots are in `cpu_profile_summary.json`. CUDA stream spans include host gaps and can overlap.\n',
        '8. Core changes: `ReusePlanner.plan` rejects impossible calibrated endpoints before RMS; `prefix_statistics` evaluates only policy-reachable full blocks; the offline `fit_policy` requires cross-frame visual validation for approximate costs and freezes cost ranges on calibration only. The measured old/new planner comparison uses the same real checkpoint/cache, fixed observed overhead and alternating GPU probes. `PrefillExecutor`, storage revision checks and M1/M2 algorithms are unchanged.\n',
        f"9. World-referenced cross-frame service acceleration claim supported: **{d['world_referenced_cross_frame_acceleration_claim_supported']}**.\n",
        '## Policy decision\nThe unique formal policy is retained. New calibration selects no admissible approximate rule with independent cross-frame evidence. Map/full-visual critical history is preserved. No min-token threshold, quality budget or generation length is relaxed to create hits.\n',
        '## Evidence\n`per_request_summary.csv/jsonl`, `per_frame_summary.jsonl`, `execution_audits.json`, `planner_equivalence.json`, `calibration_summary.json`, and SHA-indexed local tensors/profiles. No historical result was replaced. Reproduce with `REPRODUCE.md`.\n']
    report += ['\n## Measured remaining cost\n']+bottlenecks
    report.append(f"\nOnly the planner change has an isolated before/after measurement here: {micro['before']['wall_ms']['mean']:.4f} → {micro['after']['wall_ms']['mean']:.4f} ms on actual captured input/cache pairs. This does not establish cross-frame quality or E2E acceleration.\n")
    prefixes=read(root/'prefix_structure.json')
    equality={g:dict(equal_tokens=sum(x['groups'][g]['strictly_equal_tokens'] for x in prefixes),compared_tokens=sum(x['groups'][g]['tokens'] for x in prefixes)) for g in ('object','ego','map')}
    report.append(f"\nAcross {len(prefixes)} actual development cross-frame comparisons, strict LCP lengths are {dict(Counter(x['exact_prefix'] for x in prefixes))}. Exact visual token equality: {equality}. Mean carrier RMS: "+', '.join(f"{g} {np.mean([x['groups'][g]['relative_rms'] for x in prefixes]):.4f}" for g in ('object','ego','map'))+'. A stable world object is not an exact latent LLM KV slot. See `block_size_frontiers.json` for 16/32/64 raw risk frontiers; exact LCP is never block-rounded, and raw risk eligibility is not semantic approval.\n')
    report.append('\n## Validation history\nFinal GPU regression log is named by `regression_test_result.json`. Initial newly added test-fixture failures (CPU/CUDA device placement and a missing fixture field) remain in the earlier regression logs. The actual-checkpoint planner audit passed all 72 requests. Calibration semantic failures remain separate from passing automatic proxy scores.\n')
    (root/'experiment_report.md').write_text('\n'.join(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    analyze(Path(a.root))
