"""Fresh-scene, bounded-search M123 experiment; never used by serving."""
import argparse
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from tools.worldstream_m2.global_residual_benchmark import read, rows, service, evaluate, execute, wait_gpu
from tools.worldstream_m2.global_residual_analysis import compare_quality, service_summary, write, quality_latency_plot
from tools.worldstream_m123.benchmark.metrics import distribution


def validate_manifests(root):
    plan=read(root/'experiment_plan.json')
    scenes=[]
    for split in ('development','validation'):
        path=root/(split+'_sequence.json')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['manifests'][split]:
            raise ValueError('Frozen manifest changed: '+split)
        scenes.append(set(read(path)['scene_splits']))
    if scenes[0]&scenes[1]:raise ValueError('Development / heldout scene leakage')
    return plan


def per_scene_timing(root,name):
    ff=rows(root/name/'per_frame_metrics.jsonl')
    return {s:distribution(f['request_ms'] for f in ff if f['scene_id']==s) for s in sorted({f['scene_id'] for f in ff})}


def calibrate(root):
    plan=validate_manifests(root);seq=root/'development_sequence.json'
    service(root,'cal_native',seq);evaluate(root,'cal_native')
    native=service_summary(root/'cal_native',False);points=[]
    for ti,threshold in enumerate(plan['drift_thresholds']):
        for age in plan['anchor_candidates']:
            name='cal_t'+str(ti)+'_age'+str(age)
            service(root,name,seq,interval=age,threshold=threshold);evaluate(root,name)
            quality=compare_quality(root,'cal_native',name);timing=service_summary(root/name,True)
            points.append(dict(name=name,interval=age,threshold=threshold,quality=quality,timing=timing,
                per_scene_timing=per_scene_timing(root,name),service_speedup=native['request_ms']['mean']/timing['request_ms']['mean']))
    eligible=[p for p in points if p['quality']['within_quality_budgets']]
    write(root/'calibration_curve.json',dict(native=native,candidates=points))
    quality_latency_plot(root,points,native)
    if not eligible:
        write(root/'status.json',dict(status='NO_DEVELOPMENT_CANDIDATE',formal_defaults_retained=True))
        raise ValueError('No candidate passes; keep original formal defaults; do not tune on heldout')
    select(root)


def select(root):
    plan=validate_manifests(root)
    points=read(root/'calibration_curve.json')['candidates']
    eligible=[p for p in points if p['quality']['within_quality_budgets']]
    if not eligible:raise ValueError('No development candidate')
    fastest=min(p['timing']['request_ms']['mean'] for p in eligible)
    shortlist=[p for p in eligible if p['timing']['request_ms']['mean']<=fastest*1.01]
    native_scenes=per_scene_timing(root,'cal_native')
    def stability(p):
        return max(t['mean']/native_scenes[s]['mean'] for s,t in p['per_scene_timing'].items())
    best=min(shortlist,key=lambda p:(stability(p),p['timing']['request_ms']['mean'],p['interval']))
    write(root/'selection.json',dict(interval=best['interval'],threshold=best['threshold'],name=best['name'],
        validation_used=False,reason=plan['selection'],quality=best['quality'],worst_scene_latency_ratio=stability(best)))


def verify(root):
    validate_manifests(root);s=read(root/'selection.json');seq=root/'validation_sequence.json'
    for repeat in range(3):
        for formal in ((False,True) if repeat%2==0 else (True,False)):
            name='r'+str(repeat)+('_formal' if formal else '_native')
            service(root,name,seq,interval=s['interval'] if formal else None,threshold=s['threshold'],repeat=repeat)
        if repeat==0:
            evaluate(root,'r0_native');evaluate(root,'r0_formal')
            quality=compare_quality(root,'r0_native','r0_formal')
            write(root/'independent_quality.json',quality)
            # A failed quality result remains visible. Timing the same frozen
            # policy is diagnostic and cannot install or retune another policy.
    wait_gpu();execute(root,'tests_gpu_final',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
    wait_gpu();execute(root,'pure_visual',['-m','tools.worldstream_m2.window_validate','--wsgrr','--main-only','--formal','--snapshot-boundary',
        '--scene',read(root/'experiment_plan.json')['development_scenes'][0],'--frames','8','--repeats','3','--warmup','1',
        '--anchor-intervals',str(s['interval']),'--global-drift-threshold',str(s['threshold']),
        '--m1-predictions','outputs/wsgrr_mainfix_20261023/references/wsgrrr_20261022_followup/Native_full/official_eval/detection/pts_bbox/results_nusc.json',
        '--output',str(root/'pure_visual')])
    summarize(root)


def summarize(root):
    from tools.worldstream_m2.production_acceptance import validate_rows
    result=dict(selection=read(root/'selection.json'),independent_quality=compare_quality(root,'r0_native','r0_formal'),rounds=[])
    audits={}
    expected_tokens=read(root/'validation_sequence.json')['tokens']
    for i in range(3):
        n=service_summary(root/('r'+str(i)+'_native'),False);f=service_summary(root/('r'+str(i)+'_formal'),True)
        result['rounds'].append(dict(native=n,formal=f,speedup=n['request_ms']['mean']/f['request_ms']['mean'],net_saved_ms=n['request_ms']['mean']-f['request_ms']['mean']))
        name='r'+str(i)+'_formal';ff=rows(root/name/'per_frame_metrics.jsonl')
        for kind in ('native','formal'):
            actual=rows(root/f'r{i}_{kind}/per_frame_metrics.jsonl')
            if [f['frame_id'] for f in actual]!=expected_tokens:raise ValueError('Missing or reordered measured frames')
        audits[name]=validate_rows(ff,require_reuse=False)
        if any((r['m2']['global_anchor_interval'],r['m2']['global_drift_threshold'])!=(result['selection']['interval'],result['selection']['threshold']) for r in ff):raise ValueError('Policy changed')
    if any(a['status']!='PASS' for a in audits.values()):raise ValueError('Formal execution audit failed')
    result['aggregate']={}
    for kind in ('native','formal'):
        ff=[f for i in range(3) for f in rows(root/('r'+str(i)+'_'+kind)/'per_frame_metrics.jsonl')]
        qq=[q for i in range(3) for q in rows(root/('r'+str(i)+'_'+kind)/'per_question_metrics.jsonl')]
        result['aggregate'][kind]=dict(request_ms=distribution(f['request_ms'] for f in ff),
            throughput_fps=len(ff)*1000/sum(f['request_ms'] for f in ff),generated_tokens=sum(q['generated_tokens'] for q in qq),
            kv_reused_tokens=sum(q['reused_prefix_length'] for q in qq),peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),
            peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff),
            preprocessing_ms=distribution(f['preprocessing_ms'] for f in ff),
            transfer_and_collate_ms=distribution(f['transfer_and_collate_ms'] for f in ff),
            model_ms=distribution(f['model_ms'] for f in ff),
            output_transfer_ms=distribution(f['output_transfer_ms'] for f in ff),
            stages={stage:distribution(f['stages'].get(stage,{}).get('gpu_span_ms',0.) for f in ff) for stage in sorted(set().union(*(f['stages'] for f in ff)))},
            per_scene={s:dict(request_ms=distribution(f['request_ms'] for f in ff if f['scene_id']==s)) for s in sorted({f['scene_id'] for f in ff})},
            repeats_output_equal=all([f['output_digest'] for f in rows(root/('r'+str(i)+'_'+kind)/'per_frame_metrics.jsonl')]==[f['output_digest'] for f in rows(root/('r0_'+kind)/'per_frame_metrics.jsonl')] for i in (1,2)),
            repeats_answers_equal=all([q['answer'] for q in rows(root/('r'+str(i)+'_'+kind)/'per_question_metrics.jsonl')]==[q['answer'] for q in rows(root/('r0_'+kind)/'per_question_metrics.jsonl')] for i in (1,2)))
    saved=np.array([r['net_saved_ms'] for r in result['rounds']]);margin=4.302652729911275*saved.std(ddof=1)/np.sqrt(3)
    result['paired_round_saved_ms_ci95']=dict(mean=float(saved.mean()),lower=float(saved.mean()-margin),upper=float(saved.mean()+margin),scope='3 repeats of the same 24 frames in 4 scenes, t(df=2); not 12 independent scenes')
    result['service_speedup']=result['aggregate']['native']['request_ms']['mean']/result['aggregate']['formal']['request_ms']['mean']
    paired=[]
    for i in range(3):
        nn=rows(root/('r'+str(i)+'_native')/'per_frame_metrics.jsonl')
        ff=rows(root/('r'+str(i)+'_formal')/'per_frame_metrics.jsonl')
        if [(f['scene_id'],f['frame_id']) for f in nn]!=[(f['scene_id'],f['frame_id']) for f in ff]:raise ValueError('Unpaired frames')
        paired += [dict(repeat=i,scene_id=n['scene_id'],frame_id=n['frame_id'],native_ms=n['request_ms'],formal_ms=f['request_ms'],saved_ms=n['request_ms']-f['request_ms']) for n,f in zip(nn,ff)]
    scene_saved={s:distribution(p['saved_ms'] for p in paired if p['scene_id']==s) for s in sorted({p['scene_id'] for p in paired})}
    result['per_scene_paired_saved_ms']=scene_saved
    result['normal_negative_samples']=sum(p['saved_ms']<0 for p in paired)
    result['stable_net_speedup']=bool(np.all(saved>0) and saved.mean()-margin>0 and all(s['mean']>0 for s in scene_saved.values()))
    write(root/'paired_requests.json',paired)
    result['quality_pass']=result['independent_quality']['within_quality_budgets']
    result['quality_latency_goal_met']=result['quality_pass'] and result['stable_net_speedup']
    result['deployment_decision']=read(root/'deployment_decision.json') if (root/'deployment_decision.json').exists() else None
    write(root/'formal_execution_audit.json',audits);write(root/'summary.json',result)
    write(root/'status.json',dict(status='QUALITY_PASS' if result['quality_pass'] else 'QUALITY_FAIL',implementation_complete=True,
        stable_net_speedup=result['stable_net_speedup'],quality_latency_goal_met=result['quality_latency_goal_met'],
        experimental_policy=result['selection'],deployment_decision=result['deployment_decision']))
    print('FINAL',result['service_speedup'],result['quality_pass'],flush=True)


def finish(root):
    select(root)
    verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['calibrate','select','verify','summarize','finish'],required=True)
    a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():
        {'calibrate':calibrate,'select':select,'verify':verify,'summarize':summarize,'finish':finish}[a.stage](Path(a.root))
