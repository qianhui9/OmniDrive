"""Frozen, development-only residual calibration and real paired M123 runs.

All shadow suffixes belong to the teacher run, never to measured serving runs.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr
from tools.worldstream_m2.global_residual_benchmark import read,rows,execute,evaluate,wait_gpu
from tools.worldstream_m2.global_residual_analysis import write,compare_quality,service_summary
from tools.worldstream_m123.latency_quality import validate_manifests as base_validate_manifests
from tools.worldstream_m123.benchmark.metrics import distribution


def validate_manifests(root):
    plan=base_validate_manifests(root)
    for split in ('development','validation'):
        manifest=read(root/(split+'_sequence.json'));scenes=set(manifest['scene_splits'])
        if scenes!=set(plan[split+'_scenes']) or scenes&set(plan['unseen_exclusions']):
            raise ValueError('Declared scene scope or old-scene exclusion violated')
    if hashlib.sha256((root/'profile_sequence.json').read_bytes()).hexdigest()!=plan['manifests']['profile']:
        raise ValueError('Frozen profile manifest changed')
    return plan


def correlation(x,y):
    if np.std(x)==0 or np.std(y)==0:return None
    return float(spearmanr(x,y).correlation)


def fit_parameters(x,y):
    scales=np.maximum(np.quantile(x,.75,axis=0),1e-6)
    target=max(float(np.quantile(y,.75)),1e-6)
    z=x/scales
    # Fixed lambda and uniform shrinkage prior; no heldout data or grid search.
    w,_=nnls(np.vstack([z,.1*np.eye(4)]),np.r_[y/target,.1*np.full(4,.25)])
    return scales,w,target


def fit(root):
    plan=validate_manifests(root)
    if (root/'risk_fit.json').exists():raise ValueError('Calibration is already frozen')
    records=[r for r in rows(root/'teacher/residual_labels.jsonl') if r['eligible_temporally']]
    if {r['scene_id'] for r in records}!=set(plan['development_scenes']):raise ValueError('Wrong teacher scenes')
    x=np.array([[r['global_drift'],r['local_tail']*(1+r['hot_fraction']),r['world_change'],r['age']/3] for r in records])
    y=np.array([r['residual_relative_rms'] for r in records])
    scales,w,target=fit_parameters(x,y);score=(x/scales)@w
    loo=np.zeros(len(records));folds=[]
    for scene in plan['development_scenes']:
        mask=np.array([r['scene_id']==scene for r in records]);s,weights,t=fit_parameters(x[~mask],y[~mask])
        loo[mask]=(x[mask]/s)@weights*t
        folds.append(dict(scene_id=scene,weights=weights.tolist(),scales=s.tolist(),
            old_rms_spearman=correlation(x[mask,0],y[mask]),risk_spearman=correlation(loo[mask],y[mask]),
            prediction_rmse=float(np.sqrt(np.mean((loo[mask]-y[mask])**2)))))
    diagnostics=[]
    for r,v,cv in zip(records,score,loo):diagnostics.append(dict(r,risk=float(v),out_of_scene_prediction=float(cv)))
    low=float(np.median(x[:,0]));high=float(np.quantile(y,.75))
    failures=[r for r in diagnostics if r['global_drift']<=low and r['residual_relative_rms']>=high]
    candidates=[]
    for i,q in enumerate((.40,.65)):
        config=dict(global_anchor_interval=3,global_drift_threshold=plan['old_rms_threshold'],
            global_risk_weights=w.tolist(),global_risk_scales=scales.tolist(),
            global_risk_threshold=float(np.quantile(score,q)),global_tail_fraction=.1)
        write(root/f'risk_{i}.json',config);candidates.append(dict(name=f'cal_risk{i}',quantile=q,config=config))
    write(root/'risk_fit.json',dict(samples=len(records),target='exact suffix residual relative RMS, scaled by development q75',
        ridge_lambda=.01,prior=[.25]*4,scales=scales.tolist(),weights=w.tolist(),target_scale=target,
        old_rms_spearman=correlation(x[:,0],y),fit_spearman=correlation(score,y),
        leave_scene_out_spearman=correlation(loo,y),leave_scene_out=folds,
        low_rms_cut=low,high_error_cut=high,low_rms_high_error=failures,candidates=candidates,
        label_sha256=hashlib.sha256((root/'teacher/residual_labels.jsonl').read_bytes()).hexdigest(),
        limitation='Correlated anchor pairs on 32 development frames; labels use teacher closed-loop H2, not candidate-specific future trajectories'))
    write(root/'residual_risk_pairs.json',diagnostics)
    print('FIT',w.tolist(),[p['config']['global_risk_threshold'] for p in candidates],flush=True)


def service(root,name,sequence,*,formal=False,risk=None,preload=False,repeat=0,extra=(),frozen=False):
    if (root/name).exists():raise ValueError('Preserve existing run '+name)
    wait_gpu();plan=read(root/'experiment_plan.json')
    command=['-m','tools.worldstream_m123.benchmark.frozen_worker','--snapshot-root',str(root)] if frozen else ['-m','tools.worldstream_m123.benchmark.worker']
    command+=['--output',str(root/name),'--run-root',str(root),'--condition','B3' if formal else 'B0',
        '--workload','language','--limit',str(len(read(sequence)['tokens'])),'--warmup','1','--seed','0',
        '--repeat',str(repeat),'--measurement-set','efficiency','--sequence-file',str(sequence)]
    if formal:command+=['--m2-global-anchor-interval','3','--m2-global-drift-threshold',str(plan['old_rms_threshold'])]
    if risk is not None:command+=['--m2-risk-config',str(risk)]
    if preload:command+=['--preload-inputs']
    execute(root,name,command+list(extra))


def timing(root,name,formal):
    frames=rows(root/name/'per_frame_metrics.jsonl');r=service_summary(root/name,formal)
    r['model_ms']=distribution(f['model_ms'] for f in frames)
    r['per_scene_model_ms']={s:distribution(f['model_ms'] for f in frames if f['scene_id']==s) for s in sorted({f['scene_id'] for f in frames})}
    return r


def develop(root):
    if not (root/'risk_fit.json').exists():fit(root)
    seq=root/'development_sequence.json'
    service(root,'cal_native',seq,preload=True);evaluate(root,'cal_native')
    service(root,'cal_rms',seq,formal=True,preload=True);evaluate(root,'cal_rms')
    native=timing(root,'cal_native',False);points=[]
    for i in range(2):
        name=f'cal_risk{i}';service(root,name,seq,formal=True,risk=root/f'risk_{i}.json',preload=True);evaluate(root,name)
        quality=compare_quality(root,'cal_native',name);t=timing(root,name,True)
        points.append(dict(name=name,risk_file=f'risk_{i}.json',quality=quality,timing=t))
    expected=read(root/'cal_native/prepared_input_hashes.json')
    for name in ('cal_rms','cal_risk0','cal_risk1'):
        if read(root/name/'prepared_input_hashes.json')!=expected:raise ValueError('Controlled input tensors differ')
    rms=dict(quality=compare_quality(root,'cal_native','cal_rms'),timing=timing(root,'cal_rms',True))
    write(root/'calibration_curve.json',dict(native=native,rms=rms,candidates=points,input_tensors_equal=True))
    eligible=[p for p in points if p['quality']['within_quality_budgets']]
    if not eligible:
        write(root/'status.json',dict(status='NO_DEVELOPMENT_CANDIDATE',formal_defaults_retained=True));return
    fastest=min(p['timing']['model_ms']['mean'] for p in eligible)
    shortlist=[p for p in eligible if p['timing']['model_ms']['mean']<=fastest*1.01]
    def stability(p):return max(t['mean']/native['per_scene_model_ms'][s]['mean'] for s,t in p['timing']['per_scene_model_ms'].items())
    best=min(shortlist,key=lambda p:(stability(p),p['timing']['model_ms']['mean'],p['name']))
    write(root/'selection.json',dict(name=best['name'],risk_file=best['risk_file'],config=read(root/best['risk_file']),
        validation_used=False,reason=read(root/'experiment_plan.json')['selection'],worst_scene_model_ratio=stability(best)))
    print('SELECTED',best['name'],flush=True)


def verify(root):
    validate_manifests(root);chosen=read(root/'selection.json');seq=root/'validation_sequence.json';risk=root/chosen['risk_file']
    # A: identical prepared CPU tensors. Only model_ms is a model-boundary time.
    for formal in (False,True):service(root,'controlled_'+('formal' if formal else 'native'),seq,formal=formal,risk=risk if formal else None,preload=True)
    if read(root/'controlled_native/prepared_input_hashes.json')!=read(root/'controlled_formal/prepared_input_hashes.json'):raise ValueError('Prepared tensor mismatch')
    # B: three alternating normal preprocessing pairs. Never alter policy after r0.
    for repeat in range(3):
        for formal in ((False,True) if repeat%2==0 else (True,False)):
            service(root,f'r{repeat}_'+('formal' if formal else 'native'),seq,formal=formal,risk=risk if formal else None,repeat=repeat)
        if repeat==0:
            evaluate(root,'r0_native');evaluate(root,'r0_formal')
            write(root/'independent_quality.json',compare_quality(root,'r0_native','r0_formal'))
    summarize(root)


def summarize(root):
    from tools.worldstream_m2.production_acceptance import validate_rows
    result=dict(selection=read(root/'selection.json'),quality=read(root/'independent_quality.json'),rounds=[],controlled={})
    for kind in ('native','formal'):result['controlled'][kind]=timing(root,'controlled_'+kind,kind=='formal')
    result['controlled']['model_speedup']=result['controlled']['native']['model_ms']['mean']/result['controlled']['formal']['model_ms']['mean']
    paired=[];audits={};expected=read(root/'validation_sequence.json')['tokens']
    for repeat in range(3):
        n=timing(root,f'r{repeat}_native',False);f=timing(root,f'r{repeat}_formal',True)
        result['rounds'].append(dict(native=n,formal=f,speedup=n['request_ms']['mean']/f['request_ms']['mean'],saved_ms=n['request_ms']['mean']-f['request_ms']['mean']))
        nn=rows(root/f'r{repeat}_native/per_frame_metrics.jsonl');ff=rows(root/f'r{repeat}_formal/per_frame_metrics.jsonl')
        if [r['frame_id'] for r in nn]!=expected or [r['frame_id'] for r in ff]!=expected:raise ValueError('Measured sample missing or reordered')
        audits[str(repeat)]=validate_rows(ff,require_reuse=False)
        if audits[str(repeat)]['status']!='PASS':raise ValueError('Single route execution audit failed')
        for r in ff:
            for key in ('global_risk_weights','global_risk_scales','global_risk_threshold'):
                if r['m2'][key]!=result['selection']['config'][key]:raise ValueError('Risk policy changed')
        paired.extend(dict(repeat=repeat,scene_id=a['scene_id'],frame_id=a['frame_id'],native_ms=a['request_ms'],formal_ms=b['request_ms'],saved_ms=a['request_ms']-b['request_ms']) for a,b in zip(nn,ff))
    result['aggregate']={}
    for kind in ('native','formal'):
        ff=[f for i in range(3) for f in rows(root/f'r{i}_{kind}/per_frame_metrics.jsonl')]
        qq=[q for i in range(3) for q in rows(root/f'r{i}_{kind}/per_question_metrics.jsonl')]
        result['aggregate'][kind]=dict(**{key:distribution(f[key] for f in ff) for key in ('request_ms','model_ms','preprocessing_ms','transfer_and_collate_ms')},
            throughput_fps=1000*len(ff)/sum(f['request_ms'] for f in ff),generated_tokens=sum(q['generated_tokens'] for q in qq),
            kv_reused_tokens=sum(q['reused_prefix_length'] for q in qq),unique_kv_requests=len({(q['frame_id'],q['question']) for q in qq if q['reused_prefix_length']}),
            peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in ff),peak_reserved_bytes=max(f['peak_gpu_reserved_bytes'] for f in ff),
            repeats_output_equal=all([f['output_digest'] for f in rows(root/f'r{i}_{kind}/per_frame_metrics.jsonl')]==[f['output_digest'] for f in rows(root/f'r0_{kind}/per_frame_metrics.jsonl')] for i in (1,2)),
            repeats_answers_equal=all([q['answer'] for q in rows(root/f'r{i}_{kind}/per_question_metrics.jsonl')]==[q['answer'] for q in rows(root/f'r0_{kind}/per_question_metrics.jsonl')] for i in (1,2)),
            stages={stage:distribution(f['stages'].get(stage,{}).get('gpu_span_ms',0.) for f in ff) for stage in sorted(set().union(*(f['stages'] for f in ff)))},
            per_scene={s:dict(request_ms=distribution(f['request_ms'] for f in ff if f['scene_id']==s),model_ms=distribution(f['model_ms'] for f in ff if f['scene_id']==s)) for s in sorted({f['scene_id'] for f in ff})})
    saved=np.array([r['saved_ms'] for r in result['rounds']]);margin=4.302652729911275*saved.std(ddof=1)/np.sqrt(3)
    result['paired_round_saved_ms_ci95']=dict(mean=float(saved.mean()),lower=float(saved.mean()-margin),upper=float(saved.mean()+margin),scope='3 repeats on the same 4 scenes, not 12 independent scenes')
    result['per_scene_saved_ms']={s:distribution(p['saved_ms'] for p in paired if p['scene_id']==s) for s in sorted({p['scene_id'] for p in paired})}
    result['positive_request_fraction']=sum(p['saved_ms']>0 for p in paired)/len(paired)
    result['service_speedup']=result['aggregate']['native']['request_ms']['mean']/result['aggregate']['formal']['request_ms']['mean']
    result['stable_net_speedup']=bool(np.all(saved>0) and saved.mean()-margin>0 and all(v['mean']>0 for v in result['per_scene_saved_ms'].values()))
    result['goal_met']=result['quality']['within_quality_budgets'] and result['stable_net_speedup']
    write(root/'formal_execution_audit.json',audits);write(root/'paired_requests.json',paired);write(root/'summary.json',result)
    write(root/'status.json',dict(status='PASS' if result['goal_met'] else 'GOAL_NOT_MET',quality_pass=result['quality']['within_quality_budgets'],stable_net_speedup=result['stable_net_speedup']))
    print('FINAL',result['service_speedup'],result['quality']['within_quality_budgets'],result['stable_net_speedup'],flush=True)


def controls(root):
    if not (root/'before_profile/status.json').exists():
        service(root,'before_profile',root/'profile_sequence.json',formal=True,frozen=True,
            extra=('--runtime-profile','--profile-frame','1','--capture-m3-inputs'))
    elif read(root/'before_profile/status.json')['status']!='COMPLETED':
        raise ValueError('Before profile is incomplete; preserve evidence and use a fresh destination')
    # Parent already owns the cross-job lock; do not reacquire it in the child.
    wait_gpu();execute(root,'decode_fixed_tokens',['-c','from pathlib import Path; from tools.worldstream_m123.latency_controls import decode_control; decode_control(Path('+repr(str(root))+'))'])
    wait_gpu();execute(root,'tests_gpu_final',['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
    execute(root,'m1_control_final',['-m','tools.worldstream_m123.latency_controls','--root',str(root),'--stage','m1'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['fit','develop','verify','summarize','controls'],required=True);a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():globals()[a.stage](Path(a.root))
