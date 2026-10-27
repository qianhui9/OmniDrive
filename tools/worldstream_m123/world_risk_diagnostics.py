"""Offline attribution on exact teacher labels and unchanged SDK GT matches."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from tools.worldstream_m2.global_residual_benchmark import read,rows
from tools.worldstream_m2.global_residual_analysis import write
from tools.worldstream_m123.world_risk_experiment import correlation
from tools.worldstream_m123.benchmark.metrics import distribution


def per_frame_detection(root,name):
    directory=root/(name+'_official')/'detection_matches_2m'
    gt={};tp=Counter();fp=Counter()
    for r in rows(directory/'matched_objects.jsonl'):
        gt.setdefault(r['frame_id'],set()).add(r['gt_filtered_index']);tp[r['frame_id']]+=1
    for r in rows(directory/'unmatched_ground_truth.jsonl'):gt.setdefault(r['frame_id'],set()).add(r['gt_filtered_index'])
    for r in rows(directory/'prediction_status.jsonl'):fp[r['frame_id']]=sum(p['matched'] is False for p in r['predictions'])
    return {token:dict(gt=len(indices),tp=tp[token],fp=fp[token],recall_2m=tp[token]/len(indices) if indices else None) for token,indices in gt.items()}


def simulate(root,config):
    teacher=rows(root/'teacher/per_frame_metrics.jsonl');labels=rows(root/'teacher/residual_labels.jsonl')
    by_pair={(r['frame_id'],r['anchor_frame_id']):r for r in labels}
    result=[];anchor=None;age=0
    for f in teacher:
        if f['explicit_scene_memory_reset']:anchor=None;age=0
        age+=1;label=by_pair.get((f['frame_id'],anchor)) if anchor else None
        score=None;without_world=None
        if label:
            x=np.array([label['global_drift'],label['local_tail']*(1+label['hot_fraction']),label['world_change'],age/3])
            contributions=x/np.array(config['global_risk_scales'])*config['global_risk_weights']
            score=float(contributions.sum());without_world=score-float(contributions[2])
        refresh=(anchor is None or age>=3 or f['m2']['global_anchor_refresh_reason']=='m1_full_refresh' or score>=config['global_risk_threshold'])
        result.append(dict(frame_id=f['frame_id'],scene_id=f['scene_id'],age=age if anchor else 0,
            update=refresh,risk=score,risk_without_world=without_world,
            world_changed_threshold_decision=bool(score is not None and age<3 and f['m2']['global_anchor_refresh_reason']!='m1_full_refresh' and without_world<config['global_risk_threshold']<=score),
            residual_relative_rms=0. if refresh else label['residual_relative_rms'],
            final_relative_rms=0. if refresh else label['final_relative_rms'],label=label))
        if refresh:anchor=f['frame_id'];age=0
    reused=[r for r in result if not r['update']]
    return dict(frames=result,summary=dict(reuse_frames=len(reused),suffix_blocks_skipped=22*len(reused),
        world_changed_threshold_frames=sum(r['world_changed_threshold_decision'] for r in result),
        all_frame_final_relative_rms=distribution(r['final_relative_rms'] for r in result),
        reused_frame_final_relative_rms=distribution(r['final_relative_rms'] for r in reused)))


def analyze(root):
    fit=read(root/'risk_fit.json');plan=read(root/'experiment_plan.json')
    old=dict(global_risk_weights=[1,0,0,0],global_risk_scales=[1]*4,
        global_risk_threshold=float(np.nextafter(plan['old_rms_threshold'],np.inf)))
    simulations={'rms':simulate(root,old)}
    for i in range(2):simulations['risk'+str(i)]=simulate(root,read(root/f'risk_{i}.json'))
    native=per_frame_detection(root,'cal_native');rms=per_frame_detection(root,'cal_rms')
    teacher=rows(root/'teacher/per_frame_metrics.jsonl');clean=rows(root/'cal_rms/per_frame_metrics.jsonl')
    teacher_equal=[f['output_digest'] for f in teacher]==[f['output_digest'] for f in clean]
    ff={f['frame_id']:f for f in clean};labels={(r['frame_id'],r['anchor_frame_id']):r for r in rows(root/'teacher/residual_labels.jsonl')}
    pairs=[]
    for token,b in native.items():
        f=ff[token];a=rms[token];anchor=f['m2']['global_anchor_frame_id'];label=labels.get((token,anchor))
        pairs.append(dict(frame_id=token,scene_id=f['scene_id'],native=b,rms=a,
            recall_loss=b['recall_2m']-a['recall_2m'] if b['recall_2m'] is not None else None,
            reused=f['m2']['approx_global_residual_reuse'],exact_label=label))
    reused=[p for p in pairs if p['exact_label'] and p['recall_loss'] is not None]
    correlations={k:correlation([p['exact_label'][k] for p in reused],[p['recall_loss'] for p in reused]) for k in ('global_drift','local_tail','world_change','residual_relative_rms','final_relative_rms')}
    candidates={};world_contribution={}
    for i in range(2):
        actual=per_frame_detection(root,f'cal_risk{i}')
        candidates['risk'+str(i)]=[dict(frame_id=p['frame_id'],scene_id=p['scene_id'],native=p['native'],rms=p['rms'],risk=actual[p['frame_id']],
            rms_recall_loss=p['recall_loss'],risk_recall_loss=p['native']['recall_2m']-actual[p['frame_id']]['recall_2m'] if p['native']['recall_2m'] is not None else None) for p in pairs]
        decisions=[]
        for f in rows(root/f'cal_risk{i}/per_frame_metrics.jsonl'):
            d=f['m2'];score=d['global_risk_score']
            if score is None or d['global_anchor_age']>=3 or d['global_anchor_refresh_reason']=='m1_full_refresh':continue
            contribution=d['global_risk_weights'][2]*d['global_risk_normalized'][2]
            decisions.append(dict(frame_id=f['frame_id'],score=score,world_contribution=contribution,
                world_changed_threshold_decision=score-contribution<d['global_risk_threshold']<=score))
        world_contribution['risk'+str(i)]=dict(decisions=decisions,threshold_decisions_changed=sum(r['world_changed_threshold_decision'] for r in decisions),
            scope='Same actual candidate H2/anchor/state, zero world term counterfactual only; no alternative online pass or rollout')
    result=dict(teacher_clean_predictions_equal=teacher_equal,simulations=simulations,development_gt_pairs=pairs,candidate_gt_pairs=candidates,world_contribution=world_contribution,
        residual_vs_detection_recall_correlations=correlations,
        high_error_frames=sorted([p for p in pairs if p['exact_label']],key=lambda p:p['exact_label']['final_relative_rms'],reverse=True)[:8],
        scope='Exact-suffix labels on the same teacher H2 isolate anchor selection; candidate live trajectories are evaluated separately with real task outputs. Per-frame 2m recall/FP are unchanged SDK matching counts, not per-frame AP/NDS.',
        low_rms_high_error_cases=fit['low_rms_high_error'])
    write(root/'residual_and_task_diagnostics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(root/'residual_risk_pairs.json');fig,ax=plt.subplots(1,2,figsize=(10,4))
    for i,key in enumerate(('global_drift','risk')):
        for scene in plan['development_scenes']:
            q=[p for p in r if p['scene_id']==scene];ax[i].scatter([p[key] for p in q],[p['residual_relative_rms'] for p in q],s=20,label=scene[:6])
        ax[i].set(xlabel=key,ylabel='Exact suffix residual relative RMS');ax[i].grid(alpha=.2)
    ax[1].legend(title='Development scene',fontsize=8);fig.tight_layout()
    fig.savefig(root/'residual_risk.png',dpi=180);fig.savefig(root/'residual_risk.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();analyze(Path(a.root))
